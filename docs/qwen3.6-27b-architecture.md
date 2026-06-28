# Qwen3.6-27B — Model Architecture Reference

> Status: reference document (implementation guide). Date: 2026-06-25.
> Purpose: a complete, implementation-grade description of the **Qwen3.6-27B** text decoder —
> every layer, every parameter, every operator, the exact computation order, and the fusion
> opportunities — so we can implement kernels one by one and hand-write the model card.
>
> Sources (cross-checked): the local checkpoint `config.json` + safetensors headers; vLLM
> (`qwen3_5.py`, `qwen3_next.py`, GDN/FLA ops, `layernorm.py`, `mrope.py`); llama.cpp
> (`LLM_ARCH_QWEN35`, `src/models/qwen35.cpp`, `convert_hf_to_gguf.py`); and primary papers
> (Gated DeltaNet 2412.06464, DeltaNet 2406.06484, DeepSeek-V3 MTP 2412.19437, Qwen2-VL MRoPE
> 2409.12191, Gated Attention 2505.06708, Gemma 2403.08295). Per-claim provenance is in §13.

---

## 1. Overview & provenance

`Qwen3.6-27B` is internally the **`qwen3_5`** architecture (HF class
`Qwen3_5ForConditionalGeneration`), a member of the **Qwen3-Next** family. It is a **hybrid
linear/full-attention, DENSE** language model (plus a vision tower and an MTP head that are
out of v1 scope; see the project [`design.md`](design.md)).

Key identity facts that drive every design decision:

- **Hybrid token mixer, 3:1.** 64 decoder layers; layer `i` is **full attention iff
  `(i+1) % 4 == 0`**, else **Gated-DeltaNet (GDN) linear attention** ⇒ **48 GDN + 16 full**.
- **DENSE MLP.** Every layer's FFN is a plain **SwiGLU** MLP (`gate/up/down_proj`,
  intermediate 17408). This is *not* the MoE variant (`qwen3_5_moe`); there are no experts or
  router tensors in this checkpoint. (The public Qwen3-Next-80B-A3B is MoE; ignore its
  512-expert details — they do not apply here.)
- **Gated everything.** Full-attention has a sigmoid **output gate**; GDN has a SiLU **output
  gate**; both feed an output projection.
- **(1+w) RMSNorm** (Gemma/zero-centered) for all layer norms and QK-norms — *except* the GDN
  internal gated norm, which is plain (ones-init) RMSNorm.
- **MTP** (1 layer) ships for self-speculative decoding; **deferred** (see §6.8 / design.md).

llama.cpp arch enum: **`LLM_ARCH_QWEN35`** (`"qwen35"`), graph in `src/models/qwen35.cpp`
(`LLM_TYPE_27B` at `n_layer==64`). Do **not** confuse with `qwen3next` (MoE lineage).

---

## 2. Global hyperparameters

From `text_config` in `config.json`:

| Symbol | Field | Value |
|---|---|---|
| `D` | hidden_size | **5120** |
| `L` | num_hidden_layers | **64** (48 GDN + 16 full) |
| `I` | intermediate_size (SwiGLU) | **17408** |
| — | hidden_act | `silu` |
| — | rms_norm_eps `ε` | **1e-6** |
| `V` | vocab_size | **248320** |
| — | tie_word_embeddings | **false** (separate `lm_head`) |
| — | max_position_embeddings | **262144** (model capacity; current M2.8 gate `max_ctx = 8192`, 128K/256K deferred) |
| — | torch_dtype | bfloat16 |
| — | full_attention_interval | 4 |
| **Full attention** | | |
| `Hq` | num_attention_heads | **24** |
| `Hkv` | num_key_value_heads | **4** (GQA group = 6) |
| `dh` | head_dim | **256** |
| — | attn_output_gate | **true** |
| — | partial_rotary_factor | **0.25** → rotary_dim **64** |
| — | rope_theta | **1e7** |
| — | mrope_section | **[11, 11, 10]** (interleaved) |
| — | attention scale | `1/sqrt(256) = 0.0625` |
| **GDN linear attention** | | |
| `Hk` | linear_num_key_heads | **16** (q & k heads) |
| `Hv` | linear_num_value_heads | **48** (v heads) |
| `dk` | linear_key_head_dim | **128** |
| `dv` | linear_value_head_dim | **128** |
| `r` | GVA ratio `Hv/Hk` | **3** |
| `W` | linear_conv_kernel_dim | **4** |
| — | output_gate_type | `swish` → SiLU |
| — | mamba_ssm_dtype | **float32** |
| **MTP** | mtp_num_hidden_layers | 1 (deferred) |

Derived GDN dims: `key_dim = Hk·dk = 2048`, `value_dim = Hv·dv = 6144`,
`conv_dim = 2·key_dim + value_dim = 10240`.
Full-attn dims: `q_size = Hq·dh = 6144`, `kv_size = Hkv·dh = 1024`,
`q_proj_out = Hq·dh·(1+gate) = 12288`.

---

## 3. Layer schedule

```
i      : 0  1  2  3 | 4  5  6  7 | ... | 60 61 62 63
type   : G  G  G  F  | G  G  G  F  | ... | G  G  G  F
                  ^full                            ^full
```
`F` (full attention) at i = 3, 7, 11, …, 63 → **16 layers**. `G` (GDN) elsewhere → **48
layers**. All 64 layers share the same MLP and norm structure; only the token mixer differs.

---

## 4. Tensor / weight inventory (exact shapes, bf16 on disk)

PyTorch `[out, in]` convention. Prefix `model.language_model.` omitted for layer tensors.
Counts verified against the 15 safetensors shards (1199 tensors total; 333 are the vision
tower, excluded here).

### 4.1 Global (non-layer)

| Tensor | Shape | Notes |
|---|---|---|
| `embed_tokens.weight` | `[248320, 5120]` | row gather at input |
| `norm.weight` | `[5120]` | final RMSNorm, **(1+w)** |
| `lm_head.weight` | `[248320, 5120]` | untied; read every decode step |

### 4.2 Per **GDN** layer (×48) — prefix `layers.{i}.`

| Tensor | Shape | Role |
|---|---|---|
| `input_layernorm.weight` | `[5120]` | pre-mixer RMSNorm **(1+w)** |
| `linear_attn.in_proj_qkv.weight` | `[10240, 5120]` | → Q(2048)‖K(2048)‖V(6144) |
| `linear_attn.in_proj_z.weight` | `[6144, 5120]` | output gate z (V-shaped) |
| `linear_attn.in_proj_b.weight` | `[48, 5120]` | β pre-activation (per V-head) |
| `linear_attn.in_proj_a.weight` | `[48, 5120]` | α pre-activation (per V-head) |
| `linear_attn.A_log` | `[48]` | log-decay base; stored raw, runtime computes `-exp(A_log)` |
| `linear_attn.dt_bias` | `[48]` | softplus bias (per V-head) |
| `linear_attn.conv1d.weight` | HF raw `[10240, 1, 4]`; q5090 canonical `[10240, 4, 1]` | depthwise causal conv over `[q;k;v]` |
| `linear_attn.norm.weight` | `[128]` | gated RMSNorm, **plain w (NO +1)** |
| `linear_attn.out_proj.weight` | `[5120, 6144]` | V-space → hidden |
| `post_attention_layernorm.weight` | `[5120]` | pre-MLP RMSNorm **(1+w)** |
| `mlp.gate_proj.weight` | `[17408, 5120]` | SwiGLU gate |
| `mlp.up_proj.weight` | `[17408, 5120]` | SwiGLU up |
| `mlp.down_proj.weight` | `[5120, 17408]` | SwiGLU down |

### 4.3 Per **full-attention** layer (×16) — prefix `layers.{i}.`

| Tensor | Shape | Role |
|---|---|---|
| `input_layernorm.weight` | `[5120]` | pre-mixer RMSNorm **(1+w)** |
| `self_attn.q_proj.weight` | `[12288, 5120]` | per-head **interleaved** `[q(256)\|gate(256)]` ×24 — reshape `[T,24,512]`, split last dim (**NOT** two contiguous 6144 blocks) |
| `self_attn.k_proj.weight` | `[1024, 5120]` | K = 4×256 |
| `self_attn.v_proj.weight` | `[1024, 5120]` | V = 4×256 |
| `self_attn.q_norm.weight` | `[256]` | per-head QK-norm **(1+w)** |
| `self_attn.k_norm.weight` | `[256]` | per-head QK-norm **(1+w)** |
| `self_attn.o_proj.weight` | `[5120, 6144]` | attn out → hidden |
| `post_attention_layernorm.weight` | `[5120]` | pre-MLP RMSNorm **(1+w)** |
| `mlp.{gate,up,down}_proj.weight` | same as GDN | dense SwiGLU |

### 4.4 MTP module (deferred) — prefix `mtp.`

| Tensor | Shape | Role |
|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | RMSNorm on next-token embedding |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | RMSNorm on carried hidden state |
| `mtp.fc.weight` | `[5120, 10240]` | `M`: concat(emb,hidden) 2D→D |
| `mtp.layers.0.*` | full-attn decoder layer | q/k/v/o, q_norm/k_norm, in/post norms, dense MLP |
| `mtp.norm.weight` | `[5120]` | final RMSNorm |

MTP shares the main `embed_tokens` and `lm_head` (`mtp_use_dedicated_embeddings=false`); no
`mtp.embed_tokens` / `mtp.lm_head` in the checkpoint.

---

## 5. Numerics & RMSNorm conventions (read before writing kernels)

| Quantity | Precision / convention |
|---|---|
| Master activations / matmuls | bf16 (W4A16 for linear layers; see design.md) |
| **Layer RMSNorm** (input, post-attn, final) | **`x/rms(x) · (1 + w)`**, variance in **fp32**, `ε=1e-6` |
| **QK-norm** (q_norm, k_norm) | same **(1+w)** RMSNorm, over `dh=256` per head, fp32 |
| **GDN gated norm** (`linear_attn.norm`) | **plain `x/rms(x) · w`** (ones-init, **NO +1**), fp32, over `dv=128`, then `· SiLU(z)` |
| GDN `A_log`, gates `g`,`β` | **fp32** (`g=-exp(A_log)·softplus(a+dt_bias)`) |
| GDN recurrent / chunk math | accumulate in **fp32**; SSM state stored **fp32** |
| q,k L2-norm (GDN) | fp32, `ε=1e-6` |
| Softmax / reductions (full attn) | fp32 accumulate; scale `1/sqrt(256)` |
| KV cache | bf16 (v1) |

> **The single most error-prone detail:** *every* RMSNorm uses `(1+w)` **except** the GDN
> `linear_attn.norm` (plain `w`, ones-init). **Authoritative source = the Python packer.**
> Unlike llama.cpp — whose converter folds `+1` into all `*norm.weight` except
> `linear_attn.norm.weight` — our offline packer stores **all norm weights verbatim**
> (`tools/q5090_convert/layouts.py::encode_contiguous` is a pure dtype cast, no math). The
> `+1` is therefore **NOT** baked into the file: the **runtime kernel** applies the convention
> — `(1+w)` for the input/post/final and q/k norms, plain `w` for the GDN gated norm. The same
> rule holds for every other model-semantic transform: `A_log` is stored raw and the runtime
> computes `g = -exp(A_log)·softplus(a+dt_bias)`. See `design.md` §10 (packer folds nothing).

---

## 6. Component computations (ordered)

Notation: `x` = hidden state `[T, D]` (T = tokens this step: prefill T = prompt len, decode
T = 1). `⊙` = elementwise. All projections are `y = x · Wᵀ` (W4A16).

### 6.1 Embedding
```
x[t] = embed_tokens[ input_ids[t] ]            # gather row, [T, 5120] bf16
```

### 6.2 RMSNorm variants
```
gemma_rmsnorm(x, w):     # input/post/final, q_norm, k_norm
    v   = mean(x.float()^2, axis=-1)            # fp32
    xn  = x.float() * rsqrt(v + 1e-6)
    return (xn * (1.0 + w.float())).to(bf16)

gated_rmsnorm(o, z, w):  # GDN output only (per v-head, dv=128)
    v   = mean(o.float()^2, axis=-1)
    on  = o.float() * rsqrt(v + 1e-6)
    return (on * w.float() * silu(z.float())).to(bf16)   # plain w, "norm before gate"
```

### 6.3 Decoder layer skeleton (both types, pre-norm + residual)
```
# entering with running residual stream `x`
h        = gemma_rmsnorm(x, input_layernorm.w)
mix      = full_attention(h)  OR  gdn_linear_attention(h)      # §6.4 / §6.5
x        = x + mix                                             # residual add
h        = gemma_rmsnorm(x, post_attention_layernorm.w)
x        = x + swiglu_mlp(h)                                   # §6.6
# (vLLM/llama.cpp implement this as fused add+norm carrying `residual` separately;
#  mathematically identical to the two `x += …` above.)
```
After all 64 layers: `x = gemma_rmsnorm(x, norm.w)`; then `logits = x · lm_headᵀ`.

### 6.4 Full attention (gated GQA + QK-norm + partial MRoPE)
```
qkv      = [ q_proj(h)(=12288), k_proj(h)(=1024), v_proj(h)(=1024) ]
q_gate   = reshape(q_proj_out, [T, Hq=24, 2*dh=512])
q, gate  = split(q_gate, [256, 256], axis=-1)                 # per head: [query | gate]
q        = gemma_rmsnorm(q,  q_norm.w)        # per head over dh=256
k        = gemma_rmsnorm(k,  k_norm.w)        # per head over dh=256
q, k     = rope_partial_mrope(positions, q, k)  # rotate first 64 dims; §8
attn     = softmax( (q · kᵀ)/sqrt(256) + causal_mask ) · v     # GQA: 24 q-heads, 4 kv-heads
                                                               # (each kv-head shared by 6 q)
attn     = attn ⊙ sigmoid(gate)                                # output gate, [T, 6144]
out      = o_proj(attn)                                        # [T, 5120]
```
Decode appends the new `k,v` (4 heads × 256) to the KV cache and attends over the cached
window. Prefill is a causal flash-attention over the whole prompt.

### 6.5 GDN linear attention (Gated DeltaNet)
```
# --- projections ---
qkv      = in_proj_qkv(h)          # [T, 10240]
z        = in_proj_z(h)            # [T, 6144] -> reshape [T, Hv=48, dv=128]
b        = in_proj_b(h)            # [T, 48]
a        = in_proj_a(h)            # [T, 48]

# --- short causal conv (depthwise, k=4) + SiLU, on [q;k;v] only ---
qkv      = silu( causal_conv1d(qkv, conv1d.w) )       # [T, 10240]; uses/updates conv_state
q        = reshape(qkv[:,      0:2048 ], [T, 16, 128])
k        = reshape(qkv[:,   2048:4096 ], [T, 16, 128])
v        = reshape(qkv[:,   4096:10240], [T, 48, 128])

# --- gates (fp32) ---
g        = -exp(A_log) * softplus(a + dt_bias)        # [T, 48]  (log-decay, log alpha_t)
beta     = sigmoid(b)                                 # [T, 48]

# --- q/k L2 norm (per head, dk=128) ---
q        = l2norm(q); k = l2norm(k)                   # then q scaled by 1/sqrt(128) in kernel

# --- recurrence ---
if decode (T==1):  o, state = gated_delta_recurrent(q,k,v,g,beta, state)   # §7.1
else (prefill):    o, state = gated_delta_chunked  (q,k,v,g,beta, state)   # §7.2
# q,k have 16 heads, v has 48; head h_v uses k/q head (h_v // 3)  (GVA, repeat by 3)

# --- gated output norm + projection ---
o        = gated_rmsnorm(o, z, norm.w)                # [T, 48, 128] -> RMSNorm(o)⊙SiLU(z)
out      = out_proj( flatten(o) )                     # [T, 6144] -> [T, 5120]
```

### 6.6 MLP (dense SwiGLU)
```
out = down_proj( silu(gate_proj(h)) ⊙ up_proj(h) )    # [T, D] -> [T, I] -> [T, D]
```

### 6.7 Final norm + lm_head
```
x      = gemma_rmsnorm(x, norm.w)                     # [T, 5120]
logits = x · lm_headᵀ                                 # [T, 248320]
next   = argmax(logits[-1])                           # greedy (v1)
```

### 6.8 MTP (deferred — self-speculative draft)
```
emb = gemma_rmsnorm(embed_tokens[next_id], pre_fc_norm_embedding.w)
hid = gemma_rmsnorm(last_hidden_state,      pre_fc_norm_hidden.w)
h   = fc( concat(emb, hid) )                          # [.,10240] -> [.,5120]
h   = full_attention_decoder_layer(h)                 # mtp.layers.0 (full attn + dense MLP)
h   = gemma_rmsnorm(h, mtp.norm.w)
draft_logits = h · lm_headᵀ                           # shared lm_head
```

---

## 7. Gated Delta Rule mathematics

Per head, state `S ∈ R^{dv×dk}` (= `[128,128]`, `state_v_first` layout). `α_t = exp(g_t)`
is a per-(token, v-head) **scalar** decay; `β_t = σ(b_t)` is the write strength.

### 7.1 Recurrent form (decode, T=1) — preferred CUDA kernel
```
# inputs per v-head: q,k (dk=128, L2-normed; from k-head h//3), v (dv=128), g, beta (scalars)
S   *= exp(g)                       # decay whole state
Sk   = (S * k[None, :]).sum(-1)     # S @ k          -> [dv]
u    = beta * (v - Sk)              # delta / pseudo-value -> [dv]
S   += u[:, None] * k[None, :]      # rank-1 write   S += u kᵀ
o    = (S * (q * dk**-0.5)[None,:]).sum(-1)   # o = S @ (q/sqrt(dk)) -> [dv]
```
Equivalent closed form: `S_t = α_t·S_{t-1}(I − β_t k_tk_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t (q_t/√dk)`.

### 7.2 Chunked parallel form (prefill, T>1) — chunk size C=64
Within a chunk, build the decay-aware causal mask `Γ` from the cumulative log-decay
`ĝ = cumsum(g)` (`Γ_ij = exp(ĝ_i − ĝ_j)` for `i≥j`), then use the **WY / UT transform** to
turn the sequential delta solves into matmuls + one lower-triangular inverse:
```
K_β = diag(β)·K
T   = (I − tril(K_β·Kᵀ ⊙ Γ, -1))^{-1}     # UT transform (forward-substitution)
W   = T·K_β ;   U = T·(diag(β)·V)          # decay-aware WY vectors
# inter-chunk scan over chunks c, carrying state S:
for c:
    v_new = U[c] − (K_cumdecay[c] @ S)
    O[c]  = (Q[c]·diag(exp(ĝ))) @ S + ( (Q[c]·K[c]ᵀ ⊙ Γ_c) @ v_new )
    S     = exp(ĝ_last)·S + (K[c] ⊙ exp(ĝ_last−ĝ))ᵀ @ v_new
```
This is the byte-exact spec of FLA `chunk_gated_delta_rule` / HF
`torch_chunk_gated_delta_rule`. **Numerical guard:** the reference clamps the cumulative
log-decay (`ĝ_last` and `ĝ_last − ĝ`) at `max 50` before `exp`; since `g ≤ 0` here it is
effectively a no-op, but include it to match the reference exactly. Decode uses §7.1; any
`seq_len>1` uses §7.2.

### 7.3 Convolution & gate details
- Conv is **depthwise, causal, kernel 4, no bias**, over the **10240 `[q;k;v]` channels
  only** (not `z`, `a`, `b`); activation **SiLU**. Decode keeps a rolling conv state of width
  `W−1 = 3`.
- `g = -exp(A_log)·softplus(a + dt_bias)` (numerically: softplus guarded at threshold 20),
  `β = σ(b)` — both per v-head, fp32. (FLA carries `g` in log-space; the FlashInfer path
  passes `exp(g)`.)

---

## 8. Partial RoPE + MRoPE

- **Partial:** `rotary_dim = floor(dh · 0.25) = 64`. Only the first 64 of 256 head dims are
  rotated (NeoX/GPT-NeoX split: `rotate_half(x) = [-x2, x1]`); the remaining **192 pass
  through unrotated**. `θ_j = pos · 1e7^(-2j/64)`.
- **QK-norm before RoPE:** order is `q_norm(q)` / `k_norm(k)` → RoPE.
- **MRoPE (interleaved):** `mrope_section = [11,11,10]` partitions the `rotary_dim/2 = 32`
  frequency pairs across (temporal, height, width) position channels; `sum = 32`. For
  **text-only** inference all three position channels are equal, so MRoPE reduces to ordinary
  1D partial RoPE — v1 can implement plain partial RoPE and add the 3-axis split when vision
  lands. llama.cpp uses `LLAMA_ROPE_TYPE_IMROPE` + `ggml_rope_multi(sections, n_rot=64)`.

---

## 9. End-to-end computation order (model-card schedule)

### 9.1 Prefill (prompt of `T` tokens)
```
x = embed_tokens[input_ids]                      # [T, D]
for i in 0..63:
    h = gemma_rmsnorm(x, layer[i].input_layernorm)
    if (i+1)%4==0: m = full_attention(h)          # fill KV cache
    else:          m = gdn_linear_attention(h)    # fold prompt into GDN state (chunked)
    x = x + m
    h = gemma_rmsnorm(x, layer[i].post_attention_layernorm)
    x = x + swiglu_mlp(h)
x = gemma_rmsnorm(x, norm)
logits = x[T-1] · lm_headᵀ                         # only last position
first_token = argmax(logits)
```

### 9.2 Decode (one token per step)
Identical body with `T=1`; full-attn uses single-query attention against the KV cache, GDN
uses the recurrent update (§7.1). This per-token kernel sequence is the unit later captured by
a CUDA Graph / fused into a megakernel (see design.md §11).

---

## 10. Operator inventory (implementation checklist)

Each row is a kernel to implement. "Phase" = where it runs (P=prefill, D=decode).
Shapes use the symbols from §2 (decode T=1).

### 10.1 GEMM — the W4A16 workhorse (one kernel, many shapes)
| ID | Use | A `[T,K]` | Wᵀ `[K,N]` | Out `[T,N]` | Phase |
|---|---|---|---|---|---|
| G | `q_proj` | T×5120 | 5120×12288 | T×12288 | P,D |
| G | `k_proj`,`v_proj` | T×5120 | 5120×1024 | T×1024 | P,D |
| G | `o_proj` | T×6144 | 6144×5120 | T×5120 | P,D |
| G | `gate_proj`,`up_proj` | T×5120 | 5120×17408 | T×17408 | P,D |
| G | `down_proj` | T×17408 | 17408×5120 | T×5120 | P,D |
| G | `in_proj_qkv` | T×5120 | 5120×10240 | T×10240 | P,D |
| G | `in_proj_z` | T×5120 | 5120×6144 | T×6144 | P,D |
| G | `in_proj_a`,`in_proj_b` | T×5120 | 5120×48 | T×48 | P,D |
| G | `out_proj` (GDN) | T×6144 | 6144×5120 | T×5120 | P,D |
| G | `lm_head` | T×5120 | 5120×248320 | T×248320 | P(last),D |

> `in_proj_a/b` are tiny (`N=48`) and gate-critical → candidates to keep at higher precision
> (treat as "sensitive" tensors, not 4-bit). `lm_head` is the per-token bandwidth hog (§design.md).

### 10.2 Norm / activation / elementwise
| ID | Operator | In → Out | Notes |
|---|---|---|---|
| N1 | `embedding_gather` | ids[T] → [T,D] | row gather from `embed_tokens` |
| N2 | `gemma_rmsnorm` (+fused residual add) | [T,D](+res) → [T,D] | (1+w), fp32 var |
| N3 | `gemma_rmsnorm_headwise` | [T,H,256] → same | q_norm/k_norm, (1+w) |
| N4 | `gated_rmsnorm` | o[T,48,128],z[…] → [T,48,128] | plain w, ⊙SiLU(z) |
| N5 | `silu_and_mul` (SwiGLU) | gate[T,I],up[T,I] → [T,I] | fuse with G(gate,up) |
| N6 | `sigmoid_gate_mul` | attn[T,6144],gate[…] → [T,6144] | full-attn output gate |

### 10.3 Full attention
| ID | Operator | In → Out | Phase | Notes |
|---|---|---|---|---|
| A1 | `rope_partial_mrope` | q[T,24,256],k[T,4,256] → rotated | P,D | rotate 64 dims, IMROPE |
| A2 | `flash_attn_causal_gqa` | q,k,v → o[T,24,256] | P | causal, scale 1/16, write KV |
| A3 | `attn_decode_gqa` | q[1,24,256], KV → o[1,24,256] | D | append kv, attend window |

### 10.4 GDN linear attention
| ID | Operator | In → Out | Phase | Notes |
|---|---|---|---|---|
| L1 | `causal_conv1d_prefill` | qkv[T,10240] → +SiLU | P | depthwise k=4, write conv_state |
| L2 | `causal_conv1d_decode` | qkv[1,10240], conv_state → +SiLU | D | rolling state width 3 |
| L3 | `gdn_gating` | a,b[T,48],A_log,dt_bias → g,β[T,48] | P,D | fp32; `-exp·softplus`, `σ` |
| L4 | `l2norm_headwise` | q,k[T,16,128] → normed | P,D | per head, ε=1e-6 |
| L5 | `gated_delta_recurrent` | q,k,v,g,β,state → o[1,48,128] | D | §7.1; q-scale 1/√128 |
| L6 | `gated_delta_chunked` | q,k,v,g,β,state → o[T,48,128] | P | §7.2; chunk 64, WY/UT |

### 10.5 Output / sampling / state
| ID | Operator | In → Out | Notes |
|---|---|---|---|
| O1 | `argmax_vocab` | logits[1,248320] → id | greedy (v1) |
| O2 | `sampler` (later) | logits → id | temp/top-k/top-p |
| S1 | `kv_cache_append` | k,v[4,256] @ pos | full-attn layers |
| S2 | `gdn_state_rw` | in-place conv+ssm state | GDN layers |

Sub-kernels for L6 (chunked GDN) if not writing one monolithic kernel: `cumsum` (log-decay),
`tril_matmul` (K_β·Kᵀ⊙Γ), `tri_solve` (UT inverse), plus the inter-chunk matmuls.

---

## 11. Fusion opportunities & runtime transform ownership

### 11.1 Canonical q5090 storage
- q5090 stores model-semantic tensors without mathematical folding. RMSNorm weights are not stored with
  `+1` folded in, and `A_log` is not exponentiated.
- M2.8/M3 canonical TEXT_CORE q5090 stores `conv1d.weight` in the runtime-native logical shape
  `[10240,4,1]`. The HF raw source shape `[10240,1,4]` is a legacy compatibility input shape, not the
  official M3 baseline layout.
- Implementation status, 2026-06-27: P1 implements the canonical conv1d sync across converter output,
  q5090 fixtures, runtime binding, and tests. Existing raw `[10240,1,4]` q5090 files are legacy pre-M2.8
  artifacts. They must be regenerated before they are used as official M2.8/M3 baseline inputs.
- Runtime kernels apply the semantic transforms: RMSNorm `1 + w` where required,
  `A = -exp(A_log)` for GDN gating, and the `1/sqrt(dk)=1/sqrt(128)` scale in the GDN path.
- Projection fusion is a runtime/kernel scheduling choice. The stored file keeps independently
  named tensors and source identifiers so fused kernels can consume multiple q5090 payloads
  without changing the ABI.
- GDN head/channel interpretation is a runtime contract between the q5090 `source_kind` and the
  kernels. The file does not rename or semantically remap tensors to a different framework.

### 11.2 Runtime kernel fusion (per the design.md optimization ladder)
- **fused add + RMSNorm** (residual add folded into the next norm).
- **dequant + GEMV** (W4A16) — the core decode fusion.
- **QK-norm + partial-RoPE + gate-split** in one kernel (vLLM does this:
  `fused_qk_rmsnorm_rope_gate`).
- **conv + SiLU + split**, and **post-conv → l2norm + gating** (vLLM `fused_post_conv_prep`).
- **gated_rmsnorm + out_proj**; **silu_and_mul** inside the MLP.
- whole **decode step → CUDA Graph / megakernel**.

---

## 12. State & cache shapes (per sequence)

| Cache | Per-layer shape | Layers | dtype | Grows with ctx? |
|---|---|---|---|---|
| Full-attn KV (K and V) | `[Hkv=4, dh=256]` per token | 16 | bf16 | **yes** (≈64 KB/token total) |
| GDN SSM state `S` | `[Hv=48, dv=128, dk=128]` | 48 | **fp32** | no (fixed) |
| GDN conv state | `[conv_dim=10240, W−1=3]` | 48 | bf16/fp32 | no (fixed) |

GDN state is context-independent (~3 MB/layer ssm + small conv → ~150 MB total); only the 16
full-attn layers' KV grows. See [`design.md`](design.md) §5/§8 for the memory budget.

> Note: storing the SSM state in **fp32** is our design choice (the recurrence *accumulation*
> is always fp32, and `config.mamba_ssm_dtype = float32`). vLLM's runtime cache dtype is
> configurable and can fall back to bf16 unless explicitly set; we store fp32 for fidelity.

---

## 13. Cross-implementation tensor map

| Ours / HF (`model.language_model.layers.{i}.`) | llama.cpp GGUF (`blk.{i}.`) | Runtime treatment |
|---|---|---|
| `input_layernorm.weight` | `attn_norm.weight` | apply `1+w` in RMSNorm |
| `post_attention_layernorm.weight` | `post_attention_norm.weight` | apply `1+w` in RMSNorm |
| `self_attn.q_proj.weight` | `attn_q.weight` | (Q‖gate) |
| `self_attn.{k,v,o}_proj.weight` | `attn_{k,v,output}.weight` | — |
| `self_attn.{q,k}_norm.weight` | `attn_{q,k}_norm.weight` | apply `1+w` in QK norm |
| `linear_attn.in_proj_qkv.weight` | `attn_qkv.weight` | split Q/K/V in runtime |
| `linear_attn.in_proj_z.weight` | `attn_gate.weight` | V-head interpretation in runtime |
| `linear_attn.in_proj_b.weight` | `ssm_beta.weight` | V-head interpretation in runtime |
| `linear_attn.in_proj_a.weight` | `ssm_alpha.weight` | V-head interpretation in runtime |
| `linear_attn.A_log` | `ssm_a` | compute `-exp(A_log)` in runtime |
| `linear_attn.dt_bias` | `ssm_dt.bias` | softplus bias in runtime |
| `linear_attn.conv1d.weight` | `ssm_conv1d.weight` | HF raw `[10240,1,4]`; q5090 canonical `[10240,4,1]` for direct runtime view |
| `linear_attn.norm.weight` | `ssm_norm.weight` | plain weight, no `1+w` |
| `linear_attn.out_proj.weight` | `ssm_out.weight` | V-space projection in runtime |
| `mlp.{gate,up,down}_proj.weight` | `ffn_{gate,up,down}.weight` | — |
| `embed_tokens` / `norm` / `lm_head` | `token_embd` / `output_norm` / `output` | output norm applies `1+w` |

Global GGUF hparam repurposing (FYI): `ssm.state_size`=`dk`(128), `ssm.time_step_rank`=`Hv`(48),
`ssm.group_count`=`Hk`(16), `ssm.inner_size`=`value_dim`(6144), `ssm.conv_kernel`=`W`(4).

---

## 14. Open questions / verification flags

1. **GDN gated-norm weight convention** — RESOLVED: plain `w` (no +1), confirmed by vLLM
   (`RMSNormGated`), llama.cpp converter (explicit exclusion), and the FLA reference. Keep this
   the only norm without the `+1`.
2. **V-head ordering** — choose our own GDN V-layout; just be internally consistent across all
   V-shaped tensors and the recurrence (`k/q head = v_head // 3`).
3. **`lm_head` / `embed_tokens` precision** — bandwidth vs quality; decide by measurement
   (design.md §9/§15).
4. **`in_proj_a/b` precision** — tiny + gate-critical; likely keep high precision.
5. **Conv-state storage order** (`[10240,3]` vs `[3,10240]`) — pick to match the conv kernel's
   access pattern.
6. **MRoPE for vision** — v1 implements plain partial RoPE; the 3-axis `[11,11,10]` split is
   only exercised when image/video tokens are present (out of v1 scope).

---

## 15. Sources

- Checkpoint: `/home/neroued/llama.cpp/models/Qwen3.6-27B/{config.json, *.safetensors}`.
- vLLM: `model_executor/models/qwen3_5.py`, `qwen3_next.py`, `qwen3_5_mtp.py`,
  `layers/mamba/gdn/qwen_gdn_linear_attn.py`, `layers/layernorm.py`,
  `layers/mamba/mamba_utils.py`, `layers/fla/ops/{chunk,fused_recurrent,fused_sigmoid_gating,
  fused_gdn_prefill_post_conv}.py`, `layers/rotary_embedding/mrope.py`.
- llama.cpp: `src/llama-arch.{h,cpp}`, `src/llama-model.cpp` (`LLM_ARCH_QWEN35`,
  `llm_build_qwen35`), `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
  `convert_hf_to_gguf.py` (`Qwen3_5TextModel`), `gguf-py/gguf/{constants,tensor_mapping}.py`.
- Papers / docs: Gated DeltaNet arXiv 2412.06464; DeltaNet arXiv 2406.06484; Gated Attention
  arXiv 2505.06708; Gemma arXiv 2403.08295; Qwen2-VL MRoPE arXiv 2409.12191; DeepSeek-V3 MTP
  arXiv 2412.19437; flash-linear-attention (`fla/ops/gated_delta_rule`); official Qwen3-Next
  blog; vLLM Qwen3-Next blog.
