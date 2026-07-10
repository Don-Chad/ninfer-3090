# q5090 v4.1 Header-First、按需驻留与流式 GPU 加载设计

Date: 2026-07-10

Status: design proposal（尚未实现）

Scope: q5090 C++ artifact 打开/校验、内嵌 tokenizer CPU 加载、权重 residency 规划、Host→GPU
上传、Engine 启动生命周期和加载可观测性。

---

## 0. 结论

当前加载路径必须替换为一个明确的分阶段事务：

1. `open + fstat` 后只读取 4096-byte `FileHeader`，立即校验 magic/version/minor/endian、文件
   大小关系和有界的 table 描述；错误版本最多读取 4096 bytes 就失败，不创建 CUDA context，
   不分配 CPU 大缓冲或显存。
2. 只读取 module/tensor/segment/fusion/string/tokenizer record 等 metadata，完成全部结构校验和
   Qwen3.6 模型期望校验，再生成不可变的 `Q5090LoadPlan`。
3. tokenizer 的三个 CPU-only blob 单独 `pread`，构造 `QwenTokenizer` 后释放原始 blob；它们永远
   不进入 `DeviceArena`。
4. 权重不再先读入覆盖整个文件的 pageable `std::vector`。选中的 payload 通过小型 pinned
   双缓冲从文件分块读取，再用非默认 load stream 的 `cudaMemcpyAsync` 流式上传；CPU 文件读取
   与上一块 H2D DMA 重叠，整个过程只有有界 host staging 内存。
5. residency 粒度固定为 `TEXT_BASE`、`DRAFT_HEAD`、`MTP_DRAFT`、`VISION_ENCODER` 四组。
   未选择的组不读 payload、不分配显存、不建立可执行权重 view。
6. 每个 residency group 使用独立、精确容量的 `DeviceArena`。删除基于整个 `.qus` 文件大小的
   `default_weight_bytes()` 估算，也不再为未加载模块预留显存。
7. 初版“按需”指启动/功能启用时显式生成加载计划，不在推理 kernel 中做隐式缺页。MTP 的运行时
   热启用还涉及 MTP KV、workspace、模型重新绑定和 CUDA graph 重建，不在本次 loader 改造中
   假装成一个简单的 `load_module()`。

当前 artifact 的实际数值说明了这项改造不只是启动错误处理优化：

| 区域 | 当前 payload bytes | 新默认是否驻留 |
|---|---:|---|
| TEXT_CORE（含 draft head） | 16,735,369,216 | 否，拆分 |
| TEXT_BASE（扣除 draft pair） | 16,378,329,088 | 是 |
| DRAFT_HEAD weights + id-map | 357,040,128 | 仅 `lm_head_draft` 启用时 |
| MTP_DRAFT | 451,267,584 | 仅 `mtp_draft_tokens > 0` 时 |
| VISION_ENCODER | 293,396,992 | 仅 vision 功能启用时 |

默认纯文本、无 MTP、无 draft-head 进程相对“所有功能全部驻留”可跳过约 **1.026 GiB** 可选
payload（draft + MTP + vision）。相对当前默认加载行为，MTP/VISION 本来就不上传，新增的实际显存
节省来自当前仍被 TEXT_CORE 带入的 draft head，约 **340.5 MiB**。

---

## 1. 当前实现与问题

### 1.1 实际加载顺序

当前 `WeightStore::load()`（`src/core/weight_store.cpp`）执行：

```text
read_file(path)
  -> seek 到 EOF
  -> 分配 file_size 大小的 std::vector<std::byte>
  -> 顺序读取整个 17.48 GB 文件
parse_q5090_file(full_span)
  -> 此时才校验 magic/version/header
allocate DeviceArena payload
cudaMemcpyAsync(pageable_vector -> GPU)
```

因此：

- 错误 magic、错误 `version`、错误 `format_minor`、错误模型尺寸都要在完整读取 17.48 GB 后才
  被发现。
- host RSS 至少包含一个 17.48 GB 的用户态副本；buffered file I/O 还会经过 OS page cache。
- parser API 被设计成必须拿到整个文件 span，header/metadata 校验与 payload 内容扫描耦合。
- `cudaMemcpyAsync` 的源是普通 `std::vector`，即 pageable host memory。CUDA 可能需要内部
  staging，并可能同步 stream；不能依赖它提供真正的异步流水。
- 设置 progress callback 时，当前实现每 256 MiB 都 `cudaStreamSynchronize(load_stream)`，
  明确破坏了文件读取与 H2D 的重叠；不设置 callback 时又对大 tensor 发起单个超大 copy。

### 1.2 “未上传可选模块”只完成了一半

当前 `LoadOptions.load_mtp/load_vision` 已经可以跳过未选模块的 `cudaMemcpyAsync`，这是应保留的
行为。但它发生在完整文件已经读进 host 内存之后，所以未选模块仍然承担磁盘 I/O、page cache 和
17.48 GB host buffer 的成本。

另外，draft head 位于 TEXT_CORE module 内。当前选择粒度只有 module，因此：

- `--lm-head-draft` 未启用时，`lm_head_draft` 和 id-map 仍会随 TEXT_CORE 上传；
- `Qwen3_6_27B::bind()` 通过非空 metadata record 识别 draft head，resident 和 available 的
  语义没有分离；
- `qweight()` 可能返回 payload 为 null 的非 resident view，这种接口不适合真正按需驻留。

### 1.3 显存容量按文件大小估算

`Engine::default_weight_bytes()` 当前使用：

```text
file_size + 256 MiB + fixed MTP payload budget
```

这会把 header、metadata、未来约 22.26 MiB 的 tokenizer、未选择的 MTP/VISION，以及额外的固定
MTP budget 都计入一个 `cudaMalloc`。artifact 已经提供精确的 tensor payload size 和 alignment，
没有继续猜测容量的理由。

### 1.4 当前 artifact 的大小分布

对 `out/qwen3_6_27b.q5090_w4g64_mixed_v4.qus` 的只读 header/module 检查得到：

```text
file size             17,480,324,608
metadata prefix              290,816
all weight payload    17,480,033,792
TEXT_CORE             16,735,369,216
MTP_DRAFT                451,267,584
VISION_ENCODER           293,396,992
```

metadata 只有约 284 KiB。先读取并校验 metadata、再决定是否读取 17+ GB payload，成本几乎可以
忽略。

---

## 2. 目标与非目标

### 2.1 目标

- 错误 q5090 identity/version/minor 在读取 4096-byte header 后立即失败。
- 所有结构错误、模型 metadata 不匹配和缺少必需 tensor 在任何 CUDA 大分配前失败。
- loader-owned host payload memory 保持有界，与 `.qus` 文件大小无关。
- 仅对当前功能需要的 tensor 发起文件读取和 H2D。
- tokenizer 只在 CPU 加载；权重只保留最终 GPU resident copy。
- 精确计算各 residency group 的显存容量，保留稳定 device pointer。
- 上传事务失败时不发布半初始化的 `Weight`/`Tensor` view。
- 保持 q5090 payload 原字节直接进入 GPU；没有 runtime repack、dequant 或 layout 转换。
- 提供足够的阶段耗时、读字节数、上传字节数和 residency 统计，能判断瓶颈在 header、磁盘、
  H2D、CUDA 分配还是模型绑定。

### 2.2 非目标

- 不改变 Q4/Q5/Q6/W8/BF16/I32 payload 编码。
- 不通过 Unified Memory、mapped host zero-copy 或 kernel 直接反复读取 host 权重运行推理。
- 不在 decode 过程中隐式加载权重或允许 kernel 遇到非 resident tensor 后自行阻塞。
- 不在本次设计中实现 MTP 的运行时热启用；它需要单独的 Engine 状态迁移设计。
- 不把全部可选权重长期保留在 pinned CPU memory。
- 不以 GPUDirect Storage 为首个实现的硬依赖。
- 不保留当前“完整读取到 `std::vector`”作为兼容 fallback。

---

## 3. 总体架构

```text
             CPU-only, no CUDA allocation

 path
  │
  ▼
Q5090Artifact::open
  ├─ open(O_RDONLY|O_CLOEXEC) + fstat
  ├─ pread FileHeader[4096] ────────────── fail fast
  ├─ pread bounded metadata prefix
  ├─ validate container + Qwen expectations
  ├─ expose tokenizer records
  └─ build immutable Q5090Catalog
             │
             ├────────► read tokenizer blobs ─► QwenTokenizer (CPU)
             │
             ▼
       Q5090LoadRequest
             │
             ▼
       Q5090LoadPlan
  ├─ selected residency groups
  ├─ selected tensor upload jobs
  ├─ exact per-group device capacities
  └─ total read/H2D byte counts

             CUDA allocation begins only here

             ▼
       allocate module arenas
             │
             ▼
 file fd ─► pinned slot A/B ─cudaMemcpyAsync─► selected GPU arenas
             │                    │
             └──── pread overlaps previous H2D DMA
                                  │
                                  ▼
                         synchronize once at commit
                                  │
                                  ▼
                    publish resident views + model bind
```

实现保持项目专用性，不建设通用模型加载框架。新增对象只描述 q5090/Qwen3.6 artifact：

- `Q5090Artifact`：move-only RAII 文件句柄、文件大小和已验证 catalog；
- `Q5090Catalog`：header/modules/tensors/segments/fusions/tokenizer records，不包含权重 payload；
- `Q5090LoadRequest`：四个明确的 residency 选择；
- `Q5090LoadPlan`：不可变的 tensor upload jobs 和精确 device layout；
- `Q5090StagedUploader`：固定大小 pinned ring + CUDA events。

---

## 4. Phase A：只读 Header，立即失败

### 4.1 文件打开

使用 POSIX `open`/`pread` 而不是共享 seek cursor 的 `std::ifstream`：

```text
fd = open(path, O_RDONLY | O_CLOEXEC)
fstat(fd) -> file_size
pread_exact(fd, 4096, offset=0)
```

要求 regular file，所有 offset/size 算术使用 checked `uint64_t`。`pread_exact` 必须处理 `EINTR`、
short read 和大于 `SSIZE_MAX` 的分块，不允许静默读取不足。

打开时同时保存不可变文件身份：

```text
st_dev, st_ino, st_size, st_mtim, st_ctim
```

artifact 合同要求进程使用期间文件内容不可原地修改。每次 tokenizer/residency group 读取事务开始和
结束都重新 `fstat(fd)` 并比较上述身份；任何 size/mtime/ctime 变化都拒绝本次事务。打开 fd 可以防止
路径被 rename/replace 后读到另一个 inode，但不能单独防止同一 inode 被原地覆盖，因此不能省略
事务前后的身份复查。该检查提供运行时并发修改检测，不宣称替代 cryptographic integrity；需要强
完整性时启用 selected block 增量 CRC，发布 artifact 前仍由 offline verifier 做全量校验。

### 4.2 Header 阶段必须完成的检查

- magic、major version、`format_minor`、endian tag、`header_size`；
- header reserved bytes 和 reserved flags；
- `module_count/tensor_count/segment_count/fusion_group_count` 的格式上限；
- 每个 table 的乘法、加法和 range 不溢出；
- metadata/tokenizer/weight payload 的基本有序关系；
- `payload_offset + payload_bytes == fstat.file_size`；
- metadata prefix 和 tokenizer section 不超过格式规定的上限。

建议的防御上限（q5090 v4.1 固定 Qwen3.6，而非模型 zoo）：

```text
module_count             <= 3
tensor_count             <= 2048
segment_count            <= 4096
fusion_group_count       <= 512
string_table_bytes       <= 16 MiB
non-tokenizer metadata   <= 64 MiB
tokenizer data           <= 64 MiB
```

这些上限在申请 metadata buffer 前检查。错误 major/minor 只读取 4096 bytes；错误 table size 不会
诱导 loader 分配攻击者声明的巨大 vector。

### 4.3 无 CUDA 副作用

`Q5090Artifact::open()` 是纯 CPU 操作。此阶段不得：

- 创建 `DeviceContext`；
- 调用 `cudaSetDevice/cudaMalloc/cudaMallocHost`；
- 分配 cache/work/weight arena；
- 读取任何 weight payload。

这样版本错误、截断和 metadata 攻击面可以在无 GPU 的 parser test 中完整验证。

---

## 5. Phase B：有界 Metadata 读取与结构校验

### 5.1 只读取 catalog

Header 通过后，读取：

- `ModuleRecord[]`；
- `TensorEntry[]`；
- `SegmentRecord[]`；
- `FusionGroupRecord[]`；
- string table；
- v4.1 `TokenizerRecord[3]`。

当前 metadata 只有 290,816 bytes；v4.1 tokenizer raw data 不属于 catalog metadata buffer，后续按
record 单独读取。

### 5.2 Parser 拆分

当前 `parse_q5090_file(std::span<full file>)` 拆成两层：

```cpp
ParsedQ5090Header parse_q5090_header(span<4096>, uint64_t file_size);
Q5090Catalog parse_q5090_catalog(
    const ParsedQ5090Header&,
    span<const byte> metadata,
    const Q5090Expectations&);
```

所有结构关系都只依赖 header/catalog/file size，不应依赖 payload bytes：

- table adjacency、record size 和 string bounds；
- module 顺序、tensor ranges、module span；
- tensor payload offsets、alignment、无 overlap、expected payload byte math；
- segment partition、fusion group coupling、draft-head coupling；
- source kind/qtype/layout/shape/model constants；
- 必需 TEXT tensor 是否存在。

### 5.3 结构校验与内容完整性分离

当前 C++ parser 为检查所有 inter-block padding 为零而要求 full-file span。v4.1 应明确区分：

- **runtime mandatory structural validation**：只读 metadata 即可完成，失败可能导致错误寻址；
- **selected payload read validation**：`pread` 必须返回完整字节；可选增量 CRC；
- **offline verifier integrity validation**：扫描全文件 reserved/padding、所有 block CRC、SHA 和
  数值 oracle。

runtime 默认不应为了验证未加载 VISION/MTP 的 padding 或 CRC 而读取其 payload。若提供调试用
`verify_payload_crc_on_load`，也只对 selected payload 增量计算；生产默认关闭，以维持与当前规范
“C++ normal loader 不计算 weight CRC”的约定。tokenizer 很小且决定 token identity，其 CRC 则必须
在 runtime 校验。

---

## 6. Phase C：内嵌 Tokenizer 的 CPU-only 路径

本设计依赖 q5090 v4.1 在现有 v4 文档中加入三个强制 raw UTF-8 asset：

- `tokenizer.json`；
- `merges.txt`；
- `generation_config.json`。

### 6.1 冻结的 v4.1 Header 扩展

`FileHeader` 仍为 4096 bytes，现有字段保持不动，从当前 `format_minor` 开始定义：

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 232 | 4 | u32 | `format_minor = 1` |
| 236 | 4 | u32 | `tokenizer_record_count = 3` |
| 240 | 4 | u32 | `tokenizer_record_size = 64` |
| 244 | 4 | u32 | `tokenizer_flags = 0` |
| 248 | 8 | u64 | `tokenizer_index_offset` |
| 256 | 8 | u64 | `tokenizer_index_bytes` |
| 264 | 8 | u64 | `tokenizer_data_offset` |
| 272 | 8 | u64 | `tokenizer_data_bytes` |
| 280 | 3816 | u8 | reserved zero |

严格位置关系：

```text
string_end = string_table_offset + string_table_bytes
tokenizer_index_offset = align_up(string_end, 64)
tokenizer_index_bytes = 3 * 64
tokenizer_data_offset = align_up(tokenizer_index_offset + tokenizer_index_bytes, 64)
tokenizer_data_end = tokenizer_data_offset + tokenizer_data_bytes
payload_offset = align_up(tokenizer_data_end, 4096)
payload_bytes = file_size - payload_offset
```

`string_end..tokenizer_index_offset`、index/data 之间、asset 之间和
`tokenizer_data_end..payload_offset` 的 padding 都必须为零。它们总量很小，runtime 和 offline
verifier 都必须检查。

### 6.2 `TokenizerRecord`（64 bytes）

| Offset | Size | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `kind` |
| 4 | 4 | u32 | `encoding = 0 (RAW_UTF8)` |
| 8 | 8 | u64 | `payload_offset`（absolute） |
| 16 | 8 | u64 | `payload_bytes` |
| 24 | 4 | u32 | `crc32`（raw asset bytes） |
| 28 | 4 | u32 | reserved zero |
| 32 | 32 | u8 | `sha256`（raw asset bytes） |

`kind` 固定为 `1=TOKENIZER_JSON`、`2=MERGES_TXT`、`3=GENERATION_CONFIG_JSON`，record 必须按该
顺序各出现一次，不允许缺失、重复、未知 kind/encoding 或空 payload。第一个 asset 从
`tokenizer_data_offset` 开始；后续 asset 的 `payload_offset = align_up(previous_end, 64)`；
`tokenizer_data_bytes` 包含 asset 间 zero padding，但不包含 weight payload 前的 4096 alignment
padding。runtime 必须校验 record reserved、CRC32 和所有 tokenizer padding；SHA-256 由 offline
verifier 强制校验，runtime 可选校验。

### 6.3 CPU 构造流程

流程：

```text
catalog validated
  -> pread each tokenizer record range
  -> CRC32 mandatory, SHA-256 offline verifier mandatory
  -> construct Q5090TokenizerBundle
  -> QwenTokenizer parses in-memory views
  -> release bundle raw bytes after derived tables are built
```

Tokenizer record 绝不参与 `Q5090LoadPlan`、DeviceArena capacity 或 H2D 统计。`QwenTokenizer` 删除
filesystem directory constructor、`tokenizer_dir_` 和 generation config 的硬编码 fallback。

为避免 CLI/serve 先解析一次、Engine 又重新打开/解析一次，入口使用同一个 move-only artifact：

```cpp
auto artifact = Q5090Artifact::open(weights_path, qwen3_6_27b_expectations());
QwenTokenizer tokenizer(artifact.read_tokenizer_bundle());

Engine engine(options);
engine.load(std::move(artifact));
```

`qus_bench` 不需要 tokenizer，但仍执行同一个 `Q5090Artifact::open()`，随后直接 move 给 Engine。

---

## 7. Phase D：Residency Request 与精确 Load Plan

### 7.1 固定的 residency groups

不新增通用 policy framework，只定义本项目需要的四组：

| Group | 选择条件 | 内容 |
|---|---|---|
| `TEXT_BASE` | 永远 | TEXT_CORE，排除 draft-head pair |
| `DRAFT_HEAD` | `use_lm_head_draft && mtp_draft_tokens > 0` | `LM_HEAD_DRAFT` + id-map |
| `MTP_DRAFT` | `mtp_draft_tokens > 0` | MTP_DRAFT module |
| `VISION_ENCODER` | vision 功能显式启用 | VISION_ENCODER module |

`use_lm_head_draft=true` 且 `mtp_draft_tokens==0` 是无意义配置，应在 CLI/Engine options 校验阶段直接
拒绝，而不是上传 340.5 MiB 后永不使用。

未选择 group 的行为必须同时满足：

- 不 `pread` 其 weight payload；
- 不为其分配 DeviceArena；
- 不调用 H2D；
- resident lookup 返回 null；
- catalog 仍能报告它在 artifact 中 available。

### 7.2 Available 与 Resident 分离

`WeightStore` 当前把 metadata record 和可执行 view 混在一起。新语义：

- `Q5090Catalog` 回答 `available(module/source/name)`；
- `WeightStore::tensor/qweight/qfused` 只返回 resident、device pointer 有效的 view；
- 未加载项不创建 payload=null 的伪 view；
- `ResidencyState` 明确为 `Unavailable / Available / Loading / Resident / Failed`，其中
  `Loading` 状态绝不对 model card 可见。

这会直接替换旧 lookup 行为，不保留“返回 metadata 但 payload=null”的兼容分支。

### 7.3 Load plan 内容

`Q5090LoadPlan` 在任何 CUDA 分配之前产生，至少包含：

```cpp
struct PlannedTensorUpload {
    std::uint32_t tensor_index;
    ResidencyGroup group;
    std::uint64_t file_offset;
    std::uint64_t bytes;
    std::uint64_t device_offset;
};

struct PlannedResidencyGroup {
    ResidencyGroup group;
    std::uint64_t arena_capacity_bytes;
    std::vector<PlannedTensorUpload> uploads;
};
```

`device_offset` 使用 relative-to-arena-base 的 checked 256-byte alignment 计算。v4.1 runtime 的硬
不变量是所有 weight allocation alignment 都不超过 256，且 `cudaMalloc` 返回的 arena base 必须在
分配后断言为至少 256-byte aligned。Uploader 直接使用计划得到的 `base + device_offset`，不在分配
阶段再次运行一套 `DeviceArena::alloc` 算术；`DeviceArena` 需要提供只供已验证 plan 使用的
`planned_view(offset, bytes)`/commit 接口，并检查 range 和 alignment。容量是最后一个 planned end，
不使用 module/file size 猜测。

计划阶段还应完成：

- selected module/draft pair 是否存在；
- MTP canonical 12-block/16-segment/2-fusion 约束；
- selected TEXT required tensor 集；
- 每组 read bytes、H2D bytes 和 tensor 数；
- GPU owned memory 的总预算（weights + cache + workspace）；
- 所有 planned offset 的 256-byte alignment 和 arena range。

`ModuleRecord.load_policy` 不能表达 TEXT_CORE 内 draft pair 的例外，也不能成为 runtime feature
request 之外的第二真源。v4.1 将该字段改为固定 reserved zero，loader 拒绝非零；删除 active
`LoadPolicy` enum 和 `CPU_PINNED_THEN_GPU` 语义。Residency 只由 module kind、draft source kind 和
`Q5090LoadRequest` 决定。

---

## 8. Phase E：显存分配与所有权

### 8.1 每组独立 arena

Engine 从一个 `weight_arena_` 改为最多四个可选 arena：

```text
text_base_weights_
draft_head_weights_
mtp_weights_
vision_weights_
```

好处：

- 未选组没有 `cudaMalloc`；
- 后续加载 optional group 不移动 TEXT_BASE，已有 kernel/device pointer 保持稳定；
- optional group 加载失败可以只销毁自己的 arena；
- memory stats 能直接报告各组容量和使用量；
- 不需要为未来模块在一个巨型 allocation 中预留空洞。

四个大 allocation 的数量很小，不需要引入通用 GPU allocator 或 CUDA VMM。初版 resident group
加载后不支持 unload；卸载会使 model card/CUDA graph 中保存的指针失效，需要单独生命周期设计。

Device pointer 稳定还不够：`Qwen3_6_27B` 会长期保存 `WeightStore` 返回的 `const Weight*` 和
`const Tensor*` host descriptor。禁止把所有 group records append 到共享 `std::vector`，否则后续
VISION/DRAFT load 的 vector reallocation 会让已绑定 TEXT/MTP descriptor 悬空。

每个 residency group 必须拥有独立、提交后不可变的 record slab：

```cpp
struct ResidentGroup {
    DeviceArena arena;
    std::vector<TensorRecord> tensors;
    std::vector<QuantRecord> quant;
    std::vector<FusedBlockRecord> fused;
};

std::array<std::unique_ptr<const ResidentGroup>, 4> resident_groups;
```

临时 group 在提交前完成全部 vector 构造和校验；提交只 move/swap `unique_ptr`，之后不再修改 slab
中的 vectors。Lookup 遍历 immutable group slabs。加载 VISION 只安装新的 slab，不会改变 TEXT 的
host descriptor 地址或 device pointer。

### 8.2 删除文件大小估算

删除：

- `Engine::default_weight_bytes(path)`；
- 固定 MTP payload budget；
- `EngineOptions.weight_bytes` 这一总量 override。

arena capacity 只来自 `Q5090LoadPlan`。若确实需要故障注入，测试应直接构造小 arena/uploader，不在
生产 CLI 保留一个可让容量与 catalog 脱节的参数。

### 8.3 分配前预检

metadata 和 load plan 通过后才创建 `DeviceContext`。随后：

1. 计算 weights/cache/work/IO 的精确 owned bytes；
2. `cudaMemGetInfo` 做带安全余量的 preflight；
3. 分配 selected weight arenas、cache arena、workspace；
4. 任一分配或上传失败，Engine load 事务整体回滚。

preflight 是友好错误，不替代 `cudaMalloc` 的真实结果。不能把驱动/context 开销当作精确可用容量。

---

## 9. Phase F：Pinned 双缓冲流式上传

### 9.1 为什么当前 pageable copy 不够

CUDA 官方说明，pageable host memory 参与异步 H2D 时可能先被 driver staging 到 pinned memory，
并可能发生 stream 同步；真正可依赖的异步传输需要 page-locked host memory。官方也强调 pinned
memory 是稀缺资源，不应过量使用。

仓库已经有基于 `cudaMallocHost` 的 `PinnedHostBuffer`，但 weight loader 未使用。新 loader 直接
复用并扩展它，而不是分配全文件 pinned buffer。

### 9.2 默认 pipeline

初始默认：两个 64 MiB pinned slot、一个 non-default `load_stream`、每 slot 一个 CUDA event。

```text
slot A: pread chunk 0 -> H2D chunk 0 --------------------┐
slot B:                  pread chunk 1 -> H2D chunk 1 ---┤
slot A: wait event A ->  pread chunk 2 -> H2D chunk 2 ---┤
...                                                       │
final cudaStreamSynchronize(load_stream) <----------------┘
```

伪代码：

```cpp
for (const PlannedTensorUpload& job : plan.uploads_in_file_order()) {
    for (Chunk chunk : split(job, slot_bytes)) {
        Slot& slot = slots[next_slot++ % slots.size()];
        if (slot.in_flight) cudaEventSynchronize(slot.consumed);

        pread_exact(fd, slot.host.data(), chunk.bytes, chunk.file_offset);
        cudaMemcpyAsync(chunk.device_ptr, slot.host.data(), chunk.bytes,
                        cudaMemcpyHostToDevice, load_stream);
        cudaEventRecord(slot.consumed, load_stream);
        slot.in_flight = true;
    }
}
cudaStreamSynchronize(load_stream);
```

当 CPU 阻塞在 slot B 的 `pread` 时，slot A 的 DMA 可以继续执行，不需要额外 reader thread。
event 只在复用同一 slot 时等待，不再按 progress chunk 强制同步整个 stream。

### 9.3 Chunk/batch 策略

- 默认 slot/chunk 64 MiB；通过加载 benchmark 比较 16/32/64/128 MiB，而不是硬编码“越大越好”。
- WSL 官方文档指出 pinned system memory 可用量受限，因此若 `2 x 64 MiB` 分配失败，可依次尝试
  `2 x 32 MiB`、`2 x 16 MiB`，最终失败要给出明确错误；不得回退到 full-file pageable vector。
- 大 tensor 分块；小而相邻的 tensor 可在一个 slot fill 中读取并从同一 slot 发起多个 H2D，以减少
  `pread` syscall。slot event 必须记录在该 slot 的最后一次 copy 之后。
- uploads 按 file offset 排序，维持顺序 I/O；device destination 可以位于不同 group arena。
- progress 分别报告 `file_bytes_read` 和 `h2d_bytes_enqueued/completed`，不得为了刷新进度同步 stream。
- 上传完成前不发布任何 view；最后一次 stream sync 成功后一次性 commit。

### 9.4 异步失败清理合同

一旦首个 H2D enqueue 成功，pinned slots 和临时 DeviceArena 就可能仍被 DMA 使用。`pread_exact`、
progress callback、后续 `cudaMemcpyAsync`、event record/wait 任一处抛错时，不能直接依赖栈展开调用
`cudaFreeHost/cudaFree`；这些资源必须活到专用 load stream 已确认不再引用它们。

`Q5090StagedUploader` 使用 failure guard：

1. 捕获并保存第一个原始错误，停止新的 file read/copy enqueue；
2. best-effort `cudaStreamSynchronize(load_stream)`，drain 所有已 enqueue DMA；
3. 只有 drain 成功后，才按 pinned slots/events、临时 arena 的顺序释放资源；
4. 返回原始错误，并附加 drain error（若有），不让清理错误覆盖根因；
5. 若 stream/context 已进入无法确认 DMA 完成的 fatal 状态，不再假设任何 slot/arena 可安全释放或
   复用；整个 Engine 和进程 readiness 进入 fatal，停止服务并通过 CUDA context/process teardown
   回收，不能做同进程恢复，也不能承诺旧 resident groups 仍可继续服务；
6. optional group 的“失败不影响旧 groups”承诺只适用于成功 drain 且 CUDA context 健康的错误。

progress callback 也属于可抛异常边界，必须走同一个 guard。Uploader destructor 只做 noexcept
best-effort cleanup；正确性不能依赖 destructor 第一次发现仍在 flight 的 DMA。

### 9.5 Buffered `pread` 是首选基线

基线使用普通 buffered `pread`：

- 可在 WSL2/ext4 和普通 Linux 文件系统工作；
- 逻辑简单、错误语义清楚；
- 用户态 RSS 只有 pinned ring 和小 metadata；
- OS page cache 可回收，不形成第二个 17 GB 用户态副本。

可对 selected 大范围使用 `posix_fadvise(..., POSIX_FADV_SEQUENTIAL)`；上传完成后是否
`POSIX_FADV_DONTNEED` 必须实测，因为 WSL2/不同内核对 hint 的收益不同，不作为 correctness
依赖。

### 9.6 不选择的替代方案

#### 整文件 `mmap` + pageable CUDA copy

它能删除 `std::vector`，但页面仍经 page cache，且源内存不是稳定的 pinned staging；不能解决
可依赖的异步 H2D。metadata random access 可以用 mmap，但 284 KiB `pread` 已足够，不值得同时维护
两种读取模型。

#### 对整文件或大 mmap range 做 `cudaHostRegister`

注册/注销是重量操作，17 GB pinned memory 会伤害系统和 WSL2；range alignment、page fault 和
注册失败处理也复杂。固定小 ring 更直接。

#### `O_DIRECT` + pinned buffers

可绕过 page cache，但要求 file offset、size、buffer alignment 和文件系统支持；小尾块还需要 bounce
处理。它只减少一次 kernel page-cache copy，不减少必须的 H2D。作为 pinned baseline 后的独立
benchmark 变体评估，不进入首个实现。

#### Unified Memory / mapped zero-copy 权重

离散 GPU 上每 token 会重复通过 PCIe 访问约 16 GiB 权重，完全违背 decode 的 HBM bandwidth 目标；
不可采用。

---

## 10. GPUDirect Storage 的定位

cuFile/GDS 理论上可以把 selected file ranges 直接读入 device arena，去掉显式 CPU staging：

```text
NVMe/filesystem -> GPU memory
```

但它不是当前实现的默认路径：

- NVIDIA 文档要求 direct path 满足文件系统、`O_DIRECT`/alignment、driver/topology 等条件；
- 条件不满足时 cuFile 可以进入 compatibility mode，而该模式仍通过 CPU system memory staging；
- 当前开发机是 WSL2 (`6.6.87.1-microsoft-standard-WSL2`)；虽然安装了 `libcufile.so`，但没有
  `nvidia_fs` module，文件位于 WSL ext4 虚拟磁盘。仅凭 library 存在不能证明是 storage→GPU
  direct path；
- WSL2 还对 pinned system memory 有已知限制，因此基线 ring 必须保持小而有界。

后续只有同时满足以下 gate 才增加 cuFile backend：

1. `gdscheck`/cuFile properties 明确报告目标文件和 GPU 使用 direct-capable path；
2. 非 compatibility-mode 的证据可记录到 load report；
3. 真实 17 GB artifact 的 cold/warm load benchmark 显著优于 pinned pipeline；
4. 对未对齐 tensor 头尾有明确 bounce 策略；
5. WSL2 和普通 Linux 的 pinned baseline 始终可用，且不会被隐式 cuFile fallback 替代。

在这些条件满足前，不为 cuFile 建设通用 `PayloadTransport` 框架。先把 header-first、selective I/O
和 pinned staging 做对，之后再以一个受控 q5090 uploader 变体评估。

---

## 11. Phase G：提交、绑定和可选模块生命周期

### 11.1 两阶段提交

每个 residency group 的加载是事务：

1. 状态 `Available -> Loading`；
2. 创建临时 arena；
3. 上传全部 planned tensors；
4. load stream 同步成功；
5. 在临时 immutable `ResidentGroup` slab 中构造全部 records；
6. 校验 group-specific expectations；
7. 将 group-owned `unique_ptr` 安装到空 slot，状态原子式变为 `Resident`，不 append/mutate 其他
   group 的 record storage。

任一步骤失败：先遵循 §9.4 drain 所有 in-flight DMA，再销毁临时 arena/records 并将状态设为
`Failed`。只有成功 drain 且 CUDA context 健康时旧 resident groups 才保证不受影响；CUDA fatal
错误使整个 Engine/进程进入 fatal 并要求重启，不能在无法证明 DMA quiescent 时普通析构后继续运行。
初始 TEXT_BASE 失败时 Engine 始终保持 unloaded。

### 11.2 模型绑定顺序

启动时必须在构造 `Qwen3_6_27B` 之前完成所有请求的 residency：

```text
TEXT_BASE resident
[DRAFT_HEAD resident]
[MTP_DRAFT resident]
allocate KV/state/work
construct Qwen3_6_27B -> bind stable pointers
capture/warm CUDA graphs later
```

VISION 不属于当前 text model card；未来 vision card 在首次使用前显式调用 vision load transaction，
随后绑定自己的稳定 pointers。

### 11.3 “按需”不等于推理中隐式热加载

本次实现支持按 `EngineOptions`/服务能力在 load plan 阶段选择 group。MTP 如果启动时未启用，不上传
也不分配 MTP KV/workspace。

未来若确实需要运行时 `Engine::enable_mtp(k)`，它必须是单独的显式事务：停止请求、清除 graph、
加载 MTP arena、分配 MTP KV/work、重新构造或 rebind model card、重新 warmup，再恢复 ready。不能把
它实现为 `qweight()` 发现 null 后偷偷读盘。

---

## 12. Engine 与入口集成

### 12.1 统一 artifact 生命周期

主 CLI、server 和 bench 都先创建 artifact，再 move 给 Engine：

```text
qus / qus-serve:
  artifact open+validate
  tokenizer read+parse
  engine load(move artifact)

qus_bench / token-id tools:
  artifact open+validate
  engine load(move artifact)
```

为此，将当前 private `Engine::expectations()` 移到 L2 的 Qwen3.6 model contract，例如
`qwen3_6_27b_q5090_expectations()`；L0 artifact 不反向依赖 Engine。

Engine 保留 move-only artifact/fd，以便未来显式加载 VISION；当前只用 TEXT/MTP 时保持一个打开的
只读 fd 成本可忽略，并保证即使路径随后被替换，进程仍读取启动时验证过的同一 inode。fd 本身不
阻止原 inode 被原地修改；每次延迟 group load 仍必须执行 §4.1 的事务前后 `fstat` 身份复查，变化
时拒绝加载。强对抗完整性需要 selected payload CRC，不能只依赖 inode/timestamp。

### 12.2 Engine load 顺序

新的 `Engine::load(Q5090Artifact&&)`：

```text
reset previous runtime state
assert artifact catalog was validated against Qwen expectations (CPU, no file reread)
build LoadRequest + LoadPlan (CPU)
validate option dependencies (CPU)
create DeviceContext
preflight exact GPU owned bytes
allocate selected weight arenas
stream selected weight payloads
allocate cache/work/state/io
publish WeightStore
construct/bind model card
mark Engine loaded
```

可以在 weight upload 前分配 cache/work，也可以之后分配；推荐先做总量 preflight、再上传 weights、
最后分配 cache/work。任一后续分配失败由 Engine load RAII 回滚已上传 weights。

### 12.3 Readiness

服务端只有在以下全部成功后才进入 ready：

- artifact/header/catalog/tokenizer 校验；
- selected weights resident；
- model bind；
- sampler/KV/state 初始化；
- 可选 warmup/graph capture。

错误版本应在日志中显示 `read_header` 阶段，而不是笼统的 `failed to read q5090 weight file`。

---

## 13. API 和文件归属建议

### L0 artifact/catalog

- `include/qus/core/q5090_artifact.h`（新增）
- `src/core/q5090_artifact.cpp`（新增）
- `include/qus/core/weight_store_parser.h`
- `src/core/weight_store_parser.cpp`

职责：fd、header-first、catalog parsing、range validation、tokenizer records、pure-CPU load plan。

### L0 residency/upload

- `include/qus/core/weight_store.h`
- `src/core/weight_store.cpp`
- `include/qus/core/arena.h`
- `src/core/arena.cu`
- `include/qus/core/device.h`
- `src/core/device.cu`

职责：pinned slots/events、selected payload upload、per-group arenas、transactional publish、stats。

### L2/runtime

- `include/qus/model/config.h` 或专用 model-contract header
- `include/qus/runtime/engine.h`
- `src/runtime/engine.cpp`
- `src/model/qwen3_6_27b.cpp`

职责：Qwen expectations、residency request、精确 GPU 总预算、绑定顺序、MTP option coupling。

### Text/entrypoints

- `include/qus/text/tokenizer.h`
- `src/text/tokenizer.cpp`
- `src/main.cpp`
- `src/serve/generation_service.cpp`
- CLI/serve options

职责：从已验证 artifact 读取 CPU tokenizer；删除外部 `--tokenizer`。

### Format/tooling/docs

- `docs/q5090_packed_file_format_v4.md`
- `tools/q5090_convert/format.py`
- `tools/q5090_convert/convert.py`
- `tools/q5090_convert/verify.py`
- `tests/fixtures/make_q5090_fixture.py`

职责：v4.1 tokenizer section、`ModuleRecord` reserved-zero 字段、manifest 和 fixture 同步。

---

## 14. 可观测性

`Q5090Progress` 从一个模糊 callback 扩展为稳定 phase 名和 load stats；不要求复杂事件框架。

推荐 phases：

```text
open file
read header
read metadata
validate catalog
read tokenizer
plan residency
allocate gpu
read payloads
upload payloads
bind model
```

最终 report 至少包含：

- artifact file size/version/minor；
- header/metadata/tokenizer 实际读取 bytes；
- 每 residency group：available/resident、tensor count、file bytes、H2D bytes、arena capacity；
- skipped payload bytes（MTP/VISION/draft）；
- pinned slot count/bytes；
- I/O backend（`buffered_pread_pinned`）；
- header、metadata、tokenizer、cuda allocation、file read、H2D、bind 的 wall time；
- H2D throughput 和 end-to-end selected payload throughput；
- peak loader-owned host bytes。

progress 更新不得改变同步行为。是否注册 callback 不能再决定 upload 是单次 copy 还是每块
`cudaStreamSynchronize`。

---

## 15. 错误语义

错误必须指出阶段和范围，例如：

```text
q5090 read_header: unsupported format_minor 0 (expected 1)
q5090 read_metadata: tensor index range [x,y) exceeds file size z
q5090 tokenizer: TOKENIZER_JSON crc32 mismatch
q5090 plan: MTP requested but MTP_DRAFT module is absent
q5090 plan: lm-head-draft requires mtp_draft_tokens > 0
q5090 upload: short read at file offset x, wanted n, got m
q5090 upload: cudaMemcpyAsync failed for TEXT_BASE tensor ...
```

禁止：

- 对错误版本继续尝试 legacy parser；
- 缺 tokenizer 时回退外部目录；
- MTP/vision 缺失时静默关闭已请求功能；
- pinned allocation 失败时回退 full-file vector；
- optional group 半加载后仍让 lookup 返回 pointer。

---

## 16. v4.1 格式文档需要同步冻结的合同

现有 `docs/q5090_packed_file_format_v4.md` 直接更新，不新建 v5：

- `format_minor = 1`；runtime/converter 只接受 1；
- header 增加 tokenizer record/data offsets 和 sizes；
- tokenizer records/data 位于 string table 后、weight payload 前；
- `payload_bytes` 仍只覆盖 weight block payload；
- tokenizer 三个 asset 强制存在、raw UTF-8、CPU-only；
- TEXT/MTP/VISION tensor/module counts 不因 tokenizer 改变；
- `ModuleRecord` offset 40 的旧 `load_policy` 改为 reserved zero，非零拒绝；runtime residency 仅由
  module/source identity 和显式 feature request 决定；
- runtime structural checks 不再要求读取未选 weight payload 的 zero padding/CRC；全量完整性归
  offline verifier；
- manifest 增加 tokenizer hashes/sizes 和 canonical residency policy；
- 明确 draft head 是否 resident 由 runtime feature request 决定，不等同于 TEXT_CORE resident。

旧 v4.0 artifact 不兼容、不提供注入 tokenizer 的迁移工具；重新转换生成 v4.1。

---

## 17. 验证策略

### 17.1 Header/catalog 二进制合同测试

允许且必要，因为它保护真实文件格式风险：

- 仅 4096-byte 文件中的错误 magic/version/minor 立即报告对应错误，而不是 payload truncation；
- 巨大/溢出的 count、offset、size 在 metadata allocation 前拒绝；
- metadata 截断、table overlap、string 越界、payload range overlap；
- tokenizer header offsets/record size/strict order、record 缺失/重复/越界/CRC 错误，以及各段非零
  padding；
- v4.0 明确拒绝；
- malformed file 不触发 CUDA 初始化的 CPU-only parser test。
- catalog 后 truncate、rename replacement、同 inode 等长 overwrite/timestamp 变化均按 §4.1 语义
  检测；未启用 selected CRC 时明确不声称抵抗可控制 timestamps 的对手。

### 17.2 Selective residency 行为测试

使用 compact canonical fixture 验证最终可观察行为：

| Request | 应 resident | 应不 resident |
|---|---|---|
| default | TEXT_BASE | DRAFT, MTP, VISION |
| MTP | TEXT_BASE, MTP | DRAFT, VISION |
| MTP + draft | TEXT_BASE, MTP, DRAFT | VISION |
| vision | TEXT_BASE, VISION | MTP, DRAFT |

检查：

- resident device bytes 与 fixture payload bit-identical；
- non-resident lookup 返回 null，catalog available 仍为 true；
- stats 中实际 file-read/H2D bytes 不包含 skipped groups；
- group load 中途 short read/CUDA error 不发布任何该组 view；
- 对成功 drain 且 CUDA context 健康的 recoverable error，已 resident TEXT 不受后续 optional
  group 失败影响；CUDA fatal 则必须使整个 Engine/进程失效。
- TEXT model bind（以及可用时 graph capture）后再加载 VISION，随后再次执行 text inference，验证
  TEXT 的 host descriptor address 和 device pointer 均未变化。
- 每个 planned offset 为 256-byte aligned，实际 arena base 满足 256 alignment，`planned_view`
  对 offset/range 漂移必须拒绝。

### 17.3 真实 artifact 验证

重新生成 v4.1 后：

- converter verifier 全量结构/zero padding/CRC/SHA 通过；
- 所有 weight TensorEntry 的 payload bytes/CRC 与 v4.0 对应 block 一致；
- default load resident payload 为 TEXT_BASE，而不是整个 TEXT_CORE；
- MTP、draft、vision 各自增加的 resident/H2D bytes 等于 catalog 精确值；
- tokenizer HF golden encode/decode 通过；
- greedy/MTP 短 E2E 输出与转换前一致。

### 17.4 Host memory 与加载性能

添加 loader benchmark/report，不把性能阈值塞进普通单元测试：

- `/usr/bin/time -v` 记录 max RSS；loader-owned payload buffer 应固定在 metadata + tokenizer +
  pinned ring 的量级，而非 17.48 GB；
- 分别测 cold-ish 和 warm page-cache 场景；记录环境和是否能 drop cache；
- sweep pinned slot 16/32/64/128 MiB、2/3 slots；
- 记录 file read GB/s、H2D GB/s、end-to-end GB/s；
- Nsight Systems 确认 H2D 之间不再出现每 256 MiB 的 host-side stream synchronize；
- default、MTP、MTP+draft、vision 四种 selection 都报告读取/上传 bytes。

验收标准：

- 错误 version 的失败延迟不随文件大小增长；
- default loader 用户态不再持有 full-file payload copy；
- selected payload 每个字节只经过一个 bounded pinned slot 后进入最终 GPU 地址；
- warm/cold load 时间不得显著劣于当前路径，且 host RSS、skipped I/O 和显存容量必须按设计下降；
- progress callback 开/关不改变上传同步拓扑。

### 17.5 GPU 生命周期验证

- compact fixture 在 `compute-sanitizer` 下执行 staged upload + model bind smoke；
- optional arena 失败回滚后不得留下被 model card 引用的 pointer；
- 在已有 DMA in-flight 时注入下一块 short read、progress callback throw、enqueue/event failure，验证
  failure guard 先 drain 再释放 slots/arena；若注入 CUDA fatal，则验证整个 Engine 失效而非继续使用
  旧 groups；
- CUDA graph 只在所有 startup residency commit 后 capture；
- 本次不实现 unload，因此不需要测试 graph 中 pointer 被释放的行为。

---

## 18. 推荐实施顺序

### Phase 1：Header-first artifact/catalog

- 建立 `Q5090Artifact`/fd reader；
- 拆分 header/catalog parser；
- 去除 full-span parser 对 payload padding 的依赖；
- binary-contract tests；
- 此阶段尚可保留旧 uploader，但必须先完成 metadata 校验。

Definition of done：错误版本读取 4096 bytes 后失败，合法 artifact 在无 CUDA 的情况下完成完整 catalog
校验和 load plan。

### Phase 2：v4.1 tokenizer integration

- 更新 v4 文档和 converter/verify/fixture；
- artifact 读取 tokenizer；
- `QwenTokenizer` 改为内存 bundle；
- 删除运行时和 converter 的外部 tokenizer override。

Definition of done：CLI/server 只给一个 `.qus`，HF tokenizer golden 通过。

### Phase 3：Residency plan 与精确 arenas

- 四组 residency；
- available/resident API 分离；
- per-group DeviceArena + immutable record slabs；
- plan-provided device offsets 和 allocation 后 alignment assertions；
- 删除 `default_weight_bytes`、固定 MTP budget 和总量 override；
- Engine 按选项构造 plan，再分配。

Definition of done：四种 selection 的 resident bytes、arena capacity 和 lookup 行为与 catalog 精确一致。

### Phase 4：Pinned staged uploader

- 双缓冲 slots/events；
- selected `pread` + H2D pipeline；
- failure guard、文件身份复查和 transactional publish；
- progress/stats 不引入同步；
- real-file RSS/load benchmark 和 nsys 检查。

Definition of done：无 full-file host vector；default 不读取/上传 MTP/VISION/draft payload；真实模型输出
不变。

### Phase 5：可选 GDS 调研

仅在 pinned baseline 完成并有完整报告后进行。先验证当前部署是否有 direct path，再决定是否值得添加
受控 cuFile uploader；没有 direct-path 证据则关闭该任务，不以 compatibility mode 冒充 GDS 收益。

---

## 19. 风险与审查重点

1. **Parser 安全边界。** Header 是不可信输入；任何 allocation 前必须完成 caps 和 overflow check。
2. **Device offset 一致性。** Plan offset 是唯一真源；arena base/offset/range alignment 必须断言，
   不能在分配时复制第二套 placement 算术。
3. **Host descriptor 稳定性。** 已绑定 group 的 record slab 不可 append/move；新增 optional group 只能
   安装独立 immutable slab。
4. **Pinned slot 生命周期。** 只有对应 CUDA event 完成后才能让 `pread` 覆盖 slot；异常路径也必须
   drain 后释放。
5. **事务发布。** model card 绝不能看到 Loading/Failed group 的 pointer；CUDA fatal 使整个 Engine
   失效。
6. **Artifact 不可变性。** fd 防路径替换但不防 inode 原地修改；每个延迟事务必须复查 file identity。
7. **Draft-head residency。** available 不代表 resident；未启用时必须真正节省 357,040,128 bytes。
8. **MTP coupling。** 加载 MTP 权重、MTP KV/workspace 和模型绑定必须来自同一个 option 决策。
9. **Progress Heisenbug。** 注册 callback 不得改变同步、chunk 或 copy 行为，callback throw 必须进入
   uploader failure guard。
10. **WSL2 差异。** pinned capacity、page cache、fadvise、GDS 结论必须在当前 WSL2 和至少一个原生
   Linux 环境分别记录。

该改动涉及二进制格式、文件范围安全、GPU pointer lifetime、显存预算和多阶段事务。实现完成后需要
独立审查 parser/plan 和 uploader/lifetime 两个风险面，不能只依赖 E2E 能跑通。

---

## 20. 外部依据

- NVIDIA CUDA C++ Best Practices Guide：pinned memory、异步 H2D 和 non-default stream 的要求，
  以及不要过量使用 pinned memory：
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations>
- NVIDIA CUDA Runtime API synchronization behavior：pageable host memory 可能先 staging，并可能
  发生 stream synchronization：
  <https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html>
- NVIDIA GPUDirect Storage Overview：direct path 的 `O_DIRECT`/filesystem/alignment 条件和
  compatibility-mode CPU staging：
  <https://docs.nvidia.com/gpudirect-storage/overview-guide/>
- NVIDIA CUDA on WSL User Guide：WSL2 的 pinned system memory availability 限制：
  <https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications>
