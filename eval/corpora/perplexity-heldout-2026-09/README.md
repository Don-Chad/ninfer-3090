# NInfer Held-Out Perplexity Corpus

`ninfer-ppl-heldout-2026-09-v1` is a perplexity corpus whose text was written after Qwen3.8's
release (2026-08-14), so the model cannot have trained on it. It complements `ninfer-ppl-1m-v1`,
whose public sources (WikiText-2, PG-19, a 2023 Wikipedia dump) are almost certainly in the
training data and can understate the cost of a quantization change. Use this corpus to judge
weight or KV representation quality; keep `ninfer-ppl-1m-v1` for comparisons with existing
published numbers.

It contains 12 independent UTF-8 streams of about 65,536 tokens each, two per domain:

| Domain | Source | `00` (quick) | `01` |
|---|---|---|---|
| `wikipedia_en` | English Wikipedia articles created 2026-09-02..30 | articles | articles |
| `wikipedia_zh` | Chinese Wikipedia articles created 2026-09-02..30 | articles | articles |
| `arxiv` | arXiv titles and abstracts submitted in September 2026 | mathematics | computing, statistics, quantum |
| `github_code` | Python, TypeScript, Rust and Go from MIT/Apache repositories created in September 2026 | mixed | mixed |
| `own_code` | This fork's files first added after 2026-08-20 (C++/CUDA, scripts, Python) | mixed | mixed |
| `synthetic_chat` | Conversations rendered with the artifact's chat template, thinking disabled | mixed | mixed |

`quick` selects stream `00` of every domain, about 393K tokens; `full` selects all 12. Each
evaluation tokenizes the committed text with its artifact's tokenizer and reports exact counts.

## Selection

`tools/eval/build_heldout_corpus.py` built the corpus; `provenance/*.jsonl` maps every stream to
its source documents and byte ranges, and `provenance/generation.json` records the windows.

- **Wikipedia:** pages created in the main namespace within the window, at least 6,000 bytes at
  creation, not redirects, not Content Translation imports, not lists, and still in the main
  namespace when fetched (deleted and draftified pages are excluded). Reference, link and
  note sections are cut, and each article is capped at 6,144 tokens.
- **arXiv:** first versions whose primary category is one of eight per stream; abstracts shorter
  than 400 characters are skipped.
- **GitHub:** non-fork, unarchived repositories with at least 100 stars and no commits dated
  before September 2026. Tests, vendored, generated and example files are excluded, at most
  eight files per repository, each capped at 3,072 tokens.
- **Own code:** files whose adding commit is by this fork's maintainers, at the recorded commit.
- **Synthetic chat:** documents not used by the other streams, grouped three to a conversation.
  Each user turn is a templated request ("Tell me about ...", "Write `path` for this project.");
  each assistant turn is the document.

Documents are ordered by a hash of the corpus identity and the document key, so a rebuild from
the same fetch cache reproduces identical bytes.

## Limitations

- New does not mean unseen: a new Wikipedia article can paraphrase older sources, and a new
  repository can copy older code.
- Much September 2026 GitHub code, and part of this fork's code, was written with AI assistance.
  Machine-written text is easier to predict than human text, which lowers those domains'
  perplexity.
- `own_code` is mostly C++/CUDA; `github_code` carries the other languages.
- `synthetic_chat` scores every token, including the user turns and template tokens. The template
  tokens are almost free to predict, and the requests are formulaic. It tests chat-formatted input,
  not real conversation style.
- The corpus is a snapshot. A model whose training data postdates September 2026 may have seen
  it; build a new corpus under a new identity rather than changing this one.

See `THIRD_PARTY_NOTICES.md` before redistributing the corpus.
