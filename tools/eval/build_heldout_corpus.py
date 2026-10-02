"""Build ninfer-ppl-heldout-2026-09-v1, a perplexity corpus written after the model's training.

Every source postdates Qwen3.8 (released 2026-08-14): Wikipedia articles created in September
2026, arXiv abstracts submitted in September 2026, permissive GitHub repositories created in
September 2026, this fork's own files first added after 2026-08-20, and synthetic chat
conversations that wrap further held-out documents in the artifact's chat template.

Network sources change over time, so the committed text is the corpus contract and this script
is its provenance: every stream records the page revision, arXiv version, repository commit or
local commit its text came from. Fetched documents are cached under --cache so a rerun with the
same cache rebuilds identical streams.

    python tools/eval/build_heldout_corpus.py --artifact models/qwen3_8_27b.v3.ninfer \
        --cache <scratch-dir> --out eval/corpora/perplexity-heldout-2026-09
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.artifact.reader import Artifact  # noqa: E402

CORPUS_ID = "ninfer-ppl-heldout-2026-09-v1"
SEED = CORPUS_ID
STREAM_TOKENS = 65536
STREAMS_PER_DOMAIN = 2
DOC_TOKEN_CAP = 6144
USER_AGENT = "ninfer-corpus-builder/1.0 (https://github.com/ashalliants/ninfer-3090)"
WINDOW_START = "2026-09-02T00:00:00Z"  # Wikipedia recent changes keep 30 days.
WINDOW_END = "2026-09-30T23:59:59Z"
OWN_CODE_SINCE = "2026-08-20"
OWN_CODE_AUTHORS = {"Warlax", "ashalliants"}

WIKI_TAIL_SECTIONS = {
    "en": ("References", "External links", "See also", "Notes", "Further reading",
           "Bibliography", "Sources", "Citations", "Footnotes"),
    "zh": ("参考文献", "參考文獻", "参考资料", "參考資料", "外部链接", "外部連結", "参见",
           "參見", "注释", "註釋", "延伸阅读", "延伸閱讀", "相关条目", "相關條目", "来源", "來源"),
}
ARXIV_CATEGORIES = {
    "math": ("math.NT", "math.CO", "math.PR", "math.AP", "math.AG", "math.OC", "math.DG",
             "math.ST"),
    "cs": ("cs.LG", "cs.CL", "cs.DS", "cs.CR", "cs.PL", "cs.DC", "stat.ML", "quant-ph"),
}
GITHUB_LANGUAGES = {
    "python": (".py",),
    "typescript": (".ts", ".tsx"),
    "rust": (".rs",),
    "go": (".go",),
}
GITHUB_LICENSES = ("mit", "apache-2.0")
OWN_CODE_SUFFIXES = (".py", ".cpp", ".h", ".cu", ".cuh", ".ps1", ".sh", ".bat", ".mjs")
OWN_CODE_EXCLUDE = ("eval/", "tests/fixtures/", "third_party/", ".claude/", "tools/eval/")


@dataclass
class Document:
    kind: str
    key: str
    text: str
    provenance: dict
    title: str = ""
    tokens: int = 0
    extra: dict = field(default_factory=dict)


def order_key(*parts: object) -> str:
    return hashlib.sha256("\x1f".join([SEED, *map(str, parts)]).encode()).hexdigest()


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, fetch):
        path = self.root / (hashlib.sha256(name.encode()).hexdigest() + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = fetch()
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value


def http_get(url: str, *, delay: float = 0.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            time.sleep(delay)
            return body
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def gh_api(path: str, *, raw: bool = False) -> bytes:
    args = ["gh", "api", path]
    if raw:
        args += ["-H", "Accept: application/vnd.github.raw"]
    return subprocess.run(args, check=True, capture_output=True).stdout


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def cap_tokens(text: str, tokenizer: Tokenizer, cap: int) -> str:
    """Cut at a line boundary so the text encodes to at most ``cap`` tokens."""
    if len(tokenizer.encode(text).ids) <= cap:
        return text
    lines = text.split("\n")
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        if len(tokenizer.encode("\n".join(lines[:mid])).ids) <= cap:
            low = mid
        else:
            high = mid - 1
    return "\n".join(lines[:low]).rstrip() + "\n"


# --- Wikipedia --------------------------------------------------------------------------------


def wikipedia_documents(lang: str, cache: Cache, tokenizer: Tokenizer) -> list[Document]:
    api = f"https://{lang}.wikipedia.org/w/api.php"

    def created():
        pages, cont = [], {}
        while True:
            params = {
                "action": "query", "list": "recentchanges", "rctype": "new", "rcnamespace": 0,
                "rcstart": WINDOW_END, "rcend": WINDOW_START,
                "rcprop": "title|ids|timestamp|tags|sizes|redirect", "rclimit": 500,
                "format": "json", **cont,
            }
            data = json.loads(http_get(api + "?" + urllib.parse.urlencode(params), delay=0.5))
            pages += data["query"]["recentchanges"]
            if "continue" not in data:
                return pages
            cont = data["continue"]

    candidates = [
        page for page in cache.get(f"wiki-new-{lang}", created)
        if "redirect" not in page
        and page["newlen"] >= 6000
        and not any(tag.startswith("contenttranslation") for tag in page.get("tags", []))
        and not page["title"].startswith(("List of", "Lists of"))
        and "列表" not in page["title"]
    ]
    candidates.sort(key=lambda page: order_key("wiki", lang, page["pageid"]))
    candidates = candidates[:400]

    def extracts(ids: list[int]):
        params = {
            "action": "query", "prop": "extracts|revisions|info", "explaintext": 1,
            "exsectionformat": "wiki", "rvprop": "ids|timestamp", "pageids": "|".join(map(str, ids)),
            "format": "json", "formatversion": 2,
        }
        return json.loads(http_get(api + "?" + urllib.parse.urlencode(params), delay=0.5))

    documents = []
    # TextExtracts returns one full-page extract per request.
    for start in range(0, len(candidates)):
        batch = candidates[start : start + 1]
        data = cache.get(f"wiki-extract-{lang}-{batch[0]['pageid']}",
                         lambda: extracts([p["pageid"] for p in batch]))
        created_by_id = {p["pageid"]: p for p in batch}
        for page in data["query"].get("pages", []):
            # Skip deleted pages and articles since moved out of the main namespace (draftified).
            if page.get("missing") or "extract" not in page or page.get("ns") != 0:
                continue
            text = page["extract"]
            tail = WIKI_TAIL_SECTIONS[lang]
            match = re.search(r"\n=+ *(" + "|".join(map(re.escape, tail)) + r") *=+", text)
            if match:
                text = text[: match.start()]
            text = normalize(f"{page['title']}\n\n{text}")
            if len(text) < (1500 if lang == "zh" else 3000):
                continue
            text = cap_tokens(text, tokenizer, DOC_TOKEN_CAP)
            first = created_by_id[page["pageid"]]
            documents.append(Document(
                kind=f"wiki_{lang}", key=str(page["pageid"]), text=text, title=page["title"],
                provenance={
                    "source": f"{lang}.wikipedia.org", "title": page["title"],
                    "pageid": page["pageid"], "created": first["timestamp"],
                    "creation_revid": first["revid"],
                    "revid": page["revisions"][0]["revid"],
                    "url": f"https://{lang}.wikipedia.org/?curid={page['pageid']}",
                    "license": "CC BY-SA 4.0",
                },
            ))
    return documents


# --- arXiv ------------------------------------------------------------------------------------


def arxiv_documents(group: str, cache: Cache) -> list[Document]:
    atom = "{http://www.w3.org/2005/Atom}"
    arxiv = "{http://arxiv.org/schemas/atom}"
    documents = []
    for category in ARXIV_CATEGORIES[group]:
        def fetch(category=category):
            query = (f"cat:{category} AND submittedDate:[{WINDOW_START[:10].replace('-', '')}0000"
                     f" TO {WINDOW_END[:10].replace('-', '')}2359]")
            params = {"search_query": query, "start": 0, "max_results": 120,
                      "sortBy": "submittedDate", "sortOrder": "ascending"}
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
            return http_get(url, delay=3.5).decode("utf-8")

        feed = ET.fromstring(cache.get(f"arxiv-{category}", fetch))
        for entry in feed.findall(atom + "entry"):
            primary = entry.find(arxiv + "primary_category")
            identifier = entry.findtext(atom + "id").rsplit("/", 1)[-1]
            if primary is None or primary.get("term") != category or not identifier.endswith("v1"):
                continue
            title = " ".join(entry.findtext(atom + "title").split())
            abstract = " ".join(entry.findtext(atom + "summary").split())
            if len(abstract) < 400:
                continue
            documents.append(Document(
                kind=f"arxiv_{group}", key=identifier, title=title,
                text=normalize(f"{title}\n\n{abstract}"),
                provenance={
                    "source": "arxiv.org", "id": identifier, "category": category,
                    "published": entry.findtext(atom + "published"),
                    "url": f"https://arxiv.org/abs/{identifier}",
                    "license": "arXiv metadata, CC0 1.0",
                },
            ))
    documents.sort(key=lambda doc: order_key("arxiv", doc.key))
    return documents


# --- GitHub -----------------------------------------------------------------------------------


def github_documents(language: str, cache: Cache, tokenizer: Tokenizer) -> list[Document]:
    suffixes = GITHUB_LANGUAGES[language]

    def search():
        repos = []
        for license_key in GITHUB_LICENSES:
            out = subprocess.run(
                ["gh", "search", "repos", "--created", f">={WINDOW_START[:10]}",
                 "--license", license_key, "--language", language, "--stars", ">=100",
                 "--sort", "stars", "--limit", "40", "--json",
                 "fullName,createdAt,isFork,isArchived,defaultBranch,license,stargazersCount"],
                check=True, capture_output=True, text=True, encoding="utf-8").stdout
            repos += json.loads(out)
        return repos

    repos = [r for r in cache.get(f"gh-search-{language}", search)
             if not r.get("isFork") and not r.get("isArchived")
             and r["createdAt"] <= WINDOW_END]
    repos = sorted({r["fullName"]: r for r in repos}.values(),
                   key=lambda r: order_key("github", r["fullName"]))

    documents, used_repos = [], 0
    for repo in repos:
        if used_repos == 6:
            break
        name = repo["fullName"]

        def tree(name=name, branch=repo.get("defaultBranch") or "main"):
            commit = json.loads(gh_api(f"repos/{name}/commits/{branch}"))
            listing = json.loads(gh_api(f"repos/{name}/git/trees/{commit['sha']}?recursive=1"))
            first_commit = json.loads(gh_api(f"repos/{name}/commits?sha={commit['sha']}&per_page=1"
                                             "&until=2026-08-31T23:59:59Z"))
            return {"sha": commit["sha"], "tree": listing.get("tree", []),
                    "history_before_window": bool(first_commit)}

        info = cache.get(f"gh-tree-{name}", tree)
        if info["history_before_window"]:
            continue  # Repository imports commits from before the window.
        files = [
            item for item in info["tree"]
            if item["type"] == "blob" and item["path"].endswith(suffixes)
            and 800 <= item.get("size", 0) <= 40000
            and not re.search(r"(^|/)(tests?|vendor|third_party|node_modules|dist|build|"
                              r"examples?|generated|migrations)/|\.min\.|_pb2|\.d\.ts$|"
                              r"(^|/)test_|_test\.", item["path"])
        ]
        if len(files) < 4:
            continue
        files.sort(key=lambda item: order_key("github-file", name, item["path"]))
        license_text = cache.get(f"gh-license-{name}", lambda name=name: gh_api(
            f"repos/{name}/license", raw=True).decode("utf-8", "replace"))
        copyright_lines = [  # Actual notices, not license boilerplate or the [yyyy] template.
            line.strip() for line in license_text.splitlines()
            if re.match(r"\s*copyright\s+(\(c\)|©|\d{4})", line, re.IGNORECASE)
        ][:3]
        used_repos += 1
        for item in files[:8]:
            text = cache.get(f"gh-blob-{name}-{info['sha']}-{item['path']}",
                             lambda name=name, item=item: gh_api(
                                 f"repos/{name}/contents/{urllib.parse.quote(item['path'])}"
                                 f"?ref={info['sha']}", raw=True).decode("utf-8", "replace"))
            if "�" in text:
                continue
            text = cap_tokens(normalize(text), tokenizer, DOC_TOKEN_CAP // 2)
            documents.append(Document(
                kind="github", key=f"{name}:{item['path']}", title=item["path"], text=text,
                provenance={
                    "source": "github.com", "repository": name, "commit": info["sha"],
                    "path": item["path"], "created": repo["createdAt"],
                    "license": (repo.get("license") or {}).get("key", ""),
                    "copyright": copyright_lines,
                },
                extra={"language": language},
            ))
    return documents


# --- Own code ---------------------------------------------------------------------------------


def own_code_documents(tokenizer: Tokenizer) -> list[Document]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    log = subprocess.run(
        ["git", "log", f"--since={OWN_CODE_SINCE}", "--diff-filter=A", "--name-only",
         "--format=@%an|%H|%aI"], cwd=ROOT, check=True, capture_output=True, text=True,
        encoding="utf-8").stdout
    added: dict[str, tuple[str, str]] = {}
    author = commit = date = ""
    for line in log.splitlines():
        if line.startswith("@"):
            author, commit, date = line[1:].split("|")
        elif line and author in OWN_CODE_AUTHORS:
            added.setdefault(line, (commit, date))  # git log is newest first; keep the newest add.
    documents = []
    for path, (commit, date) in sorted(added.items()):
        file = ROOT / path
        if (not path.endswith(OWN_CODE_SUFFIXES) or path.startswith(OWN_CODE_EXCLUDE)
                or not file.is_file()):
            continue
        text = file.read_text(encoding="utf-8", errors="replace")
        if "�" in text or len(text) < 800:
            continue
        text = cap_tokens(normalize(text), tokenizer, DOC_TOKEN_CAP // 2)
        documents.append(Document(
            kind="own_code", key=path, title=path, text=text,
            provenance={"source": "ashalliants/ninfer-3090", "commit": head, "path": path,
                        "added_commit": commit, "added": date},
        ))
    documents.sort(key=lambda doc: order_key("own", doc.key))
    return documents


# --- Synthetic chat ---------------------------------------------------------------------------

USER_PROMPTS = {
    "wiki_en": ("Tell me about {title}.", "Can you write an encyclopedia-style article on {title}?",
                "What should I know about {title}?"),
    "wiki_zh": ("请介绍一下{title}。", "能写一篇关于{title}的百科条目吗？", "{title}是什么？"),
    "arxiv": ('Write the abstract for a paper titled "{title}".',
              'Summarize the paper "{title}" in one paragraph.'),
    "code": ("Show me `{title}`.", "Write `{title}` for this project.",
             "Can you give me the full contents of {title}?"),
}


def render_chat(chat: PreTrainedTokenizerFast, messages: list[dict]) -> str:
    return chat.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                    enable_thinking=False)


def chat_documents(pool: list[Document], chat: PreTrainedTokenizerFast) -> list[Document]:
    # Interleave the leftover documents by kind so every conversation mixes subjects.
    by_kind: dict[str, list[Document]] = {}
    for doc in sorted(pool, key=lambda doc: order_key("chat", doc.kind, doc.key)):
        by_kind.setdefault(doc.kind, []).append(doc)
    pool = [doc for group in zip(*by_kind.values()) for doc in group]
    conversations = []
    for start in range(0, len(pool) - 2, 3):
        turns, sources = [], []
        for doc in pool[start : start + 3]:
            family = ("arxiv" if doc.kind.startswith("arxiv") else
                      "code" if doc.kind in ("github", "own_code") else doc.kind)
            prompts = USER_PROMPTS[family]
            prompt = prompts[int(order_key("prompt", doc.key), 16) % len(prompts)]
            turns += [{"role": "user", "content": prompt.format(title=doc.title)},
                      {"role": "assistant", "content": doc.text.strip()}]
            sources.append({"kind": doc.kind, **doc.provenance})
        conversations.append(Document(
            kind="chat", key=str(start), text=render_chat(chat, turns),
            provenance={"source": "synthetic chat", "template": "artifact chat_template.jinja",
                        "enable_thinking": False, "turns": sources},
        ))
    return conversations


# --- Assembly ---------------------------------------------------------------------------------


def take_streams(documents: list[Document], tokenizer: Tokenizer, count: int,
                 separator: str = "\n\n") -> tuple[list[list[Document]], list[Document]]:
    """Fill ``count`` streams round-robin to STREAM_TOKENS; return the streams and leftovers."""
    for doc in documents:
        doc.tokens = len(tokenizer.encode(doc.text).ids)
    streams: list[list[Document]] = [[] for _ in range(count)]
    totals = [0] * count
    leftovers = []
    for doc in documents:
        open_streams = [i for i in range(count) if totals[i] + doc.tokens <= STREAM_TOKENS]
        if not open_streams or min(totals) >= STREAM_TOKENS - 512:
            leftovers.append(doc)
            continue
        target = min(open_streams, key=lambda i: totals[i])
        streams[target].append(doc)
        totals[target] += doc.tokens + 1
    short = [total for total in totals if total < STREAM_TOKENS * 0.9]
    if short:
        raise RuntimeError(f"{documents[0].kind if documents else '?'}: streams short {totals}")
    return streams, leftovers


def write_stream(out: Path, domain: str, index: int, docs: list[Document],
                 separator: str) -> tuple[str, str]:
    stream_id = f"{domain}-{index:02d}"
    relative = f"data/{domain}/{index:02d}.txt"
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = separator.join(doc.text.rstrip("\n") for doc in docs) + "\n"
    path.write_bytes(text.encode("utf-8"))
    segments, offset = [], 0
    for position, doc in enumerate(docs):
        body = doc.text.rstrip("\n").encode("utf-8")
        segments.append({**doc.provenance, "start_byte": offset, "end_byte": offset + len(body)})
        offset += len(body) + len(separator.encode("utf-8"))
    with (out / "provenance" / f"{domain}.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps({"stream_id": stream_id, "domain": domain, "segments": segments},
                                ensure_ascii=False) + "\n")
    return stream_id, relative


NOTICES_HEAD = """# Third-Party Notices

The files below are evaluation inputs. They retain the licenses and attribution requirements of
their sources and are not relicensed as NInfer source code. This file is generated by
`tools/eval/build_heldout_corpus.py` from `provenance/*.jsonl`.

## Wikipedia (English and Chinese)

`data/wikipedia_en/`, `data/wikipedia_zh/` and the Wikipedia turns of `data/synthetic_chat/` contain
plain-text extracts of articles created in September 2026. Wikipedia text is available under the
Creative Commons Attribution-ShareAlike 4.0 International license. Each provenance segment records
the article title, page ID, creating and extracted revision IDs, and its URL
`https://<lang>.wikipedia.org/?curid=<pageid>`, which lists the article's authors.

- <https://creativecommons.org/licenses/by-sa/4.0/>
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>

## arXiv

`data/arxiv/` and the arXiv turns of `data/synthetic_chat/` contain titles and abstracts of
September 2026 submissions. arXiv releases its metadata, including abstracts, under CC0 1.0. Each
segment records the arXiv identifier and URL.

- <https://info.arxiv.org/help/api/tou.html>
- <https://creativecommons.org/publicdomain/zero/1.0/>

## NInfer source

`data/own_code/` and the own-code turns of `data/synthetic_chat/` are files from this repository at
the commit recorded in `provenance/own_code.jsonl` and remain covered by the repository's Apache
License 2.0.

## GitHub repositories

`data/github_code/` and the GitHub turns of `data/synthetic_chat/` contain source files from the
repositories below at the recorded commits. Each file remains under its repository's license; the
copyright lines are reproduced from each repository's license file.

- MIT: <https://opensource.org/license/mit>
- Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>

| Repository | Commit | License | Copyright |
|---|---|---|---|
"""


def write_notices(out: Path) -> None:
    repos: dict[str, dict] = {}
    for path in sorted((out / "provenance").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            for segment in json.loads(line)["segments"]:
                for source in segment.get("turns", [segment]):
                    if source.get("source") == "github.com":
                        repos[source["repository"]] = source
    rows = [
        f"| [{name}](https://github.com/{name}) | `{info['commit'][:12]}` | {info['license']} | "
        + ("; ".join(info["copyright"]).replace("|", "\\|") or "(none stated)") + " |"
        for name, info in sorted(repos.items(), key=lambda item: item[0].lower())
    ]
    (out / "THIRD_PARTY_NOTICES.md").write_text(NOTICES_HEAD + "\n".join(rows) + "\n",
                                               encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with Artifact.open(args.artifact) as artifact:
        tokenizer = Tokenizer.from_str(artifact.read_object("frontend/tokenizer.json").decode())
        chat = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        chat.chat_template = artifact.read_object("frontend/chat_template.jinja").decode()
    cache = Cache(args.cache)

    out = args.out
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    for stale in (out / "provenance").glob("*.jsonl"):
        stale.unlink()

    sources = {
        "wikipedia_en": wikipedia_documents("en", cache, tokenizer),
        "wikipedia_zh": wikipedia_documents("zh", cache, tokenizer),
        "arxiv_math": arxiv_documents("math", cache),
        "arxiv_cs": arxiv_documents("cs", cache),
        "github_code": [doc for language in GITHUB_LANGUAGES
                        for doc in github_documents(language, cache, tokenizer)],
        "own_code": own_code_documents(tokenizer),
    }
    sources["github_code"].sort(key=lambda doc: order_key("github-order", doc.key))

    layout = {  # domain -> (source keys, separator)
        "wikipedia_en": (("wikipedia_en",), "\n\n\n"),
        "wikipedia_zh": (("wikipedia_zh",), "\n\n\n"),
        "arxiv": (("arxiv_math", "arxiv_cs"), "\n\n"),
        "github_code": (("github_code",), "\n\n\n"),
        "own_code": (("own_code",), "\n\n\n"),
    }
    manifest_streams, quick, full, chat_pool, counts = [], [], [], [], {}
    for domain, (keys, separator) in layout.items():
        if domain == "arxiv":
            # Stream 00 is mathematics and stream 01 is computing, so quick mode measures math.
            streams = []
            for key in keys:
                taken, rest = take_streams(sources[key], tokenizer, 1)
                streams += taken
                chat_pool += rest
        else:
            docs = sources[keys[0]]
            streams, rest = take_streams(docs, tokenizer, STREAMS_PER_DOMAIN)
            chat_pool += rest
        for index, docs in enumerate(streams):
            stream_id, relative = write_stream(out, domain, index, docs, separator)
            manifest_streams.append({"id": stream_id, "domain": domain, "path": relative})
            (quick if index == 0 else full).append(stream_id)
            counts[stream_id] = sum(doc.tokens for doc in docs)

    chats = chat_documents(chat_pool, chat)
    streams, _ = take_streams(chats, tokenizer, STREAMS_PER_DOMAIN, "\n")
    for index, docs in enumerate(streams):
        stream_id, relative = write_stream(out, "synthetic_chat", index, docs, "\n")
        manifest_streams.append({"id": stream_id, "domain": "synthetic_chat", "path": relative})
        (quick if index == 0 else full).append(stream_id)
        counts[stream_id] = sum(doc.tokens for doc in docs)

    manifest = {"corpus_id": CORPUS_ID, "streams": manifest_streams,
                "modes": {"quick": quick, "full": quick + full}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                       newline="\n")
    generation = {
        "corpus_id": CORPUS_ID,
        "builder": "tools/eval/build_heldout_corpus.py",
        "window": {"start": WINDOW_START, "end": WINDOW_END, "own_code_since": OWN_CODE_SINCE},
        "sizing": {"role": "approximate stream boundary selection only",
                   "tokenizer": f"{args.artifact.name} frontend/tokenizer.json",
                   "add_special_tokens": False, "approximate_tokens_per_stream": STREAM_TOKENS},
        "approximate_tokens": counts,
    }
    (out / "provenance" / "generation.json").write_text(
        json.dumps(generation, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_notices(out)
    for stream_id, tokens in counts.items():
        print(f"{stream_id:<18} {tokens:>7} tokens")


if __name__ == "__main__":
    main()
