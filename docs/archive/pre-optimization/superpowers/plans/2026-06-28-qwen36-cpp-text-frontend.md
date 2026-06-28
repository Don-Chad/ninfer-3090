# Qwen3.6 C++ Text Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primary `qus` token-id CLI with a Qwen3.6 text frontend that accepts `--prompt` or text-only `--messages`, runs the existing token-id `Engine`, and prints decoded text.

**Architecture:** Add a host-only `qus::text` layer above `Engine`. The text layer owns tokenizer file loading, Qwen byte-level BPE, text-only chat-template rendering, generation input validation, and detokenization; `Engine` stays token-id based. `bench/` and `tools/parity/` remain token-id oriented and tokenizer-free.

**Tech Stack:** C++20, CUDA runtime through existing `qus_core`, vendored `utf8proc` for Unicode NFC/category primitives, existing `nlohmann/json.hpp`, Python/HF only for generating golden tokenizer fixtures.

---

## Source Spec

Implement the approved design:

```text
docs/superpowers/specs/2026-06-28-qwen36-cpp-text-frontend-design.md
```

## File Map

Create:

- `third_party/utf8proc/utf8proc.h`
- `third_party/utf8proc/utf8proc.c`
- `third_party/utf8proc/utf8proc_data.c`
- `third_party/utf8proc/LICENSE.md`
- `tools/text/generate_qwen_text_golden.py`
- `tests/fixtures/text/qwen36_text_golden.json`
- `include/qus/text/tokenizer.h`
- `include/qus/text/chat_template.h`
- `include/qus/text/text_runner.h`
- `include/qus/text/cli.h`
- `src/text/unicode.cpp`
- `src/text/tokenizer.cpp`
- `src/text/chat_template.cpp`
- `src/text/text_runner.cpp`
- `src/text/cli.cpp`
- `tests/test_qwen_text_tokenizer.cpp`
- `tests/test_qwen_chat_template.cpp`
- `tests/test_qwen_text_cli.cpp`

Modify:

- `CMakeLists.txt`
- `src/CMakeLists.txt`
- `src/main.cpp`
- `tests/CMakeLists.txt`
- `README.md`
- `docs/design.md`

Do not modify for this feature:

- `bench/e2e_bench.cpp`
- `bench/e2e_bench_support.*`
- `tools/parity/*`
- `src/runtime/engine.cpp`, unless a compile-time header include fix is required.

## Commit Sequence

Each task ends with a commit. Keep commits narrow so tokenizer failures, CLI behavior, and docs can be reviewed separately.

---

### Task 1: Vendor `utf8proc` and Wire CMake

**Files:**
- Create: `third_party/utf8proc/utf8proc.h`
- Create: `third_party/utf8proc/utf8proc.c`
- Create: `third_party/utf8proc/utf8proc_data.c`
- Create: `third_party/utf8proc/LICENSE.md`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Download pinned utf8proc sources**

Use the pinned release from the upstream project:

```bash
mkdir -p /tmp/qus_utf8proc
curl -L \
  -o /tmp/qus_utf8proc/utf8proc-2.11.3.tar.gz \
  https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.11.3.tar.gz
tar -xf /tmp/qus_utf8proc/utf8proc-2.11.3.tar.gz -C /tmp/qus_utf8proc
mkdir -p third_party/utf8proc
cp /tmp/qus_utf8proc/utf8proc-2.11.3/utf8proc.h third_party/utf8proc/
cp /tmp/qus_utf8proc/utf8proc-2.11.3/utf8proc.c third_party/utf8proc/
cp /tmp/qus_utf8proc/utf8proc-2.11.3/utf8proc_data.c third_party/utf8proc/
cp /tmp/qus_utf8proc/utf8proc-2.11.3/LICENSE.md third_party/utf8proc/
```

Expected: the four files exist under `third_party/utf8proc/`.

- [ ] **Step 2: Include utf8proc in `qus_core` build**

Modify `src/CMakeLists.txt` so the core source glob includes text frontend sources and utf8proc:

```cmake
file(GLOB_RECURSE QUS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp     ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp    ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/*.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
```

Add private include directories:

```cmake
target_include_directories(qus_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
```

Keep the existing public include and CUDA link lines.

- [ ] **Step 3: Configure build**

Run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

Expected: configure completes and mentions no missing utf8proc package because the source is vendored.

- [ ] **Step 4: Commit**

Run:

```bash
git add third_party/utf8proc src/CMakeLists.txt
git commit -m "build: vendor utf8proc for text frontend"
```

---

### Task 2: Generate and Commit HF Golden Text Fixtures

**Files:**
- Create: `tools/text/generate_qwen_text_golden.py`
- Create: `tests/fixtures/text/qwen36_text_golden.json`

- [ ] **Step 1: Add golden fixture generator**

Create `tools/text/generate_qwen_text_golden.py` with this interface:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES: list[dict[str, Any]] = [
    {"name": "ascii", "text": "Hello, world!"},
    {"name": "chinese", "text": "你好，世界！"},
    {"name": "leading_space", "text": " leading space"},
    {"name": "trailing_space", "text": "trailing space "},
    {"name": "multi_space", "text": "alpha  beta   gamma"},
    {"name": "newline", "text": "line1\nline2"},
    {"name": "emoji", "text": "emoji 😀 test"},
    {"name": "combining_nfc", "text": "Cafe\u0301"},
    {"name": "code", "text": "def f(xs):\n    return xs or []\n"},
    {"name": "chat_markers", "text": "<|im_start|>user\nhi<|im_end|>\n"},
    {"name": "thinking_markers", "text": "<think>\n\n</think>\n\n"},
]

MESSAGES_CASES: list[dict[str, Any]] = [
    {
        "name": "prompt_user_cn",
        "messages": [{"role": "user", "content": "你好，简单介绍一下你自己。"}],
    },
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Describe prefill briefly."},
        ],
    },
    {
        "name": "assistant_history",
        "messages": [
            {"role": "user", "content": "Say one word."},
            {"role": "assistant", "content": "Ready."},
            {"role": "user", "content": "Now say two words."},
        ],
    },
]


def load_tokenizer(tokenizer_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer_path)
    text_cases = []
    for case in CASES:
        ids = tok.encode(case["text"], add_special_tokens=False)
        text_cases.append(
            {
                "name": case["name"],
                "text": case["text"],
                "ids": list(ids),
                "raw_decoded": tok.decode(ids, skip_special_tokens=False),
                "clean_decoded": tok.decode(ids, skip_special_tokens=True),
            }
        )

    message_cases = []
    for case in MESSAGES_CASES:
        rendered = tok.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        ids = tok.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        message_cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "rendered": rendered,
                "ids": list(ids),
                "raw_decoded": tok.decode(ids, skip_special_tokens=False),
                "clean_decoded": tok.decode(ids, skip_special_tokens=True),
            }
        )

    value = {
        "tokenizer_model_id": "Qwen/Qwen3.6-27B",
        "text_cases": text_cases,
        "message_cases": message_cases,
        "default_stop_token_ids": [248046, 248044],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Generate golden fixture**

Run:

```bash
python3 tools/text/generate_qwen_text_golden.py \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out tests/fixtures/text/qwen36_text_golden.json
```

Expected: command exits 0 and writes `tests/fixtures/text/qwen36_text_golden.json`.

- [ ] **Step 3: Inspect key fixture values**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
value = json.loads(Path("tests/fixtures/text/qwen36_text_golden.json").read_text(encoding="utf-8"))
print(value["text_cases"][0]["ids"])
print(value["message_cases"][0]["ids"])
print(value["message_cases"][0]["rendered"])
PY
```

Expected:

- ASCII ids include `[9419, 11, 1814, 0]`.
- The first message case starts with token id `248045`.
- Rendered prompt includes `<think>\n\n</think>\n\n`.

- [ ] **Step 4: Commit**

Run:

```bash
git add tools/text/generate_qwen_text_golden.py tests/fixtures/text/qwen36_text_golden.json
git commit -m "test: add qwen text frontend golden fixtures"
```

---

### Task 3: Add Text Frontend Public Types and CLI Parser Tests

**Files:**
- Create: `include/qus/text/cli.h`
- Create: `src/text/cli.cpp`
- Create: `tests/test_qwen_text_cli.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add CLI header**

Create `include/qus/text/cli.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qus::text {

enum class OutputMode {
    Clean,
    Raw,
};

struct CliOptions {
    bool help_requested = false;
    std::string weights_path;
    std::string tokenizer_path;
    std::string prompt;
    std::string messages_path;
    int max_new = 128;
    std::uint32_t max_context = 2048;
    int device = 0;
    OutputMode output_mode = OutputMode::Clean;
    bool print_token_ids = false;
    std::vector<int> stop_token_ids;
};

CliOptions parse_cli(int argc, char** argv);
std::string usage_text(const char* argv0);

} // namespace qus::text
```

- [ ] **Step 2: Add failing CLI parser tests**

Create `tests/test_qwen_text_cli.cpp`:

```cpp
#include "qus/text/cli.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

qus::text::CliOptions parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) { argv.push_back(arg.data()); }
    return qus::text::parse_cli(static_cast<int>(argv.size()), argv.data());
}

template <class Exception, class Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) {
        return 0;
    }
    return fail(std::string(label) + " did not throw");
}

int test_prompt_mode() {
    const auto options = parse({"qus", "weights.qus", "--tokenizer", "tok", "--prompt", "hi"});
    if (options.weights_path != "weights.qus") { return fail("weights path mismatch"); }
    if (options.tokenizer_path != "tok") { return fail("tokenizer path mismatch"); }
    if (options.prompt != "hi") { return fail("prompt mismatch"); }
    if (!options.messages_path.empty()) { return fail("messages path should be empty"); }
    if (options.max_new != 128) { return fail("default max_new mismatch"); }
    if (options.max_context != 2048) { return fail("default max_context mismatch"); }
    return 0;
}

int test_messages_mode() {
    const auto options =
        parse({"qus", "weights.qus", "--tokenizer", "tok", "--messages", "chat.json",
               "--max-new", "16", "--max-context", "4096", "--device", "1", "--raw-output",
               "--print-token-ids", "--stop-token-id", "248046", "--stop-token-id", "248044"});
    if (options.messages_path != "chat.json") { return fail("messages path mismatch"); }
    if (options.max_new != 16) { return fail("max_new mismatch"); }
    if (options.max_context != 4096) { return fail("max_context mismatch"); }
    if (options.device != 1) { return fail("device mismatch"); }
    if (options.output_mode != qus::text::OutputMode::Raw) { return fail("raw output mismatch"); }
    if (!options.print_token_ids) { return fail("print token ids mismatch"); }
    if (options.stop_token_ids != std::vector<int>{248046, 248044}) {
        return fail("stop token ids mismatch");
    }
    return 0;
}

int test_rejections() {
    int failures = 0;
    failures += expect_throws<std::invalid_argument>(
        [] { (void)parse({"qus", "weights.qus", "--tokenizer", "tok"}); },
        "missing input");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)parse({"qus", "weights.qus", "--prompt", "hi"}); },
        "missing tokenizer");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)parse({"qus", "weights.qus", "--tokenizer", "tok", "--prompt", "hi",
                          "--messages", "chat.json"}); },
        "two inputs");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)parse({"qus", "weights.qus", "--tokenizer", "tok", "--prompt", "hi",
                          "--max-new", "0"}); },
        "zero max-new");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prompt_mode();
    failures += test_messages_mode();
    failures += test_rejections();
    return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 3: Register CLI test**

Modify `tests/CMakeLists.txt`:

```cmake
qus_add_test(qus_qwen_text_cli_test SOURCES test_qwen_text_cli.cpp)
```

- [ ] **Step 4: Run failing test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_cli_test
```

Expected: build fails because `src/text/cli.cpp` is empty or `parse_cli` is undefined.

- [ ] **Step 5: Implement CLI parser**

Create `src/text/cli.cpp` with direct argument parsing:

```cpp
#include "qus/text/cli.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace qus::text {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <weights.qus> --tokenizer <dir> (--prompt <text>|--messages <messages.json>) "
           "[--max-context N] [--max-new N] [--device N] [--raw-output] "
           "[--print-token-ids] [--stop-token-id N]...\n";
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("weights path is required"); }
    options.weights_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--tokenizer") {
            options.tokenizer_path = require_value("--tokenizer");
        } else if (arg == "--prompt") {
            options.prompt = require_value("--prompt");
        } else if (arg == "--messages") {
            options.messages_path = require_value("--messages");
        } else if (arg == "--max-new") {
            options.max_new = parse_nonnegative_int(require_value("--max-new"), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--raw-output") {
            options.output_mode = OutputMode::Raw;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--stop-token-id") {
            options.stop_token_ids.push_back(
                parse_nonnegative_int(require_value("--stop-token-id"), "stop-token-id"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.tokenizer_path.empty()) { throw std::invalid_argument("--tokenizer is required"); }
    const bool has_prompt = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.max_new <= 0) { throw std::invalid_argument("--max-new must be positive"); }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    return options;
}

} // namespace qus::text
```

- [ ] **Step 6: Run CLI test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_cli_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_cli_test
```

Expected: test passes.

- [ ] **Step 7: Commit**

Run:

```bash
git add include/qus/text/cli.h src/text/cli.cpp tests/test_qwen_text_cli.cpp tests/CMakeLists.txt
git commit -m "feat: add qwen text cli parser"
```

---

### Task 4: Add Chat Message Parser and Renderer

**Files:**
- Create: `include/qus/text/chat_template.h`
- Create: `src/text/chat_template.cpp`
- Create: `tests/test_qwen_chat_template.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add chat-template header**

Create `include/qus/text/chat_template.h`:

```cpp
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qus::text {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;
};

std::vector<ChatMessage> messages_from_prompt(std::string prompt);
std::vector<ChatMessage> read_messages_json(const std::filesystem::path& path);
std::string render_qwen_chat(const std::vector<ChatMessage>& messages,
                             ChatRenderOptions options = {});

} // namespace qus::text
```

- [ ] **Step 2: Add failing chat tests**

Create `tests/test_qwen_chat_template.cpp`:

```cpp
#include "qus/text/chat_template.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

template <class Exception, class Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) {
        return 0;
    }
    return fail(std::string(label) + " did not throw");
}

std::filesystem::path write_temp(std::string_view name, std::string_view text) {
    const auto path = std::filesystem::temp_directory_path() / std::string(name);
    std::ofstream out(path);
    out << text;
    return path;
}

int test_prompt_render() {
    const auto messages = qus::text::messages_from_prompt("你好");
    const std::string rendered = qus::text::render_qwen_chat(messages);
    const std::string expected =
        "<|im_start|>user\n"
        "你好<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    return rendered == expected ? 0 : fail("prompt render mismatch");
}

int test_messages_json() {
    const auto path = write_temp(
        "qus_messages_ok.json",
        "[{\"role\":\"system\",\"content\":\"You are concise.\"},"
        "{\"role\":\"user\",\"content\":\"Explain prefill.\"}]");
    const auto messages = qus::text::read_messages_json(path);
    if (messages.size() != 2) { return fail("messages size mismatch"); }
    if (messages[0].role != "system") { return fail("system role mismatch"); }
    const std::string rendered = qus::text::render_qwen_chat(messages);
    if (rendered.find("<|im_start|>system\nYou are concise.<|im_end|>\n") != 0) {
        return fail("system prefix mismatch");
    }
    if (rendered.find("<|im_start|>user\nExplain prefill.<|im_end|>\n") == std::string::npos) {
        return fail("user message missing");
    }
    return 0;
}

int test_rejections() {
    int failures = 0;
    failures += expect_throws<std::invalid_argument>(
        [] { (void)qus::text::render_qwen_chat({}); },
        "empty messages");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)qus::text::render_qwen_chat({{"tool", "x"}}); },
        "tool role");
    const auto multimodal = write_temp(
        "qus_messages_multimodal.json",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\"}]}]");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::text::read_messages_json(multimodal); },
        "multimodal content");
    const auto tool_calls = write_temp(
        "qus_messages_tool_calls.json",
        "[{\"role\":\"assistant\",\"content\":\"x\",\"tool_calls\":[]}]");
    failures += expect_throws<std::invalid_argument>(
        [&] { (void)qus::text::read_messages_json(tool_calls); },
        "tool calls");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prompt_render();
    failures += test_messages_json();
    failures += test_rejections();
    return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 3: Register chat test**

Modify `tests/CMakeLists.txt`:

```cmake
qus_add_test(qus_qwen_chat_template_test SOURCES test_qwen_chat_template.cpp)
target_include_directories(qus_qwen_chat_template_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
```

- [ ] **Step 4: Run failing test**

Run:

```bash
cmake --build build -j --target qus_qwen_chat_template_test
```

Expected: build fails because chat-template functions are undefined.

- [ ] **Step 5: Implement chat parser and renderer**

Create `src/text/chat_template.cpp` using `nlohmann/json.hpp`. Required behavior:

```cpp
std::vector<ChatMessage> messages_from_prompt(std::string prompt) {
    if (prompt.empty()) { throw std::invalid_argument("--prompt text is empty"); }
    return {ChatMessage{"user", std::move(prompt)}};
}
```

For JSON parsing:

- require top-level array;
- reject empty arrays;
- require each item to contain exactly `role` and `content`;
- allow only roles `system`, `user`, `assistant`;
- require `content` to be a non-empty string;
- reject any item containing `tool_calls`, `reasoning_content`, `name`, or multimodal content.

For rendering:

```cpp
out += "<|im_start|>" + role + "\n" + trim(content) + "<|im_end|>\n";
out += "<|im_start|>assistant\n";
out += "<think>\n\n</think>\n\n";
```

Use ASCII whitespace trim for rendered message content to match the supported fixture behavior.

- [ ] **Step 6: Run chat tests**

Run:

```bash
cmake --build build -j --target qus_qwen_chat_template_test
ctest --test-dir build --output-on-failure -R qus_qwen_chat_template_test
```

Expected: test passes.

- [ ] **Step 7: Commit**

Run:

```bash
git add include/qus/text/chat_template.h src/text/chat_template.cpp tests/test_qwen_chat_template.cpp tests/CMakeLists.txt
git commit -m "feat: add qwen text chat renderer"
```

---

### Task 5: Add Tokenizer Interfaces, Metadata Loader, and Stop IDs

**Files:**
- Create: `include/qus/text/tokenizer.h`
- Create: `src/text/tokenizer.cpp`
- Create: `src/text/unicode.cpp`
- Create: `tests/test_qwen_text_tokenizer.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add tokenizer header**

Create `include/qus/text/tokenizer.h`:

```cpp
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qus::text {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
    bool special = false;
};

class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::filesystem::path& tokenizer_dir);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] const std::vector<AddedToken>& added_tokens() const noexcept {
        return added_tokens_;
    }

private:
    std::filesystem::path tokenizer_dir_;
    std::vector<std::string> id_to_token_;
    std::vector<AddedToken> added_tokens_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace qus::text
```

- [ ] **Step 2: Add tokenizer fixture loading tests**

Create the first version of `tests/test_qwen_text_tokenizer.cpp`:

```cpp
#include "qus/text/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path repo_file(std::string_view relative) {
#ifdef QUS_SOURCE_DIR
    return std::filesystem::path(QUS_SOURCE_DIR) / std::string(relative);
#else
    return std::filesystem::path(relative);
#endif
}

int test_load_real_tokenizer_metadata() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping real tokenizer metadata test: local tokenizer not present\n";
        return 0;
    }
    const qus::text::QwenTokenizer tok(tokenizer_path);
    if (tok.default_stop_token_ids() != std::vector<int>{248046, 248044}) {
        return fail("default stop token ids mismatch");
    }
    bool saw_im_start = false;
    bool saw_think = false;
    for (const auto& token : tok.added_tokens()) {
        if (token.id == 248045 && token.content == "<|im_start|>" && token.special) {
            saw_im_start = true;
        }
        if (token.id == 248068 && token.content == "<think>" && !token.special) {
            saw_think = true;
        }
    }
    if (!saw_im_start) { return fail("missing im_start added token"); }
    if (!saw_think) { return fail("missing think added token"); }
    return 0;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_load_real_tokenizer_metadata();
    return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 3: Register tokenizer test**

Modify `tests/CMakeLists.txt`:

```cmake
qus_add_test(qus_qwen_text_tokenizer_test
             SOURCES test_qwen_text_tokenizer.cpp
             NEEDS_SOURCE_DIR)
target_include_directories(qus_qwen_text_tokenizer_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
```

- [ ] **Step 4: Run failing test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
```

Expected: build fails because `QwenTokenizer` constructor and methods are undefined.

- [ ] **Step 5: Implement metadata loading**

In `src/text/tokenizer.cpp`, implement:

- file reads using `std::ifstream`;
- JSON parsing using `nlohmann::json`;
- required file validation for `tokenizer.json`;
- `model.type == "BPE"` validation;
- `id_to_token_` sized to the maximum token id plus one;
- base vocab load from `model.vocab`;
- added token load from `added_tokens`;
- default stop ids from `generation_config.json` `eos_token_id`, accepting an integer or array;
- fixed fallback `[248046, 248044]` only when `generation_config.json` is absent.

The constructor must throw `std::invalid_argument` with the path or field name when a required file
or field is malformed.

- [ ] **Step 6: Run tokenizer metadata test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: test passes or prints the documented skip line if the local HF tokenizer directory is absent.

- [ ] **Step 7: Commit**

Run:

```bash
git add include/qus/text/tokenizer.h src/text/tokenizer.cpp src/text/unicode.cpp tests/test_qwen_text_tokenizer.cpp tests/CMakeLists.txt
git commit -m "feat: load qwen tokenizer metadata"
```

---

### Task 6: Implement Decode and Added-Token Encoding

**Files:**
- Modify: `src/text/tokenizer.cpp`
- Modify: `tests/test_qwen_text_tokenizer.cpp`

- [ ] **Step 1: Extend tokenizer tests for exact added tokens and decode**

Add to `tests/test_qwen_text_tokenizer.cpp`:

```cpp
int test_added_token_encode_decode() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping added token test: local tokenizer not present\n";
        return 0;
    }
    const qus::text::QwenTokenizer tok(tokenizer_path);
    const std::vector<int> ids = tok.encode("<|im_start|><|im_end|><think></think>");
    if (ids != std::vector<int>{248045, 248046, 248068, 248069}) {
        return fail("added token ids mismatch");
    }
    const std::string raw = tok.decode(ids, qus::text::DecodeOptions{false, {}});
    if (raw != "<|im_start|><|im_end|><think></think>") { return fail("raw decode mismatch"); }
    const std::string clean = tok.decode(ids, qus::text::DecodeOptions{true, {}});
    if (clean != "<think></think>") { return fail("clean decode mismatch"); }
    return 0;
}
```

Call it from `main()`.

- [ ] **Step 2: Run failing tokenizer test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: test fails because ordinary BPE or byte-level decode is not complete.

- [ ] **Step 3: Implement token-to-piece decode**

In `src/text/tokenizer.cpp`:

- if `skip_special_tokens` and `is_special_token(id)`, skip the token;
- if `id` is a terminal stop token from `DecodeOptions::stop_token_ids`, skip only the final matching id;
- map every remaining id to `id_to_token_[id]`;
- for added tokens, append content directly;
- for byte-level vocabulary tokens, inverse-map byte-level characters to bytes and then UTF-8.

Implement the GPT/Qwen byte mapping used by ByteLevel:

- bytes `33..126`, `161..172`, `174..255` map to themselves;
- remaining bytes map to Unicode codepoints starting at 256 in encounter order.

- [ ] **Step 4: Implement added-token partitioning**

In `encode`, before ordinary BPE:

- scan the input left to right;
- match the earliest added token occurrence among loaded added tokens;
- emit text fragments and token fragments;
- support exact content matching for all added tokens;
- for first implementation, enforce `single_word=false`, `lstrip=false`, and `rstrip=false` as in Qwen3.6 loaded metadata, and throw if a future tokenizer changes these fields.

The text fragments can still use a temporary byte-level no-merge path until Task 7 completes.

- [ ] **Step 5: Run added-token test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: added-token encode/decode test passes. Ordinary text fragments are covered by the HF
golden tests in Task 7.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/text/tokenizer.cpp tests/test_qwen_text_tokenizer.cpp
git commit -m "feat: encode and decode qwen added tokens"
```

---

### Task 7: Implement Qwen Unicode Pre-Tokenizer and BPE Merge Engine

**Files:**
- Modify: `src/text/unicode.cpp`
- Modify: `src/text/tokenizer.cpp`
- Modify: `tests/test_qwen_text_tokenizer.cpp`

- [ ] **Step 1: Extend tests to use committed golden fixture**

Add to `tests/test_qwen_text_tokenizer.cpp`:

```cpp
int test_hf_golden_text_cases() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping golden text cases: local tokenizer not present\n";
        return 0;
    }
    const auto fixture_path = repo_file("tests/fixtures/text/qwen36_text_golden.json");
    std::ifstream in(fixture_path);
    const auto fixture = nlohmann::json::parse(in);
    const qus::text::QwenTokenizer tok(tokenizer_path);
    for (const auto& item : fixture.at("text_cases")) {
        const std::string name = item.at("name").get<std::string>();
        const std::string text = item.at("text").get<std::string>();
        const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
        const std::vector<int> actual = tok.encode(text);
        if (actual != expected) {
            std::cerr << "golden encode mismatch for " << name << '\n';
            return 1;
        }
        const std::string raw = tok.decode(expected, qus::text::DecodeOptions{false, {}});
        if (raw != item.at("raw_decoded").get<std::string>()) {
            std::cerr << "golden raw decode mismatch for " << name << '\n';
            return 1;
        }
    }
    return 0;
}
```

Call it from `main()`.

- [ ] **Step 2: Run failing golden test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: test fails on ordinary text cases until Qwen pre-tokenization and BPE merges are implemented.

- [ ] **Step 3: Implement UTF-8 and category helpers**

In `src/text/unicode.cpp`, implement internal helpers used by `tokenizer.cpp`:

- `std::string normalize_nfc(std::string_view text)` using `utf8proc_map` with `UTF8PROC_STABLE | UTF8PROC_COMPOSE`;
- UTF-8 codepoint iteration using `utf8proc_iterate`;
- category predicates:
  - letter: `UTF8PROC_CATEGORY_LU`, `LL`, `LT`, `LM`, `LO`;
  - mark: `MN`, `MC`, `ME`;
  - number: `ND`, `NL`, `NO`;
  - whitespace: ASCII whitespace plus codepoints where `utf8proc_category` returns separators.

Keep helper declarations private to `tokenizer.cpp` through an unnamed namespace or a small internal
header under `src/text/` if needed.

- [ ] **Step 4: Implement Qwen split state machine**

Implement split behavior equivalent to:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)
| [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
| \p{N}
| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
| \s*[\r\n]+
| \s+(?!\S)
| \s+
```

Use codepoint spans and convert each split back to its source UTF-8 substring.

- [ ] **Step 5: Implement BPE merge engine**

In `src/text/tokenizer.cpp`:

- load merge ranks from `merges.txt`, skipping `#version` header;
- after Qwen split, convert each word to byte-level representation;
- split each byte-level word into UTF-8 codepoint symbols;
- repeatedly merge the adjacent pair with the lowest rank;
- append the final symbols' token ids from `id_to_token_` reverse map;
- if a final symbol has no token id, append token ids for its byte-level bytes.

Use `std::priority_queue` or repeated scan. Prefer repeated scan for clarity unless profiling shows
tokenization latency dominates prompt setup.

- [ ] **Step 6: Run golden tokenizer tests**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: all golden text cases pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/text/unicode.cpp src/text/tokenizer.cpp tests/test_qwen_text_tokenizer.cpp
git commit -m "feat: implement qwen byte-level bpe tokenizer"
```

---

### Task 8: Verify Chat Rendering Against HF Golden IDs

**Files:**
- Modify: `tests/test_qwen_chat_template.cpp`
- Modify: `tests/test_qwen_text_tokenizer.cpp`

- [ ] **Step 1: Add message golden test**

Add to `tests/test_qwen_text_tokenizer.cpp`:

```cpp
#include "qus/text/chat_template.h"
```

Add:

```cpp
int test_hf_golden_message_cases() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping golden message cases: local tokenizer not present\n";
        return 0;
    }
    const auto fixture_path = repo_file("tests/fixtures/text/qwen36_text_golden.json");
    std::ifstream in(fixture_path);
    const auto fixture = nlohmann::json::parse(in);
    const qus::text::QwenTokenizer tok(tokenizer_path);
    for (const auto& item : fixture.at("message_cases")) {
        std::vector<qus::text::ChatMessage> messages;
        for (const auto& msg : item.at("messages")) {
            messages.push_back({msg.at("role").get<std::string>(),
                                msg.at("content").get<std::string>()});
        }
        const std::string rendered = qus::text::render_qwen_chat(messages);
        if (rendered != item.at("rendered").get<std::string>()) {
            std::cerr << "rendered prompt mismatch for "
                      << item.at("name").get<std::string>() << '\n';
            return 1;
        }
        const std::vector<int> actual = tok.encode(rendered);
        const std::vector<int> expected = item.at("ids").get<std::vector<int>>();
        if (actual != expected) {
            std::cerr << "message ids mismatch for "
                      << item.at("name").get<std::string>() << '\n';
            return 1;
        }
    }
    return 0;
}
```

Call it from `main()`.

- [ ] **Step 2: Run message golden test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_tokenizer_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_tokenizer_test
```

Expected: message rendering and token ids match HF fixture. If it fails on whitespace, adjust
`render_qwen_chat` to match the fixture exactly.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/test_qwen_text_tokenizer.cpp src/text/chat_template.cpp
git commit -m "test: verify qwen chat rendering golden ids"
```

---

### Task 9: Add TextGenerationRunner

**Files:**
- Create: `include/qus/text/text_runner.h`
- Create: `src/text/text_runner.cpp`
- Create: `tests/test_qwen_text_runner.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add runner header**

Create `include/qus/text/text_runner.h`:

```cpp
#pragma once

#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <string>
#include <vector>

namespace qus::text {

struct TextGenerationOptions {
    int max_new_tokens = 128;
    bool raw_output = false;
    bool print_token_ids = false;
    std::vector<int> stop_token_ids;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;
};

class TextGenerationRunner {
public:
    TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine);

    TextGenerationResult generate(const std::vector<ChatMessage>& messages,
                                  const TextGenerationOptions& options);

private:
    QwenTokenizer& tokenizer_;
    qus::Engine& engine_;
};

std::vector<int> resolve_stop_token_ids(const QwenTokenizer& tokenizer,
                                        const std::vector<int>& overrides);

} // namespace qus::text
```

- [ ] **Step 2: Add stop-id behavior test**

Create `tests/test_qwen_text_runner.cpp`:

```cpp
#include "qus/text/tokenizer.h"
#include "qus/text/text_runner.h"

#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int test_resolve_stop_ids() {
    const auto tokenizer_path =
        std::filesystem::path("/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16");
    if (!std::filesystem::exists(tokenizer_path / "tokenizer.json")) {
        std::cout << "skipping stop id test: local tokenizer not present\n";
        return 0;
    }
    const qus::text::QwenTokenizer tok(tokenizer_path);
    if (qus::text::resolve_stop_token_ids(tok, {}) != std::vector<int>{248046, 248044}) {
        return fail("default stop ids mismatch");
    }
    if (qus::text::resolve_stop_token_ids(tok, {9, 9, 8}) != std::vector<int>{9, 8}) {
        return fail("override stop ids mismatch");
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_resolve_stop_ids();
    return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 3: Register runner test**

Modify `tests/CMakeLists.txt`:

```cmake
qus_add_test(qus_qwen_text_runner_test SOURCES test_qwen_text_runner.cpp)
target_include_directories(qus_qwen_text_runner_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
```

- [ ] **Step 4: Run failing runner test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_runner_test
```

Expected: build fails because runner symbols are undefined.

- [ ] **Step 5: Implement runner**

Create `src/text/text_runner.cpp`:

- `resolve_stop_token_ids` returns tokenizer defaults when overrides are empty;
- duplicate override ids are removed preserving first occurrence;
- negative ids throw `std::invalid_argument`;
- `generate` renders messages, encodes prompt ids, checks `max_new_tokens > 0`, checks prompt length
  against `engine_.max_context()`, calls `engine_.generate`, strips terminal stop token only for clean
  text, and decodes generated ids.

- [ ] **Step 6: Run runner test**

Run:

```bash
cmake --build build -j --target qus_qwen_text_runner_test
ctest --test-dir build --output-on-failure -R qus_qwen_text_runner_test
```

Expected: runner test passes.

- [ ] **Step 7: Commit**

Run:

```bash
git add include/qus/text/text_runner.h src/text/text_runner.cpp tests/test_qwen_text_runner.cpp tests/CMakeLists.txt
git commit -m "feat: add qwen text generation runner"
```

---

### Task 10: Replace Main `qus` CLI With Text Frontend

**Files:**
- Modify: `src/main.cpp`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Replace driver comment**

In `src/CMakeLists.txt`, replace:

```cmake
# --- Driver executable (token-ids in -> token-ids out) -----------------------
```

with:

```cmake
# --- Driver executable (text in -> text out) ---------------------------------
```

- [ ] **Step 2: Replace `src/main.cpp`**

Rewrite `src/main.cpp` to:

- parse CLI through `qus::text::parse_cli`;
- print usage for `--help`;
- construct `EngineOptions` with device, max context, and stop ids;
- load weights;
- construct `QwenTokenizer`;
- build messages from `--prompt` or `--messages`;
- run `TextGenerationRunner`;
- print result text to stdout;
- print generated token ids to stderr only when `--print-token-ids` is set;
- print `generated=<n> elapsed_s=<seconds> tok_s=<rate>` to stderr.

The old positional token-id behavior must be deleted.

- [ ] **Step 3: Build main binary**

Run:

```bash
cmake --build build -j --target qus
```

Expected: `build/src/qus` builds.

- [ ] **Step 4: Check help output**

Run:

```bash
./build/src/qus --help
```

Expected: output mentions `--tokenizer`, `--prompt`, and `--messages`; it does not show positional
token ids.

- [ ] **Step 5: Check error path without weights load**

Run:

```bash
./build/src/qus /tmp/missing.qus --tokenizer /tmp/missing-tokenizer --prompt hi
```

Expected: exits nonzero with an error naming the missing tokenizer path or missing weights path. The
error must not print token-id usage.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/main.cpp src/CMakeLists.txt
git commit -m "feat: make qus cli text-first"
```

---

### Task 11: Update Active Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/design.md`

- [ ] **Step 1: Update README scope**

In `README.md`, replace the v1 scope bullet:

```text
Text-only; token-ids in -> token-ids out (tokenizer runs in Python, outside the engine).
```

with:

```text
Text-only; the primary `qus` binary accepts Qwen3.6 chat text and prints decoded text.
The runtime engine and benchmark/parity tools still use token ids internally.
```

Update the data-flow diagram so it shows:

```text
text/messages -> C++ Qwen tokenizer/chat template -> token ids -> forward -> greedy -> ids -> text
```

- [ ] **Step 2: Add README CLI example**

Add a short usage example:

```bash
./build/src/qus /path/to/weights.qus \
  --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 128
```

Mention that `bench/qus_e2e_bench` still consumes `.ids` fixtures.

- [ ] **Step 3: Update `docs/design.md` scope**

In `docs/design.md`, replace the old runtime I/O decision that says tokenizer/detokenizer are outside
C++ with:

```text
Primary user CLI: text/messages in -> text out through a project-owned Qwen3.6 C++ text frontend.
Core Engine, M2.8 benchmark, and parity tools: token ids in/out for stable performance and debug contracts.
```

Keep M2.8 benchmark sections tokenizer-free.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md docs/design.md
git commit -m "docs: document qwen text frontend cli"
```

---

### Task 12: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Build affected targets**

Run:

```bash
cmake --build build -j --target qus qus_qwen_text_cli_test qus_qwen_chat_template_test qus_qwen_text_tokenizer_test qus_qwen_text_runner_test
```

Expected: all targets build.

- [ ] **Step 2: Run focused CTest**

Run:

```bash
ctest --test-dir build --output-on-failure \
  -R 'qus_qwen_text_(cli|tokenizer|runner)_test|qus_qwen_chat_template_test'
```

Expected: all focused tests pass.

- [ ] **Step 3: Run existing tokenizer/bench Python tests**

Run:

```bash
pytest -q tests/test_bench_tokenizer_tools.py tests/test_bench_report_tools.py
```

Expected: tests pass, proving benchmark/report tokenizer sidecar behavior remains intact.

- [ ] **Step 4: Build e2e benchmark support target**

Run:

```bash
cmake --build build -j --target qus_e2e_bench qus_e2e_bench_support_test
ctest --test-dir build --output-on-failure -R qus_e2e_bench_support_test
```

Expected: build and test pass, proving `.ids` benchmark support still works.

- [ ] **Step 5: Run optional real-weight smoke**

If a local q5090 weight file is available, run:

```bash
./build/src/qus /path/to/weights.qus \
  --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --prompt "用一句话介绍你自己。" \
  --max-context 2048 \
  --max-new 32 \
  --print-token-ids
```

Expected:

- stdout contains decoded text rather than token ids;
- stderr contains generated token ids and timing;
- process stops at `max-new` or Qwen stop token.

If no q5090 weight file is available, record that this smoke was skipped.

- [ ] **Step 6: Scan for stale main CLI wording**

Run:

```bash
rg -n 'token-ids in|token ids in|<token-id>|tokenizer runs in Python|C\\+\\+ tokenizer' README.md docs src/main.cpp src/CMakeLists.txt
```

Expected: no stale claim describes the primary `qus` binary as token-id-only. Matches in benchmark
or historical planning docs do not block completion.

- [ ] **Step 7: Final status**

Run:

```bash
git status --short
```

Expected: clean working tree after all commits.
