"""Contract tests for the v4 draft-head shortlist selection.

The shortlist -> vocab-id map is a value contract: entry i of the id-map is the
real vocab id of draft row i, so a wrong or nondeterministic shortlist would make
the runtime remap draft argmaxes to the wrong tokens. These tests pin the
frequency ordering, force-include guarantee, exact size, and determinism.

Runs on CPU. Either:
  pytest tools/q5090_convert/tests/test_draft_head.py
  python -m tools.q5090_convert.tests.test_draft_head
"""

from __future__ import annotations

import numpy as np

from .. import draft_head as dh


def _totals(vocab: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1_000_000, size=vocab).astype(np.int64)


def test_shortlist_is_frequency_ranked_and_exact_size():
    vocab, n = 512, 64
    total = _totals(vocab, 1)
    sel = dh.select_shortlist(total, n, force_include=[])
    assert sel.dtype == np.int64
    assert sel.size == n
    assert np.unique(sel).size == n
    # frequency descending, id ascending on ties
    freqs = total[sel]
    assert np.all(freqs[:-1] >= freqs[1:])
    top = np.argsort(-total.astype(np.int64), kind="stable")[:n]
    assert set(sel.tolist()) == set(top.tolist())


def test_shortlist_forces_special_ids_and_reranks():
    vocab, n = 256, 16
    total = _totals(vocab, 2)
    # pick rare ids as force-include: not in the natural top-n
    natural = set(np.argsort(-total.astype(np.int64), kind="stable")[:n].tolist())
    rare = [i for i in range(vocab) if i not in natural][:4]
    sel = dh.select_shortlist(total, n, force_include=rare)
    assert sel.size == n
    assert np.unique(sel).size == n
    for r in rare:
        assert r in sel.tolist()
    freqs = total[sel]
    assert np.all(freqs[:-1] >= freqs[1:])


def test_shortlist_is_deterministic():
    total = _totals(1024, 3)
    a = dh.select_shortlist(total, 100, force_include=[5, 6, 7])
    b = dh.select_shortlist(total, 100, force_include=[7, 6, 5])
    assert np.array_equal(a, b)


def test_shortlist_all_forced_when_force_exceeds_n():
    vocab, n = 128, 8
    total = _totals(vocab, 4)
    forced = list(range(20))  # more than n
    sel = dh.select_shortlist(total, n, force_include=forced)
    assert sel.size == n
    assert set(sel.tolist()).issubset(set(forced))


def test_load_counts_rejects_ragged(tmp_path):
    vocab = 100
    p = tmp_path / "bad.i64"
    np.arange(vocab + 3, dtype="<i8").tofile(p)
    try:
        dh.load_counts(str(p), vocab)
    except SystemExit as exc:
        assert "not a multiple" in str(exc)
    else:
        raise AssertionError("load_counts accepted a ragged histogram")


def test_reserved_model_rows_are_never_selected():
    total = np.zeros(dh.VOCAB_SIZE, dtype=np.int64)
    total[-1] = np.iinfo(np.int64).max
    selected = dh.select_shortlist(total, 8, force_include=[])
    assert np.all(selected < dh.TOKENIZER_ID_COUNT)
    assert dh.VOCAB_SIZE - 1 not in selected


def test_reserved_force_include_is_rejected():
    total = np.zeros(dh.VOCAB_SIZE, dtype=np.int64)
    try:
        dh.select_shortlist(total, 8, force_include=[dh.TOKENIZER_ID_COUNT])
    except SystemExit as exc:
        assert "outside tokenizer domain" in str(exc)
    else:
        raise AssertionError("reserved force-include id was accepted")


def _run_all():
    import tempfile
    from pathlib import Path

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        if fn.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
