from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

from tools.bench.ttft import cases


class Clock:
    now = 1_000_000_000

    def perf_counter_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Gate:
    def __init__(self, clock, trace):
        self.clock = clock
        self.trace = trace
        self.handle = None

    def set(self):
        handle = self.handle
        self.trace.append(("send", handle.role, self.clock.now))
        handle.sent_ns = self.clock.now + handle.lateness_ns


class Handle:
    def __init__(self, role, usage, lateness_ns):
        self.role = role
        self.usage = usage
        self.lateness_ns = lateness_ns
        self.sent_ns = None

    def start(self, gate):
        gate.handle = self

    def as_record(self):
        return {"usage": self.usage}


class Context:
    model = "test-model"

    def __init__(self, clock, trace, usages):
        self.clock = clock
        self.trace = trace
        self.usages = usages
        self.notes = {}
        self.requests = []

    def prepare(self, role, request):
        self.trace.append(("prepare", role, self.clock.now))
        self.requests.append(request)
        # Deliberately finish request A's send after B's planned arrival. B must still be
        # released at its own deadline, without waiting for A to send or produce output.
        return Handle(role, self.usages[len(self.requests) - 1], 20_000_000 if role == "a" else 0)

    def wait_all(self, handles):
        assert all(handle.sent_ns is not None for handle in handles)
        self.trace.append(("wait", None, self.clock.now))

    def require_success(self, handle, *, prerequisite):
        assert prerequisite is False
        self.trace.append(("success", handle.role, self.clock.now))


class Corpus:
    def shape_messages(self, name):
        assert name == "interferer-256"
        return [{"role": "system", "content": "Writer."},
                {"role": "user", "content": "Write a long story."}]


@pytest.mark.parametrize("mode", ["replay", "snapshot"])
def test_fixed_preemption_arrivals_do_not_wait_for_server_progress(monkeypatch, mode):
    clock = Clock()
    trace = []
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "threading", SimpleNamespace(Event=lambda: Gate(clock, trace)))
    context = Context(clock, trace, [
        {"input_tokens": 192, "output_tokens": 256},
        {"input_tokens": 192, "output_tokens": 256},
    ])
    definition = cases.get_case(f"preemption-{mode}")
    definition.run(context, Corpus())

    assert definition.profile == f"preemption-{mode}"
    assert [(event, role) for event, role, _ in trace[:4]] == [
        ("prepare", "a"), ("prepare", "b"), ("send", "a"), ("send", "b"),
    ]
    arrivals = context.notes["arrivals"]
    assert [entry["offset_ns"] for entry in arrivals] == [0, 10_000_000]
    assert arrivals[1]["released_ns"] - arrivals[0]["released_ns"] == 10_000_000
    assert arrivals[0]["sent_ns"] > arrivals[1]["sent_ns"]
    assert [entry["lateness_ns"] for entry in arrivals] == [20_000_000, 0]
    assert context.notes["arrival_mode"] == "fixed_schedule"
    assert context.notes["mechanism_requirements"] == ["preemption", f"{mode}_restore"]
    assert context.requests[0].payload["messages"] != context.requests[1].payload["messages"]
    for request in context.requests:
        assert request.payload["max_completion_tokens"] == 256
        assert "ignore_eos" not in request.payload
        assert "min_tokens" not in request.payload


def test_short_generation_and_missing_usage_remain_observations(monkeypatch):
    clock = Clock()
    trace = []
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "threading", SimpleNamespace(Event=lambda: Gate(clock, trace)))
    context = Context(clock, trace, [{"input_tokens": 188, "output_tokens": 3}, {}])
    cases.get_case("preemption-replay").run(context, Corpus())

    assert context.notes["observed_workload"] == [
        {"role": "a", "input_tokens": 188, "output_tokens": 3,
         "nominal_input_matched": False, "output_limit_reached": False},
        {"role": "b", "input_tokens": None, "output_tokens": None,
         "nominal_input_matched": None, "output_limit_reached": None},
    ]
    assert [role for event, role, _ in trace if event == "success"] == ["a", "b"]


@pytest.mark.parametrize("run", [cases._anonymous_hot, cases._session_hot])
def test_generated_continuations_are_not_labelled_as_matching_fixed_work(run):
    requests = []

    def start(role, request):
        requests.append(request)
        return SimpleNamespace(role=role, output_text="generated answer", response_id="resp_seed")

    context = SimpleNamespace(
        model="test-model", notes={"input_dependency": "fixed"}, start=start,
        require_success=lambda handle: None,
    )
    corpus = SimpleNamespace(
        shape_messages=lambda name: [{"role": "user", "content": "fixed seed input"}],
        shape=lambda name: {"max_output_tokens": 16},
    )
    run(context, corpus)

    assert len(requests) == 2
    assert context.notes["input_dependency"] == "generated"
    assert context.notes["throughput_comparable"] is False
    assert "generated assistant output" in context.notes["throughput_limitation"]
    continuation = requests[-1].payload
    if run is cases._anonymous_hot:
        assert {"role": "assistant", "content": "generated answer"} in continuation["messages"]
    else:
        assert continuation["previous_response_id"] == "resp_seed"


def test_live_log_waits_for_complete_lines_and_ignores_warmup(tmp_path):
    path = tmp_path / "request.jsonl"
    path.write_text(json.dumps({"event": "throughput", "scheduler": {"paused": 1}}) + "\n")
    reader = cases._LiveRequestLog(SimpleNamespace(notes={"request_log_jsonl": str(path)}))
    assert reader.poll() == []
    first = json.dumps({"event": "throughput", "scheduler": {"paused": 0}}) + "\n"
    second = json.dumps({"event": "throughput", "scheduler": {"paused": 1}}) + "\n"
    with path.open("a") as output:
        output.write(first + second[:12])
    assert [item["event"]["scheduler"]["paused"] for item in reader.poll()] == [0]
    with path.open("a") as output:
        output.write(second[12:])
    assert [item["event"]["scheduler"]["paused"] for item in reader.poll()] == [1]
    assert reader.poll() == []
    assert reader.samples == 2


class MechanismHandle:
    def __init__(self, role):
        self.role = role
        self.is_done = False
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        return 1000

    def outcome(self):
        return "cancelled" if self.cancelled else "success"


def mechanism_context():
    started = []
    def start(role, request):
        started.append((role, request))
        return MechanismHandle(role)
    def success(handle, *, prerequisite=True):
        assert handle.outcome() == "success"
    def require(condition, *args, **kwargs):
        assert condition, args
    return SimpleNamespace(model="test-model", notes={}, start=start,
                           require_success=success, require=require, started=started)


@pytest.mark.parametrize(("schedulers", "expected"), [
    ([{"replaying": 1, "decode_ready": 0, "prefilling": 0},
      {"replaying": 0, "decode_ready": 1, "prefilling": 0}], "not_observed"),
    ([{"replaying": 1, "decode_ready": 1, "prefilling": 0}], "observed"),
    ([], "unavailable"),
])
def test_replay_overlap_requires_one_simultaneous_sample(monkeypatch, schedulers, expected):
    context = mechanism_context()
    corpus = SimpleNamespace(shape_messages=lambda name: [
        {"role": "system", "content": "fixed reference"},
        {"role": "user", "content": "fixed question"},
    ])
    def fixed(ctx, requests, offsets, facts, on_sample):
        assert offsets == (0, 10_000_000, 20_000_000, 50_000_000, 100_000_000)
        assert len(requests) == 5
        for scheduler in schedulers:
            on_sample({"event": {
                "scheduler": scheduler, "scheduling": {"replayed_tokens": 128},
                "decode_batch": {"rounds": 1}, "host_work": {"units": {"prefill": 0}},
            }, "observed_ns": 500}, [])
        return []
    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("shared-growth-fairness").run(context, corpus)
    assert context.notes["mechanism_observations"]["sampled_replay_copresence_and_progress"] == expected


@pytest.mark.parametrize(("replayed", "rounds", "expected"), [
    (0, 1, "not_observed"), (128, 0, "not_observed"), (None, 1, "unavailable"),
])
def test_phase_copresence_alone_does_not_prove_actual_work(monkeypatch, replayed, rounds, expected):
    context = mechanism_context()
    corpus = SimpleNamespace(shape_messages=lambda name: [
        {"role": "system", "content": "fixed reference"},
        {"role": "user", "content": "fixed question"},
    ])
    def fixed(ctx, requests, offsets, facts, on_sample):
        on_sample({"event": {
            "scheduler": {"replaying": 1, "decode_ready": 1, "prefilling": 0},
            "scheduling": {"replayed_tokens": replayed},
            "decode_batch": {"rounds": rounds}, "host_work": {"units": {"prefill": 0}},
        }, "observed_ns": 500}, [])
        return []
    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("shared-growth-fairness").run(context, corpus)
    assert context.notes["mechanism_observations"]["sampled_replay_copresence_and_progress"] == expected


@pytest.mark.parametrize(("paused", "already_done", "expected"), [
    (0, False, "not_observed"), (1, False, "observed"), (1, True, "not_observed"),
])
def test_pressure_cancel_requires_observed_pause_and_a_live_target(
    monkeypatch, paused, already_done, expected,
):
    context = mechanism_context()
    handles = [MechanismHandle("a"), MechanismHandle("b")]
    handles[0].is_done = already_done
    def fixed(ctx, requests, offsets, facts, on_sample):
        on_sample({"event": {"scheduler": {"paused": paused}}, "observed_ns": 500}, handles)
        return handles
    monkeypatch.setattr(cases, "_fixed_request_graph", fixed)
    cases.get_case("snapshot-history-cancel").run(context, Corpus())
    assert context.notes["mechanism_observations"]["pressure_cancellation"] == expected
    assert handles[0].cancelled == (expected == "observed")
    assert [role for role, _ in context.started] == ["history-seed", "history-warm", "history-probe"]
    if expected == "observed":
        assert context.notes["pressure_cancel"]["target_state"] == "unavailable"
        assert context.notes["pressure_cancel"]["sample"]["event"]["scheduler"]["paused"] == 1
