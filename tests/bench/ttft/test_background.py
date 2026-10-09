from __future__ import annotations

from collections import deque
import threading
import time

import pytest

from tools.bench.ttft.cases import _BackgroundResponseLoops
from tools.bench.ttft.execution import CaseContext, CaseExecutionError
from tools.ninfer_serve.client import ProtocolEvent, ProtocolRequest, ServeExchangeResult
from tools.streaming_http.client import HttpExchangeResult


class ControlledExchange:
    """A stream whose completion and cancellation are controlled by the test's threads."""

    body_bytes = 0
    body = b"{}"

    def __init__(self, *, first_output=True, failure=None):
        self.request = ProtocolRequest("openai_responses", "/test", {})
        self.first_output = first_output
        self.failure = failure
        self.cancel_ns = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.allow_return = threading.Event()
        self.allow_return.set()
        self.completed_normally = False

    def cancel(self):
        self.cancel_ns = time.perf_counter_ns()
        self.finished.set()
        return self.cancel_ns

    def complete(self):
        self.completed_normally = True
        self.finished.set()

    def execute(self, *, on_sent, on_body_sent, on_event):
        sent_ns = time.perf_counter_ns()
        on_sent(sent_ns)
        on_body_sent(sent_ns)
        events = []
        if self.first_output:
            events.append(ProtocolEvent("model_output", "delta", sent_ns, output="x"))
            on_event(events[-1])
        self.started.set()
        assert self.finished.wait(5), "test did not terminate its stream"
        assert self.allow_return.wait(5), "test did not release its completed stream"
        ended_ns = time.perf_counter_ns()
        if self.completed_normally:
            events.append(ProtocolEvent("terminal", "done", ended_ns))
            on_event(events[-1])
        if self.failure == "stream":
            events.append(ProtocolEvent("error", "error", ended_ns))
            on_event(events[-1])
        cancelled = self.cancel_ns is not None and not self.completed_normally
        return ServeExchangeResult(
            protocol=self.request.protocol,
            body_bytes=0,
            http=HttpExchangeResult(
                sent_ns=sent_ns, body_sent_ns=sent_ns, ended_ns=ended_ns,
                status=503 if self.failure == "http" else 200,
                error="socket timeout" if self.failure == "transport" else None,
                cancelled=cancelled, cancel_requested=self.cancel_ns is not None,
                cancel_ns=self.cancel_ns,
            ),
            events=events,
            protocol_error="invalid SSE" if self.failure == "protocol" else None,
        )


class ControlledClient:
    def __init__(self, exchanges):
        self.exchanges = deque(exchanges)
        self.prepare_entered = threading.Event()
        self.allow_prepare = threading.Event()
        self.allow_prepare.set()

    def prepare(self, request):
        self.prepare_entered.set()
        assert self.allow_prepare.wait(5), "test did not release request preparation"
        return self.exchanges.popleft()


def start_loop(exchanges, *, timeout=2, on_progress=None):
    client = ControlledClient(exchanges)
    context = CaseContext(client, "test-model", timeout, on_progress)
    initial = context.start("background-0-generation-0", exchanges[0].request)
    assert exchanges[0].started.wait(2)
    return context, client, _BackgroundResponseLoops(context, [initial])


@pytest.mark.parametrize("first_output", [False, True])
def test_stop_keeps_cancelled_replacement_in_raw_results(first_output):
    initial = ControlledExchange()
    replacement = ControlledExchange(first_output=first_output)
    context, _, loops = start_loop([initial, replacement])
    try:
        initial.complete()
        assert replacement.started.wait(2)
    finally:
        loops.stop()

    records = context.records()
    assert [record["outcome"] for record in records] == ["success", "cancelled"]
    assert (records[1]["first_output_ns"] is not None) == first_output
    assert records[1]["cancel_requested"] and records[1]["transport_cancelled"]
    assert context.notes["background_stream_generations"] == [1]


def test_stop_serializes_with_replacement_registration():
    initial = ControlledExchange()
    replacement = ControlledExchange(first_output=False)
    context, client, loops = start_loop([initial, replacement])
    client.prepare_entered.clear()
    client.allow_prepare.clear()
    initial.complete()
    assert client.prepare_entered.wait(2)
    errors = []
    stopping = threading.Event()

    def stop():
        stopping.set()
        try:
            loops.stop()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert stopping.wait(2)
    finally:
        client.allow_prepare.set()
        thread.join(3)

    assert not thread.is_alive()
    assert not errors
    assert [record["outcome"] for record in context.records()] == ["success", "cancelled"]


@pytest.mark.parametrize("failure", ["http", "transport", "protocol", "stream"])
def test_stop_does_not_hide_request_failure(failure):
    exchange = ControlledExchange(failure=failure)
    context, _, loops = start_loop([exchange])
    with pytest.raises(CaseExecutionError, match="ended before its replacement"):
        loops.stop()
    record = context.records()[0]
    assert record["http_status"] == (503 if failure == "http" else 200)
    if failure == "transport":
        assert record["transport_error"] == "socket timeout"
    elif failure == "protocol":
        assert record["protocol_error"] == "invalid SSE"
    elif failure == "stream":
        assert any(event["kind"] == "error" for event in record["events"])


def test_stop_does_not_hide_worker_wait_timeout():
    timed_out = threading.Event()

    def progress(stage, timestamp, fields):
        if stage == "request.wait_failed" and fields.get("target") == "worker_done":
            timed_out.set()

    exchange = ControlledExchange()
    context, _, loops = start_loop([exchange], timeout=0.05, on_progress=progress)
    try:
        assert timed_out.wait(2)
    finally:
        with pytest.raises(CaseExecutionError, match="did not terminate"):
            loops.stop()
        context.wait_all()
    assert context.records()[0]["outcome"] == "cancelled"


def test_cancellation_outside_loop_shutdown_remains_a_failure(monkeypatch):
    exchange = ControlledExchange()
    context, _, loops = start_loop([exchange])
    exchange.allow_return.clear()
    handle = context.handles[0]
    handle.cancel()
    original_cancel = handle.cancel
    stopping = threading.Event()
    errors = []

    def cancel(**kwargs):
        result = original_cancel(**kwargs)
        stopping.set()
        return result

    def stop():
        try:
            loops.stop()
        except CaseExecutionError as error:
            errors.append(str(error))

    monkeypatch.setattr(handle, "cancel", cancel)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
        assert stopping.wait(2)
    finally:
        exchange.allow_return.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and "ended before its replacement" in errors[0]
    assert context.records()[0]["outcome"] == "cancelled"
