from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.codex_gateway.lifecycle import (
    EngineLifecycle,
    LaunchSpec,
    LifecycleError,
    WindowsJobObject,
)


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = 0

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class FakeChildGuard:
    def __init__(self) -> None:
        self.bound: list[FakeProcess] = []
        self.closed = 0

    def bind(self, process: FakeProcess) -> None:
        self.bound.append(process)

    def close(self) -> None:
        self.closed += 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object contract")
def test_windows_kill_on_parent_close_job_can_initialize_and_close() -> None:
    guard = WindowsJobObject()
    guard.close()


def make_spec(tmp_path: Path, **changes: float) -> LaunchSpec:
    executable = tmp_path / "ninfer-serve.exe"
    executable.touch()
    values = {
        "command": (str(executable), "model.ninfer"),
        "cwd": tmp_path,
        "startup_timeout": 5.0,
        "readiness_interval": 0.001,
        "stop_timeout": 0.01,
        "idle_timeout": 10.0,
        "log_dir": tmp_path / "logs",
    }
    values.update(changes)
    return LaunchSpec(**values)  # type: ignore[arg-type]


def test_parallel_acquires_coalesce_to_one_start(tmp_path: Path) -> None:
    ready = threading.Event()
    processes: list[FakeProcess] = []

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(4000 + len(processes))
        processes.append(process)
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=ready.is_set,
        listener_probe=lambda: False,
        start_reaper=False,
    )
    leases = []
    errors = []

    def acquire() -> None:
        try:
            leases.append(lifecycle.acquire())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=acquire) for _ in range(8)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 2
    while not processes and time.monotonic() < deadline:
        time.sleep(0.005)
    ready.set()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert len(processes) == 1
    assert len(leases) == 8
    assert lifecycle.status()["active_leases"] == 8
    for lease in leases:
        lease.close()
    lifecycle.close()


def test_unowned_listener_is_refused_and_never_terminated(tmp_path: Path) -> None:
    launches = 0

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal launches
        launches += 1
        return FakeProcess(1)

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=lambda: True,
        listener_probe=lambda: True,
        start_reaper=False,
    )
    with pytest.raises(LifecycleError, match="unowned process"):
        lifecycle.acquire()
    assert launches == 0
    lifecycle.close()


def test_idle_reaper_waits_for_every_lease_and_stops_only_owned_pid(tmp_path: Path) -> None:
    clock = FakeClock()
    process = FakeProcess(8123)
    lifecycle = EngineLifecycle(
        make_spec(tmp_path, idle_timeout=10.0),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=lambda *args, **kwargs: process,
        readiness_probe=lambda: True,
        listener_probe=lambda: False,
        clock=clock,
        start_reaper=False,
    )
    lease = lifecycle.acquire()
    clock.value += 20
    assert lifecycle.reap_once() is False
    assert process.terminated == 0

    lease.close()
    clock.value += 11
    assert lifecycle.reap_once() is True
    assert process.terminated == 1
    assert process.killed == 0
    assert lifecycle.status()["state"] == "stopped"
    lifecycle.close()


def test_crashed_owned_process_is_replaced_on_next_request(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(9000 + len(processes))
        processes.append(process)
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=lambda: True,
        listener_probe=lambda: False,
        start_reaper=False,
    )
    lifecycle.acquire().close()
    processes[0].returncode = 42
    lifecycle.acquire().close()

    assert [process.pid for process in processes] == [9000, 9001]
    assert lifecycle.status()["owned_pid"] == 9001
    lifecycle.close()


def test_pid_identity_mismatch_is_never_terminated(tmp_path: Path) -> None:
    clock = FakeClock()
    process = FakeProcess(7000)
    lifecycle = EngineLifecycle(
        make_spec(tmp_path, idle_timeout=1.0),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=lambda *args, **kwargs: process,
        readiness_probe=lambda: True,
        listener_probe=lambda: False,
        clock=clock,
        start_reaper=False,
    )
    lifecycle.acquire().close()
    process.pid = 7001
    clock.value += 2

    assert lifecycle.reap_once() is False
    assert process.terminated == 0
    assert process.killed == 0
    lifecycle.close()


def test_startup_failure_is_shared_by_waiters_and_later_request_can_recover(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []
    ready = False
    launch_gate = threading.Event()
    callers_ready = threading.Barrier(5)

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(5000 + len(processes))
        processes.append(process)
        if len(processes) == 1:
            launch_gate.wait(timeout=2)
            process.returncode = 7
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=lambda: ready,
        listener_probe=lambda: False,
        start_reaper=False,
    )
    errors: list[BaseException] = []

    def acquire() -> None:
        try:
            callers_ready.wait(timeout=2)
            lifecycle.acquire()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=acquire) for _ in range(5)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 2
    while not processes and time.monotonic() < deadline:
        time.sleep(0.005)
    launch_gate.set()
    for thread in threads:
        thread.join(timeout=2)

    assert len(processes) == 1
    assert len(errors) == 5
    assert all("exited during startup" in str(error) for error in errors)

    ready = True
    lifecycle.acquire().close()
    assert len(processes) == 2
    lifecycle.close()


def test_child_is_bound_to_parent_guard_before_readiness(tmp_path: Path) -> None:
    process = FakeProcess(6100)
    guard = FakeChildGuard()
    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=lambda *args, **kwargs: process,
        readiness_probe=lambda: guard.bound == [process],
        listener_probe=lambda: False,
        child_guard=guard,
        start_reaper=False,
    )

    lifecycle.acquire().close()
    lifecycle.close()

    assert guard.bound == [process]
    assert guard.closed == 1
    assert process.terminated == 1


def test_invalidation_drains_active_leases_before_stopping_and_recovers(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(6200 + len(processes))
        processes.append(process)
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=lambda: True,
        listener_probe=lambda: False,
        start_reaper=False,
    )
    first = lifecycle.acquire()
    second = lifecycle.acquire()
    first.invalidate("transport wedged")
    first.close()
    assert processes[0].terminated == 0

    with pytest.raises(LifecycleError, match="draining active requests"):
        lifecycle.acquire()
    second.close()
    assert processes[0].terminated == 1

    lifecycle.acquire().close()
    assert [process.pid for process in processes] == [6200, 6201]
    lifecycle.close()


def test_live_but_unready_owned_process_is_replaced(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []
    readiness = iter([True, False, True])

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(6300 + len(processes))
        processes.append(process)
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=lambda: next(readiness),
        listener_probe=lambda: False,
        start_reaper=False,
    )
    lifecycle.acquire().close()
    lifecycle.acquire().close()

    assert [process.pid for process in processes] == [6300, 6301]
    assert processes[0].terminated == 1
    lifecycle.close()


def test_close_waits_for_active_lease_before_terminating_child(tmp_path: Path) -> None:
    process = FakeProcess(6400)
    guard = FakeChildGuard()
    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=lambda *args, **kwargs: process,
        readiness_probe=lambda: True,
        listener_probe=lambda: False,
        child_guard=guard,
        start_reaper=False,
    )
    lease = lifecycle.acquire()
    closed = threading.Event()

    def close() -> None:
        lifecycle.close()
        closed.set()

    thread = threading.Thread(target=close)
    thread.start()
    time.sleep(0.05)
    assert not closed.is_set()
    assert process.terminated == 0

    lease.close()
    thread.join(timeout=2)
    assert closed.is_set()
    assert process.terminated == 1
    assert guard.closed == 1


def test_close_during_startup_stops_bound_child_and_fails_acquire(tmp_path: Path) -> None:
    process = FakeProcess(6500)
    guard = FakeChildGuard()
    ready = threading.Event()
    launched = threading.Event()

    def launch(*args: object, **kwargs: object) -> FakeProcess:
        launched.set()
        return process

    lifecycle = EngineLifecycle(
        make_spec(tmp_path),
        "huihui-qwen3.8-27b-abliterated",
        process_factory=launch,
        readiness_probe=ready.is_set,
        listener_probe=lambda: False,
        child_guard=guard,
        start_reaper=False,
    )
    acquire_errors: list[BaseException] = []
    acquire_thread = threading.Thread(
        target=lambda: _capture_error(lifecycle.acquire, acquire_errors)
    )
    acquire_thread.start()
    assert launched.wait(timeout=2)
    close_thread = threading.Thread(target=lifecycle.close)
    close_thread.start()
    time.sleep(0.05)
    ready.set()
    acquire_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert len(acquire_errors) == 1
    assert "closed during startup" in str(acquire_errors[0])
    assert process.terminated == 1
    assert guard.closed == 1


def _capture_error(callable_value, errors: list[BaseException]) -> None:
    try:
        callable_value()
    except BaseException as error:
        errors.append(error)
