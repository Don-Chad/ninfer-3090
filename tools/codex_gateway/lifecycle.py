"""Owned-process lifecycle for the candidate Huihui NInfer server.

The lifecycle never adopts or terminates an arbitrary listener.  Shutdown
authority is represented by the exact ``Popen`` handle returned by the launch
operation, plus its recorded PID and a monotonically increasing generation.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class ProcessHandle(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


class ChildGuard(Protocol):
    """Binds launched children to the gateway parent's lifetime."""

    def bind(self, process: ProcessHandle) -> None: ...
    def close(self) -> None: ...


class NoopChildGuard:
    """Non-Windows guard; normal parent/child process semantics apply."""

    def bind(self, process: ProcessHandle) -> None:
        del process

    def close(self) -> None:
        pass


class WindowsJobObject:
    """Windows Job Object with KILL_ON_JOB_CLOSE parent-death semantics."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._handle = handle

    def bind(self, process: ProcessHandle) -> None:
        if self._handle is None:
            raise LifecycleError("Windows child job is already closed")
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise LifecycleError("launcher did not expose a Windows process handle")
        if not self._kernel32.AssignProcessToJobObject(self._handle, int(process_handle)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def create_child_guard() -> ChildGuard:
    return WindowsJobObject() if sys.platform == "win32" else NoopChildGuard()


class LifecycleError(RuntimeError):
    """The managed engine could not safely reach the requested state."""


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    cwd: Path
    required_files: tuple[Path, ...] = ()
    host: str = "127.0.0.1"
    port: int = 8080
    startup_timeout: float = 180.0
    readiness_interval: float = 0.25
    stop_timeout: float = 20.0
    idle_timeout: float = 600.0
    log_dir: Path = Path("logs")

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("engine command must not be empty")
        if self.host != "127.0.0.1":
            raise ValueError("the candidate engine must bind to 127.0.0.1")
        if not 1 <= self.port <= 65535:
            raise ValueError("engine port is out of range")
        for name, value in (
            ("startup_timeout", self.startup_timeout),
            ("readiness_interval", self.readiness_interval),
            ("stop_timeout", self.stop_timeout),
            ("idle_timeout", self.idle_timeout),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass
class _OwnedProcess:
    process: ProcessHandle
    pid: int
    generation: int
    stdout_handle: object | None = None
    stderr_handle: object | None = None

    def close_logs(self) -> None:
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle is not None:
                try:
                    handle.close()  # type: ignore[attr-defined]
                except OSError:
                    pass
        self.stdout_handle = None
        self.stderr_handle = None


class EngineLease:
    """Keeps the owned engine resident for one complete proxied request."""

    def __init__(self, owner: "EngineLifecycle", generation: int) -> None:
        self._owner = owner
        self._generation = generation
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._owner._release(self._generation)

    def invalidate(self, reason: str) -> None:
        self._owner.invalidate(self._generation, reason)

    def __enter__(self) -> "EngineLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def port_is_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ninfer_is_ready(host: str, port: int, model_slug: str, timeout: float = 0.5) -> bool:
    """Require both HTTP health and the exact loaded model identity."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        health = connection.getresponse()
        health.read()
        if health.status != 200:
            return False
    except OSError:
        return False
    finally:
        connection.close()

    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            return False
        payload = json.loads(raw)
        models = payload.get("data", payload.get("models", []))
        return isinstance(models, list) and any(
            isinstance(item, dict)
            and (item.get("id") == model_slug or item.get("slug") == model_slug)
            for item in models
        )
    except (OSError, ValueError, TypeError):
        return False
    finally:
        connection.close()


class EngineLifecycle:
    """Coalesced lazy start, request leases, idle stop, and crash recovery."""

    def __init__(
        self,
        spec: LaunchSpec,
        model_slug: str,
        *,
        process_factory: Callable[..., ProcessHandle] = subprocess.Popen,
        readiness_probe: Callable[[], bool] | None = None,
        listener_probe: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        child_guard: ChildGuard | None = None,
        start_reaper: bool = True,
    ) -> None:
        self.spec = spec
        self.model_slug = model_slug
        self._process_factory = process_factory
        self._readiness_probe = readiness_probe or (
            lambda: ninfer_is_ready(spec.host, spec.port, model_slug)
        )
        self._listener_probe = listener_probe or (
            lambda: port_is_open(spec.host, spec.port)
        )
        self._clock = clock
        self._sleep = sleep
        self._child_guard = child_guard or (
            create_child_guard() if process_factory is subprocess.Popen else NoopChildGuard()
        )
        self._condition = threading.Condition(threading.RLock())
        self._owned: _OwnedProcess | None = None
        self._starting = False
        self._stopping = False
        self._generation = 0
        self._start_failure: tuple[int, str] | None = None
        self._invalid_generations: set[int] = set()
        self._leases: dict[int, int] = {}
        self._last_activity = clock()
        self._closed = False
        self._reaper_stop = threading.Event()
        self._reaper: threading.Thread | None = None
        if start_reaper and spec.idle_timeout > 0:
            self._reaper = threading.Thread(
                target=self._reaper_loop,
                name="huihui-engine-idle-reaper",
                daemon=True,
            )
            self._reaper.start()

    def acquire(self) -> EngineLease:
        """Return a lease after exactly one coalesced start reaches readiness."""
        while True:
            with self._condition:
                if self._closed:
                    raise LifecycleError("engine lifecycle is closed")
                while self._starting:
                    waited_generation = self._generation
                    self._condition.wait()
                    if self._closed:
                        raise LifecycleError("engine lifecycle is closed")
                    if (
                        self._start_failure is not None
                        and self._start_failure[0] == waited_generation
                    ):
                        raise LifecycleError(self._start_failure[1])
                while self._stopping:
                    self._condition.wait()
                    if self._closed:
                        raise LifecycleError("engine lifecycle is closed")

                owned = self._owned
                if owned is not None and self._is_same_live_process(owned):
                    if owned.generation in self._invalid_generations:
                        if self._leases.get(owned.generation, 0):
                            raise LifecycleError(
                                "owned Huihui engine is unhealthy and draining active requests"
                            )
                        self._terminate_owned(owned)
                        self._owned = None
                        self._invalid_generations.discard(owned.generation)
                        continue
                    if not self._readiness_probe() or not self._is_same_live_process(owned):
                        self._invalid_generations.add(owned.generation)
                        if self._leases.get(owned.generation, 0):
                            raise LifecycleError(
                                "owned Huihui engine failed readiness and is draining active requests"
                            )
                        self._terminate_owned(owned)
                        self._owned = None
                        self._invalid_generations.discard(owned.generation)
                        continue
                    self._leases[owned.generation] = self._leases.get(owned.generation, 0) + 1
                    self._last_activity = self._clock()
                    return EngineLease(self, owned.generation)

                if owned is not None:
                    logging.warning(
                        "owned Huihui engine pid=%s exited with code %s; recovering on demand",
                        owned.pid,
                        owned.process.poll(),
                    )
                    owned.close_logs()
                    self._owned = None

                # A listener with no matching owned process is not authority to
                # adopt it, proxy to it, or later terminate it.
                if self._listener_probe():
                    raise LifecycleError(
                        f"port {self.spec.port} is occupied by an unowned process"
                    )
                self._starting = True
                self._generation += 1
                generation = self._generation

            try:
                started = self._launch_and_wait(generation)
            except BaseException as error:
                with self._condition:
                    self._start_failure = (generation, str(error))
                    self._starting = False
                    self._condition.notify_all()
                raise

            with self._condition:
                if self._closed:
                    self._starting = False
                    self._condition.notify_all()
                    self._terminate_owned(started)
                    raise LifecycleError("engine lifecycle closed during startup")
                self._owned = started
                self._start_failure = None
                self._starting = False
                self._leases[generation] = self._leases.get(generation, 0) + 1
                self._last_activity = self._clock()
                self._condition.notify_all()
                return EngineLease(self, generation)

    def _launch_and_wait(self, generation: int) -> _OwnedProcess:
        executable = Path(self.spec.command[0])
        if not executable.is_file():
            raise LifecycleError(f"engine executable not found: {executable}")
        if not self.spec.cwd.is_dir():
            raise LifecycleError(f"engine working directory not found: {self.spec.cwd}")
        for required in self.spec.required_files:
            if not required.is_file():
                raise LifecycleError(f"required engine file not found: {required}")
        self.spec.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = open(self.spec.log_dir / "huihui-engine.out.log", "ab", buffering=0)
        stderr_handle = open(self.spec.log_dir / "huihui-engine.err.log", "ab", buffering=0)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = self._process_factory(
                list(self.spec.command),
                cwd=str(self.spec.cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        owned = _OwnedProcess(process, int(process.pid), generation, stdout_handle, stderr_handle)
        if owned.pid <= 0:
            self._terminate_owned(owned)
            raise LifecycleError("engine launcher returned an invalid PID")
        try:
            self._child_guard.bind(process)
        except BaseException:
            self._terminate_owned(owned)
            raise

        logging.info("started owned Huihui engine pid=%s generation=%s", owned.pid, generation)
        deadline = self._clock() + self.spec.startup_timeout
        try:
            while self._clock() < deadline:
                if not self._is_same_live_process(owned):
                    code = process.poll()
                    owned.close_logs()
                    raise LifecycleError(f"Huihui engine exited during startup with code {code}")
                if self._readiness_probe():
                    return owned
                self._sleep(self.spec.readiness_interval)

            raise LifecycleError(
                f"Huihui engine did not become ready within {self.spec.startup_timeout:g}s"
            )
        except BaseException:
            self._terminate_owned(owned)
            raise

    @staticmethod
    def _is_same_live_process(owned: _OwnedProcess) -> bool:
        return owned.process.pid == owned.pid and owned.process.poll() is None

    def _release(self, generation: int) -> None:
        stop: _OwnedProcess | None = None
        with self._condition:
            current = self._leases.get(generation, 0)
            if current > 1:
                self._leases[generation] = current - 1
            else:
                self._leases.pop(generation, None)
            self._last_activity = self._clock()
            if (
                generation in self._invalid_generations
                and generation not in self._leases
                and self._owned is not None
                and self._owned.generation == generation
            ):
                stop = self._owned
                self._owned = None
                self._stopping = True
            self._condition.notify_all()
        if stop is not None:
            try:
                self._terminate_owned(stop)
            finally:
                with self._condition:
                    self._invalid_generations.discard(generation)
                    self._stopping = False
                    self._condition.notify_all()

    def invalidate(self, generation: int, reason: str) -> None:
        """Mark an owned generation unhealthy without killing active leases."""
        stop: _OwnedProcess | None = None
        with self._condition:
            owned = self._owned
            if owned is None or owned.generation != generation:
                return
            logging.warning("invalidating Huihui engine generation=%s: %s", generation, reason)
            self._invalid_generations.add(generation)
            if not self._leases.get(generation, 0):
                stop = owned
                self._owned = None
                self._stopping = True
        if stop is not None:
            try:
                self._terminate_owned(stop)
            finally:
                with self._condition:
                    self._invalid_generations.discard(generation)
                    self._stopping = False
                    self._condition.notify_all()

    def reap_once(self, now: float | None = None) -> bool:
        """Stop one owned, idle engine. Return whether a stop was attempted."""
        with self._condition:
            if self._starting or self._stopping or self._owned is None:
                return False
            owned = self._owned
            if not self._is_same_live_process(owned):
                owned.close_logs()
                self._owned = None
                self._condition.notify_all()
                return False
            if owned.generation in self._invalid_generations:
                return False
            if self._leases.get(owned.generation, 0):
                return False
            instant = self._clock() if now is None else now
            if instant - self._last_activity < self.spec.idle_timeout:
                return False
            self._stopping = True
            self._owned = None

        try:
            self._terminate_owned(owned)
            logging.info("stopped idle owned Huihui engine pid=%s", owned.pid)
        finally:
            with self._condition:
                self._stopping = False
                self._last_activity = self._clock()
                self._condition.notify_all()
        return True

    def _terminate_owned(self, owned: _OwnedProcess) -> None:
        # Both handle identity and the recorded PID must still match. There is
        # deliberately no port-owner lookup and no Stop-Process/fuser fallback.
        if self._is_same_live_process(owned):
            owned.process.terminate()
            try:
                owned.process.wait(timeout=self.spec.stop_timeout)
            except subprocess.TimeoutExpired:
                if self._is_same_live_process(owned):
                    owned.process.kill()
                    owned.process.wait(timeout=self.spec.stop_timeout)
        owned.close_logs()

    def _reaper_loop(self) -> None:
        interval = min(10.0, max(0.1, self.spec.idle_timeout / 4.0))
        while not self._reaper_stop.wait(interval):
            try:
                self.reap_once()
            except Exception:
                logging.exception("Huihui idle reaper failed")

    def status(self) -> dict[str, object]:
        with self._condition:
            owned = self._owned
            live = owned is not None and self._is_same_live_process(owned)
            return {
                "state": (
                    "closed" if self._closed else
                    "starting" if self._starting else
                    "stopping" if self._stopping else
                    "running" if live else
                    "stopped"
                ),
                "owned_pid": owned.pid if live else None,
                "active_leases": sum(self._leases.values()),
                "model": self.model_slug,
            }

    def close(self) -> None:
        self._reaper_stop.set()
        with self._condition:
            self._closed = True
            while self._starting or self._stopping or self._leases:
                self._condition.wait()
            owned = self._owned
            self._owned = None
            self._condition.notify_all()
        if owned is not None:
            self._terminate_owned(owned)
        self._child_guard.close()
        if self._reaper is not None and self._reaper is not threading.current_thread():
            self._reaper.join(timeout=2.0)
