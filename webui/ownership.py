"""Process ownership shared by the dashboard and command-line launcher.

Records include process creation times so stale state cannot claim recycled PIDs.
Only descendants of a still-identifiable process we spawned can be claimed.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import threading

STATE_PATH = Path(__file__).resolve().parent / '.state' / 'launcher.json'
LOCK = threading.RLock()


def identity(pid):
    """Return a stable birth identifier, or None if it cannot be verified."""
    try:
        if os.name == 'nt':
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                    return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
            finally:
                kernel.CloseHandle(handle)
        else:
            # /proc's starttime is independent of the process's display name.
            stat = Path('/proc/%s/stat' % int(pid)).read_text()
            return stat.rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        pass
    return None


def parents():
    if os.name != 'nt':
        result = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                result[int(path.parent.name)] = int(path.read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
        return result
    class Entry(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD),
                    ('pid', wintypes.DWORD), ('heap', ctypes.c_size_t),
                    ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
                    ('parent', wintypes.DWORD), ('priority', wintypes.LONG),
                    ('flags', wintypes.DWORD), ('exe', wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return {}
    result = {}
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            result[entry.pid] = entry.parent
            ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return result


def load():
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix('.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    os.replace(temporary, STATE_PATH)


def record(kind, pid):
    birth = identity(pid)
    if birth is None:
        raise RuntimeError('Cannot verify spawned process identity')
    with LOCK:
        state = load()
        state[kind] = {'owned': {str(pid): birth}}
        save(state)


def claim(kind, candidates):
    with LOCK:
        state = load()
        owned = state.get(kind, {}).get('owned', {})
        if not isinstance(owned, dict):
            return []  # old PID-only state is never trusted
        valid = {p: birth for p, birth in owned.items() if identity(p) == birth}
        tree = parents()
        for candidate in set(map(int, candidates)) | set(tree):
            pid = int(candidate)
            birth = identity(pid)
            ancestor, seen = pid, set()
            while ancestor and ancestor not in seen:
                seen.add(ancestor)
                if str(ancestor) in valid:
                    # A parent PID can itself be recycled after an orphan exits.
                    if birth and int(birth) >= int(valid[str(ancestor)]):
                        valid[str(pid)] = birth
                    break
                parent = tree.get(ancestor, 0)
                parent_birth = identity(parent) if parent else None
                child_birth = identity(ancestor)
                if not parent_birth or not child_birth or int(parent_birth) > int(child_birth):
                    break
                ancestor = parent
        if state.get(kind) != {'owned': valid}:
            state[kind] = {'owned': valid}
            save(state)
        return list(valid)


def terminate(pid, birth):
    if os.name != 'nt':
        if identity(pid) != birth:
            return False
        return subprocess.run(['kill', '-TERM', pid], capture_output=True, timeout=20).returncode == 0
    # Verify and terminate through one handle: a PID cannot be recycled between
    # the identity check and taskkill selecting its target.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1001, False, int(pid))
    if not handle:
        return False
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return False
        actual = str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        return actual == birth and bool(kernel.TerminateProcess(handle, 0))
    finally:
        kernel.CloseHandle(handle)


def stop(kind, candidates=()):
    with LOCK:
        owned = claim(kind, candidates)
        state = load()
        records = state.get(kind, {}).get('owned', {})
        killed = []
        # Kill individually, never taskkill /T on a stale or changing tree.
        for pid in reversed(owned):
            if identity(pid) != records.get(pid):
                continue
            if terminate(pid, records[pid]):
                killed.append(pid)
        state[kind] = {'owned': {p: b for p, b in records.items() if p not in killed and identity(p) == b}}
        save(state)
        skipped = [str(p) for p in candidates if str(p) not in killed]
        return {'ok': not skipped, 'killed': killed, 'skipped': skipped}
