# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host memory, without a dependency on psutil.

torchtitan's ``DeviceMemoryMonitor`` asks for device allocator stats that do not
exist on ``torch.cpu``, so ``cpu_compat`` substitutes a host-memory monitor. That
stand-in used psutil, which is not installed in this venv -- so every run logged
``memory: -1.00GiB`` and the one number that says whether a size *fits* was
missing.

The replacement uses whatever is available, in order:

1. ``psutil`` when importable;
2. ``ctypes`` on Windows (``GetProcessMemoryInfo`` + ``GlobalMemoryStatusEx``);
3. ``/proc`` on Linux.

Two figures are reported, and they answer different questions:

- **working set / RSS** -- pages actually resident. This is what competes for
  physical RAM.
- **commit / pagefile usage** -- pages the process has *reserved*. This is what
  the OS refuses to hand out when it runs out (Windows counts commit against
  RAM + pagefile, not against RAM alone), so it is the figure that decides
  whether a bigger model can start at all.

On Windows, ``GetCurrentProcess`` must have its ``restype`` set to ``HANDLE``:
ctypes defaults to ``c_int``, which truncates the pseudo-handle to 32 bits and
makes ``GetProcessMemoryInfo`` fail silently with a zeroed struct.
"""

from __future__ import annotations

import ctypes
import os

_IMPL = None


def _init_windows() -> bool:
    global _k32, _psapi, _PMC
    try:
        from ctypes import wintypes

        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _psapi = ctypes.WinDLL("psapi", use_last_error=True)
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        # OpenProcess's handle must also be declared, for the same reason: with
        # the default c_int return type the handle is truncated to 32 bits, and a
        # truncated handle makes GetProcessMemoryInfo report a *different*
        # process's (tiny, wrong) numbers instead of failing. That looked exactly
        # like an 8-rank run using 5 MiB per rank.
        _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _k32.OpenProcess.restype = wintypes.HANDLE
        _k32.CloseHandle.argtypes = [wintypes.HANDLE]
        _k32.CloseHandle.restype = wintypes.BOOL

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        _psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PMC),
            wintypes.DWORD,
        ]
        _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        _PMC = _PMC

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
        _k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        globals()["_MEMORYSTATUSEX"] = _MEMORYSTATUSEX
        return True
    except Exception:  # pragma: no cover - non-Windows
        return False


def _init() -> str:
    global _IMPL
    if _IMPL is not None:
        return _IMPL
    try:
        import psutil  # type: ignore  # noqa: F401

        _IMPL = "psutil"
    except ImportError:
        if os.name == "nt" and _init_windows():
            _IMPL = "windows"
        elif os.path.exists("/proc/self/statm"):
            _IMPL = "proc"
        else:
            _IMPL = "none"
    return _IMPL


def available() -> bool:
    return _init() != "none"


def implementation() -> str:
    return _init()


def process_memory(pid: int | None = None) -> dict:
    """``{rss, peak_rss, commit, peak_commit}`` in bytes (-1 if unknown).

    ``pid=None`` means this process. A pid is useful for watching *other*
    processes -- the launcher samples its ranks this way, because "how much
    memory did rank 3 need" is a property of the rank, not of the launcher.
    """
    impl = _init()
    if impl == "psutil":
        import psutil

        proc = psutil.Process() if pid is None else psutil.Process(pid)
        info = proc.memory_info()
        return {
            "rss": int(info.rss),
            "peak_rss": int(getattr(info, "peak_wset", info.rss)),
            "commit": int(getattr(info, "private", info.rss)),
            "peak_commit": int(getattr(info, "peak_pagefile", info.rss)),
            "source": "psutil",
        }
    if impl == "windows":
        counter = _PMC()
        counter.cb = ctypes.sizeof(_PMC)
        if pid is None:
            handle = _k32.GetCurrentProcess()
            owned = False
        else:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            owned = True
        if not handle:
            return _unknown(f"windows(no handle for pid {pid})")
        try:
            ok = _psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counter), counter.cb
            )
        finally:
            if owned:
                _k32.CloseHandle(handle)
        if not ok:
            return _unknown("windows(query failed)")
        return {
            "rss": int(counter.WorkingSetSize),
            "peak_rss": int(counter.PeakWorkingSetSize),
            "commit": int(counter.PagefileUsage),
            "peak_commit": int(counter.PeakPagefileUsage),
            "source": "windows",
        }
    if impl == "proc":
        base = "/proc/self" if pid is None else f"/proc/{pid}"
        try:
            with open(f"{base}/statm", "r", encoding="ascii") as fh:
                fields = fh.read().split()
            page = os.sysconf("SC_PAGE_SIZE")
            rss = int(fields[1]) * page
            peak = rss
            try:
                with open(f"{base}/status", "r", encoding="ascii") as fh:
                    for line in fh:
                        if line.startswith("VmHWM:"):
                            peak = int(line.split()[1]) * 1024
            except OSError:
                pass
            return {
                "rss": rss,
                "peak_rss": peak,
                "commit": rss,
                "peak_commit": peak,
                "source": "proc",
            }
        except OSError:
            return _unknown(f"proc(no such pid {pid})")
    return _unknown("none")


def _unknown(why: str) -> dict:
    return {
        "rss": -1,
        "peak_rss": -1,
        "commit": -1,
        "peak_commit": -1,
        "source": why,
    }


def descendants(pid: int) -> list[int]:
    """All descendants of ``pid``, including itself.

    Needed because the venv's ``python.exe`` is a **trampoline**: it spawns the
    real interpreter as a child and mirrors its exit code. So the pid returned by
    ``subprocess.Popen`` owns almost no memory (0.1 MiB observed), while the
    process that actually holds the model is one level down -- which made an
    8-rank run look like it used 10 MiB per rank.
    """
    impl = _init()
    if impl == "psutil":
        import psutil

        try:
            proc = psutil.Process(pid)
            return [pid] + [child.pid for child in proc.children(recursive=True)]
        except Exception:  # pragma: no cover - process gone
            return [pid]
    if impl == "windows":
        return _descendants_windows(pid)
    if impl == "proc":
        children: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "r", encoding="ascii") as fh:
                    fields = fh.read().rsplit(")", 1)[1].split()
                children.setdefault(int(fields[1]), []).append(int(entry))
            except (OSError, IndexError):
                continue
        out = [pid]
        queue = [pid]
        while queue:
            for child in children.get(queue.pop(), []):
                out.append(child)
                queue.append(child)
        return out
    return [pid]


def _descendants_windows(pid: int) -> list[int]:
    """Walk the process tree with a Toolhelp snapshot."""
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
    k32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot:
        return [pid]
    children: dict[int, list[int]] = {}
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    try:
        if k32.Process32First(snapshot, ctypes.byref(entry)):
            while True:
                children.setdefault(entry.th32ParentProcessID, []).append(
                    entry.th32ProcessID
                )
                if not k32.Process32Next(snapshot, ctypes.byref(entry)):
                    break
    finally:
        k32.CloseHandle(snapshot)

    out = [pid]
    queue = [pid]
    while queue:
        for child in children.get(queue.pop(), []):
            out.append(child)
            queue.append(child)
    return out


def tree_memory(pid: int) -> dict:
    """Summed memory over ``pid`` and its descendants."""
    total = {"rss": 0, "peak_rss": 0, "commit": 0, "peak_commit": 0}
    seen = 0
    for one in descendants(pid):
        info = process_memory(one)
        if info["rss"] < 0:
            continue
        seen += 1
        for key in total:
            total[key] += info[key]
    if not seen:
        return _unknown(f"tree(no reachable process for {pid})")
    total["processes"] = seen
    total["source"] = _init()
    return total


def system_memory() -> dict:
    """Physical RAM and commit limit/usage, in bytes (-1 if unknown)."""
    impl = _init()
    if impl == "psutil":
        import psutil

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "total_phys": int(vm.total),
            "avail_phys": int(vm.available),
            "commit_limit": int(vm.total + sm.total),
            "commit_used": int(vm.total - vm.available + sm.used),
            "source": "psutil",
        }
    if impl == "windows":
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not _k32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_phys": -1, "avail_phys": -1, "commit_limit": -1,
                    "commit_used": -1, "source": "windows(query failed)"}
        return {
            "total_phys": int(status.ullTotalPhys),
            "avail_phys": int(status.ullAvailPhys),
            "commit_limit": int(status.ullTotalPageFile),
            "commit_used": int(status.ullTotalPageFile - status.ullAvailPageFile),
            "memory_load_pct": int(status.dwMemoryLoad),
            "source": "windows",
        }
    if impl == "proc":
        info = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        return {
            "total_phys": info.get("MemTotal", -1),
            "avail_phys": info.get("MemAvailable", -1),
            "commit_limit": info.get("CommitLimit", -1),
            "commit_used": info.get("Committed_AS", -1),
            "source": "proc",
        }
    return {"total_phys": -1, "avail_phys": -1, "commit_limit": -1,
            "commit_used": -1, "source": "none"}


def gib(value: float) -> float:
    return value / (1024**3) if value is not None and value >= 0 else -1.0


if __name__ == "__main__":
    proc = process_memory()
    system = system_memory()
    print(f"source: {proc['source']}")
    print(f"  this process  rss {gib(proc['rss']):.2f} GiB | "
          f"peak {gib(proc['peak_rss']):.2f} GiB | "
          f"commit {gib(proc['commit']):.2f} GiB")
    print(f"  system        phys {gib(system['total_phys']):.2f} GiB | "
          f"available {gib(system['avail_phys']):.2f} GiB")
    print(f"  commit        used {gib(system['commit_used']):.2f} GiB / "
          f"limit {gib(system['commit_limit']):.2f} GiB")
