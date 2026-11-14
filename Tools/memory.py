from __future__ import annotations

import ctypes
import gc
import os
from typing import Optional


_GIB = float(1024 ** 3)


def _proc_value_gib(path: str, key: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    # Linux /proc reports these fields in KiB.
                    return float(line.split()[1]) * 1024.0 / _GIB
    except (OSError, ValueError, IndexError):
        pass
    return None


def current_rss_gib() -> Optional[float]:
    return _proc_value_gib("/proc/self/status", "VmRSS")


def _cgroup_available_gib() -> Optional[float]:
    candidates = (
        (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.stat",
        ),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.stat",
        ),
    )
    for limit_path, usage_path, stat_path in candidates:
        try:
            with open(limit_path, "r", encoding="utf-8") as handle:
                limit_text = handle.read().strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            if limit >= (1 << 60):
                continue

            with open(usage_path, "r", encoding="utf-8") as handle:
                usage = int(handle.read().strip())

            reclaimable = 0
            try:
                with open(stat_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) != 2:
                            continue
                        key, value = parts[0], int(parts[1])
                        if key in ("file", "slab_reclaimable"):
                            reclaimable += value
            except (OSError, ValueError):
                reclaimable = 0

            used_unreclaimable = max(0, usage - reclaimable)
            return max(0.0, float(limit - used_unreclaimable) / _GIB)
        except (OSError, ValueError):
            continue
    return None


def available_memory_gib() -> Optional[float]:
    host_available = _proc_value_gib("/proc/meminfo", "MemAvailable")
    if host_available is None:
        try:
            host_available = (
                os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / _GIB
            )
        except (AttributeError, OSError, ValueError):
            pass
    cgroup_available = _cgroup_available_gib()
    values = [x for x in (host_available, cgroup_available) if x is not None]
    return min(values) if values else None


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def memory_status(stage: str) -> str:
    rss = current_rss_gib()
    available = available_memory_gib()
    rss_text = "unknown" if rss is None else f"{rss:.2f} GiB"
    available_text = "unknown" if available is None else f"{available:.2f} GiB"
    return f"[memory] {stage}: RSS={rss_text}, available={available_text}"


def ensure_memory_headroom(stage: str, min_available_gib: float) -> None:
    release_memory()
    print(memory_status(stage), flush=True)
    available = available_memory_gib()
    if available is not None and available < float(min_available_gib):
        raise MemoryError(
            f"{stage}: only {available:.2f} GiB memory is available; "
            f"at least {float(min_available_gib):.2f} GiB is required before loading "
            "the next large matrix. Free container memory before retrying."
        )
