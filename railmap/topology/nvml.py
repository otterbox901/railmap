"""GPU names and UUIDs from NVML, keyed by PCI address. Optional: returns nothing if NVML is unavailable.

Inside a container or a GPU-constrained job, NVML lists only the GPUs this process was given, unlike sysfs.
"""

from __future__ import annotations

from railmap.topology.sysfs import normalize_bdf


def gpu_info() -> dict[str, tuple[str, str]]:
    try:
        import pynvml
    except ImportError:
        return {}
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return {}
    out: dict[str, tuple[str, str]] = {}
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                bus_id = pynvml.nvmlDeviceGetPciInfo(h).busId
                name = pynvml.nvmlDeviceGetName(h)
                uuid = pynvml.nvmlDeviceGetUUID(h)
            except pynvml.NVMLError:
                continue  # e.g. NO_PERMISSION for a GPU the device cgroup hides from this job
            bus_id = bus_id.decode() if isinstance(bus_id, bytes) else bus_id
            out[normalize_bdf(bus_id)] = (
                name.decode() if isinstance(name, bytes) else name,
                uuid.decode() if isinstance(uuid, bytes) else uuid,
            )
    except pynvml.NVMLError:
        return out
    finally:
        pynvml.nvmlShutdown()
    return out
