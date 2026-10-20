"""GPU names and UUIDs from NVML, keyed by PCI address. Optional: returns nothing if NVML is unavailable."""

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
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            bus_id = pynvml.nvmlDeviceGetPciInfo(h).busId
            bus_id = bus_id.decode() if isinstance(bus_id, bytes) else bus_id
            name = pynvml.nvmlDeviceGetName(h)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            out[normalize_bdf(bus_id)] = (
                name.decode() if isinstance(name, bytes) else name,
                uuid.decode() if isinstance(uuid, bytes) else uuid,
            )
    finally:
        pynvml.nvmlShutdown()
    return out
