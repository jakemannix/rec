import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceProfile:
    chip: str
    architecture: str
    workers: int = 36
    mlp_workers: int = 32
    head_workers: int = 32
    prefill_chunk_size: int = 2048
    specialize_mixed_experts: bool = True


@lru_cache(maxsize=1)
def _m4_max_gpu_cores():
    # MLX reports the same device name for the 32- and 40-core M4 Max.
    try:
        report = json.loads(
            subprocess.check_output(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
                text=True,
                timeout=10,
            )
        )
        counts = {
            int(gpu["sppci_cores"])
            for gpu in report["SPDisplaysDataType"]
            if gpu.get("sppci_model") == "Apple M4 Max"
        }
        if len(counts) != 1:
            raise ValueError("missing or ambiguous GPU core count")
        return counts.pop()
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot verify M4 Max GPU core count; this build requires 40 GPU cores"
        ) from exc


def profile_for_device(info):
    chips = {
        ("applegpu_g16s", "Apple M4 Pro"): "M4 Pro",
        ("applegpu_g16s", "Apple M4 Max"): "M4 Max",
        ("applegpu_g17s", "Apple M5 Pro"): "M5 Pro",
        ("applegpu_g17s", "Apple M5 Max"): "M5 Max",
    }
    architecture = info["architecture"]
    identity = (architecture, info["device_name"])
    if identity not in chips:
        raise ValueError(f"Unsupported resident kernel device: {identity}")
    if chips[identity] == "M4 Max":
        cores = _m4_max_gpu_cores()
        if cores != 40:
            raise ValueError(f"M4 Max resident profile requires 40 GPU cores; detected {cores}")
        return DeviceProfile(
            chip="M4 Max",
            architecture=architecture,
            workers=64,
            mlp_workers=64,
            head_workers=64,
            prefill_chunk_size=2048,
            specialize_mixed_experts=False,
        )
    if chips[identity] == "M5 Max":
        return DeviceProfile(
            chip=chips[identity],
            architecture=architecture,
            workers=76,
            mlp_workers=64,
            head_workers=64,
            prefill_chunk_size=4096,
        )
    return DeviceProfile(
        chip=chips[identity],
        architecture=architecture,
        prefill_chunk_size=4096 if chips[identity] == "M5 Pro" else 2048,
        # The specialized M4 resident shader fails scalar/packet parity.
        specialize_mixed_experts=chips[identity] != "M4 Pro",
    )
