import json
import subprocess
from unittest.mock import patch

import pytest

from rec_mlx.kernels import device


@pytest.mark.parametrize(
    "name,architecture,workers,mlp,head,chunk,specialize",
    [
        ("M4 Pro", "applegpu_g16s", 36, 32, 32, 2048, False),
        ("M4 Max", "applegpu_g16s", 64, 64, 64, 2048, False),
        ("M5 Pro", "applegpu_g17s", 36, 32, 32, 4096, True),
        ("M5 Max", "applegpu_g17s", 76, 64, 64, 4096, True),
    ],
)
def test_device_profiles(name, architecture, workers, mlp, head, chunk, specialize):
    with patch.object(device, "_m4_max_gpu_cores", return_value=40) as probe:
        profile = device.profile_for_device(
            {"device_name": f"Apple {name}", "architecture": architecture}
        )
    assert profile == device.DeviceProfile(
        name, architecture, workers, mlp, head, chunk, specialize
    )
    assert probe.call_count == (1 if name == "M4 Max" else 0)


def test_unqualified_m4_max_rejected():
    with patch.object(device, "_m4_max_gpu_cores", return_value=32):
        with pytest.raises(ValueError, match="requires 40 GPU cores; detected 32"):
            device.profile_for_device(
                {"device_name": "Apple M4 Max", "architecture": "applegpu_g16s"}
            )


@pytest.mark.parametrize(
    "name,architecture", [("Apple M4", "applegpu_g16s"), ("Apple M4 Max", "applegpu_g17s")]
)
def test_unsupported_identity_rejected_before_probe(name, architecture):
    with patch.object(device, "_m4_max_gpu_cores") as probe:
        with pytest.raises(ValueError, match="Unsupported resident kernel device"):
            device.profile_for_device({"device_name": name, "architecture": architecture})
        probe.assert_not_called()


def test_core_probe_parses_and_caches():
    device._m4_max_gpu_cores.cache_clear()
    report = {"SPDisplaysDataType": [{"sppci_model": "Apple M4 Max", "sppci_cores": "40"}]}
    try:
        with patch.object(
            device.subprocess, "check_output", return_value=json.dumps(report)
        ) as run:
            assert device._m4_max_gpu_cores() == 40
            assert device._m4_max_gpu_cores() == 40
            run.assert_called_once_with(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=10
            )
    finally:
        device._m4_max_gpu_cores.cache_clear()


@pytest.mark.parametrize("output", ["invalid json", "{}", '{"SPDisplaysDataType": []}'])
def test_unknown_core_count_fails_closed(output):
    device._m4_max_gpu_cores.cache_clear()
    with patch.object(device.subprocess, "check_output", return_value=output):
        with pytest.raises(ValueError, match="Cannot verify M4 Max GPU core count"):
            device._m4_max_gpu_cores()


def test_core_probe_timeout_fails_closed():
    device._m4_max_gpu_cores.cache_clear()
    with patch.object(
        device.subprocess, "check_output", side_effect=subprocess.TimeoutExpired("probe", 10)
    ):
        with pytest.raises(ValueError, match="Cannot verify M4 Max GPU core count"):
            device._m4_max_gpu_cores()
