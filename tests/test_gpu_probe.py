from __future__ import annotations

from pathlib import Path

import numpy as np

from arw_denoise.engines import DenoiseResult, EngineInfo, EngineRunStats
from arw_denoise.gpu_probe import GpuDeviceInfo, GpuProbe


class FakeEngine:
    def __init__(self, provider: str = "CUDAExecutionProvider"):
        self.provider = provider
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return DenoiseResult(
            packed=request.packed.copy(),
            engine=EngineInfo(
                engine_id="onnx-pmrid",
                display_name="PMRID",
                provider=self.provider,
                is_gpu=self.provider == "CUDAExecutionProvider",
                model_id="pmrid-general-raw",
                model_version="1.0.0",
            ),
            stats=EngineRunStats(inference_seconds=0.125, tile_size=request.packed.shape[0]),
        )


def test_gpu_probe_runs_real_inference_contract_and_caches_result() -> None:
    engine = FakeEngine()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return engine

    probe = GpuProbe(
        factory,
        device_query=lambda: GpuDeviceInfo("RTX test", 8192, "581.57"),
        test_size=32,
    )
    first = probe.run()
    second = probe.run()

    assert first is second
    assert first.success
    assert first.device_name == "RTX test"
    assert first.memory_total_mb == 8192
    assert first.provider == "CUDAExecutionProvider"
    assert first.model_id == "pmrid-general-raw"
    assert first.inference_seconds == 0.125
    assert factory_calls == 1
    assert engine.calls == 1


def test_gpu_probe_force_repeats_inference() -> None:
    engine = FakeEngine()
    probe = GpuProbe(lambda: engine, device_query=lambda: GpuDeviceInfo("RTX", 8192, "1"), test_size=32)
    probe.run()
    probe.run(force=True)
    assert engine.calls == 2


def test_gpu_probe_returns_actionable_failure_instead_of_raising() -> None:
    def factory():
        raise RuntimeError("cudnn64_9.dll missing")

    probe = GpuProbe(factory, device_query=lambda: GpuDeviceInfo("RTX", 8192, "1"), test_size=32)
    result = probe.run()

    assert not result.success
    assert "cudnn64_9.dll" in (result.error or "")


def test_gpu_probe_rejects_cpu_fallback() -> None:
    engine = FakeEngine(provider="CPUExecutionProvider")
    probe = GpuProbe(lambda: engine, device_query=lambda: GpuDeviceInfo("RTX", 8192, "1"), test_size=32)
    result = probe.run()
    assert not result.success
    assert "CPU" in (result.error or "")

