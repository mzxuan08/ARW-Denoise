from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .engines import DenoiseRequest, RawDenoiseEngine
from .model_manifest import default_model_root, load_manifest
from .onnx_engine import OnnxRuntimeEngine


@dataclass(frozen=True)
class GpuDeviceInfo:
    name: str
    memory_total_mb: int | None
    driver_version: str | None


@dataclass(frozen=True)
class GpuProbeResult:
    success: bool
    device_name: str
    memory_total_mb: int | None
    driver_version: str | None
    provider: str | None
    model_id: str | None
    model_version: str | None
    inference_seconds: float | None
    error: str | None = None


def query_nvidia_device() -> GpuDeviceInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuDeviceInfo("NVIDIA GPU（名称未知）", None, None)
    if result.returncode != 0 or not result.stdout.strip():
        return GpuDeviceInfo("NVIDIA GPU（名称未知）", None, None)
    try:
        row = next(csv.reader([result.stdout.splitlines()[0]], skipinitialspace=True))
        return GpuDeviceInfo(row[0].strip(), int(row[1].strip()), row[2].strip())
    except (IndexError, ValueError, StopIteration):
        return GpuDeviceInfo("NVIDIA GPU（名称未知）", None, None)


class GpuProbe:
    def __init__(
        self,
        engine_factory: Callable[[], RawDenoiseEngine],
        *,
        device_query: Callable[[], GpuDeviceInfo] = query_nvidia_device,
        test_size: int = 64,
    ):
        if test_size < 16 or test_size % 16:
            raise ValueError("GPU 自检尺寸必须是不小于 16 的 16 倍数")
        self.engine_factory = engine_factory
        self.device_query = device_query
        self.test_size = test_size
        self._cached: GpuProbeResult | None = None

    def run(self, *, force: bool = False) -> GpuProbeResult:
        if self._cached is not None and not force:
            return self._cached
        device = self.device_query()
        try:
            engine = self.engine_factory()
            axis = np.linspace(0.05, 0.35, self.test_size, dtype=np.float32)
            plane = np.add.outer(axis, axis) * np.float32(0.5)
            packed = np.stack([plane, plane * 0.95, plane * 1.05, plane], axis=-1)
            result = engine.run(DenoiseRequest(packed=packed, effective_iso=1600.0, strength=1.0))
            if not result.engine.is_gpu or result.engine.provider != "CUDAExecutionProvider":
                raise RuntimeError(f"GPU 自检实际使用了 {result.engine.provider or 'CPU'}")
            if result.packed.shape != packed.shape or not np.isfinite(result.packed).all():
                raise RuntimeError("GPU 自检输出无效")
            value = GpuProbeResult(
                success=True,
                device_name=device.name,
                memory_total_mb=device.memory_total_mb,
                driver_version=device.driver_version,
                provider=result.engine.provider,
                model_id=result.engine.model_id,
                model_version=result.engine.model_version,
                inference_seconds=result.stats.inference_seconds,
            )
        except Exception as exc:
            value = GpuProbeResult(
                success=False,
                device_name=device.name,
                memory_total_mb=device.memory_total_mb,
                driver_version=device.driver_version,
                provider=None,
                model_id=None,
                model_version=None,
                inference_seconds=None,
                error=str(exc) or exc.__class__.__name__,
            )
        self._cached = value
        return value


def create_default_gpu_probe(
    *,
    model_root: Path | None = None,
    dll_directory: Path | None = None,
    test_size: int = 64,
) -> GpuProbe:
    root = (model_root or default_model_root()).resolve()

    def factory() -> OnnxRuntimeEngine:
        manifest = load_manifest(root / "pmrid" / "manifest.json")
        return OnnxRuntimeEngine(manifest, dll_directory=dll_directory, require_cuda=True)

    return GpuProbe(factory, test_size=test_size)

