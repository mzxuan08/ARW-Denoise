from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .domain import ArwDenoiseError
from .engines import DenoiseRequest, DenoiseResult, EngineInfo, EngineRunStats
from .model_manifest import ModelManifest


class GpuRuntimeError(ArwDenoiseError):
    """The GPU runtime or model session cannot safely run inference."""


def _version_tuple(value: str) -> tuple[int, int, int]:
    try:
        parts = value.split(".")[:3]
        return tuple(int(part.split("+")[0].split("-")[0]) for part in parts)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise GpuRuntimeError(f"无法解析 ONNX Runtime 版本：{value}") from exc


class OnnxRuntimeEngine:
    def __init__(
        self,
        manifest: ModelManifest,
        *,
        runtime_module: Any | None = None,
        dll_directory: Path | None = None,
        require_cuda: bool = True,
        device_id: int = 0,
    ):
        self.manifest = manifest
        self.runtime = runtime_module or self._import_runtime()
        self.dll_directory = Path(dll_directory).resolve() if dll_directory else None
        self.require_cuda = require_cuda
        self.device_id = device_id
        self._dll_handles: list[Any] = []
        self._register_dll_directories()
        self._preload_runtime_dlls()
        self._validate_runtime_version()
        self.session = self._create_session()
        self._validate_session_contract()
        self._raw_buffer: np.ndarray | None = None
        self._noise_buffer = np.empty((1, 1, 1, 1), dtype=np.float32)

    @staticmethod
    def _import_runtime() -> Any:
        try:
            return importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise GpuRuntimeError("缺少 ONNX Runtime GPU 运行库；请修复安装包或切换 CPU") from exc

    def _preload_runtime_dlls(self) -> None:
        if not hasattr(self.runtime, "preload_dlls"):
            return
        try:
            self.runtime.preload_dlls(
                cuda=True,
                cudnn=True,
                msvc=True,
                directory=str(self.dll_directory) if self.dll_directory is not None else "",
            )
        except Exception as exc:
            raise GpuRuntimeError(
                f"无法加载离线 CUDA/cuDNN 运行库：{exc}。请重新解压完整安装包或切换 CPU"
            ) from exc

    def _register_dll_directories(self) -> None:
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is None:
            return
        candidates: list[Path] = []
        if self.dll_directory is not None:
            candidates.append(self.dll_directory)
            candidates.extend(path.parent for path in self.dll_directory.rglob("*.dll"))
        else:
            for entry in sys.path:
                nvidia_root = Path(entry) / "nvidia"
                if nvidia_root.is_dir():
                    candidates.extend(path for path in nvidia_root.glob("*/bin") if path.is_dir())
        seen: set[Path] = set()
        registered: list[str] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            registered.append(str(resolved))
            try:
                self._dll_handles.append(add_directory(str(resolved)))
            except OSError as exc:
                raise GpuRuntimeError(f"无法注册 CUDA DLL 目录：{resolved}") from exc
        if registered:
            current_path = os.environ.get("PATH", "")
            existing = {item.casefold() for item in current_path.split(os.pathsep) if item}
            additions = [item for item in registered if item.casefold() not in existing]
            if additions:
                os.environ["PATH"] = os.pathsep.join([*additions, current_path])

    def _validate_runtime_version(self) -> None:
        actual = str(getattr(self.runtime, "__version__", "0.0.0"))
        required = self.manifest.runtime.minimum_onnxruntime
        if _version_tuple(actual) < _version_tuple(required):
            raise GpuRuntimeError(f"ONNX Runtime {actual} 低于模型要求的 {required}")

    def _provider_configuration(self) -> list[Any]:
        available = set(self.runtime.get_available_providers())
        cuda = "CUDAExecutionProvider"
        if self.require_cuda and cuda not in available:
            raise GpuRuntimeError("未找到 CUDAExecutionProvider；请更新 NVIDIA 驱动或切换 CPU")
        providers: list[Any] = []
        if cuda in available:
            providers.append(
                (
                    cuda,
                    {
                        "device_id": self.device_id,
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": 1,
                    },
                )
            )
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            raise GpuRuntimeError("ONNX Runtime 没有可用的执行 provider")
        return providers

    def _create_session(self) -> Any:
        providers = self._provider_configuration()
        try:
            session = self.runtime.InferenceSession(str(self.manifest.artifact_path), providers=providers)
        except Exception as exc:
            raise GpuRuntimeError(f"无法创建 PMRID ONNX 会话：{exc}") from exc
        actual = session.get_providers()
        if self.require_cuda and (not actual or actual[0] != "CUDAExecutionProvider"):
            raise GpuRuntimeError("PMRID 会话未实际启用 CUDAExecutionProvider")
        return session

    def _validate_session_contract(self) -> None:
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = {item.name: item for item in self.session.get_outputs()}
        expected_inputs = {self.manifest.input.name, self.manifest.noise_input.name}
        if set(inputs) != expected_inputs or set(outputs) != {self.manifest.output.name}:
            raise GpuRuntimeError("ONNX 模型输入输出名称与清单不一致")
        raw_input = inputs[self.manifest.input.name]
        iso_input = inputs[self.manifest.noise_input.name]
        output = outputs[self.manifest.output.name]
        if raw_input.type != "tensor(float)" or iso_input.type != "tensor(float)" or output.type != "tensor(float)":
            raise GpuRuntimeError("ONNX 模型输入输出精度与清单不一致")
        if len(raw_input.shape) != 4 or raw_input.shape[1] != 4:
            raise GpuRuntimeError("ONNX RAW 输入不是 NCHW 四通道")
        if list(iso_input.shape[1:]) != [1, 1, 1]:
            raise GpuRuntimeError("ONNX 等效 ISO 输入形状无效")
        if len(output.shape) != 4 or output.shape[1] != 4:
            raise GpuRuntimeError("ONNX RAW 输出不是 NCHW 四通道")

    @property
    def info(self) -> EngineInfo:
        provider = self.session.get_providers()[0]
        return EngineInfo(
            engine_id="onnx-pmrid",
            display_name=self.manifest.display_name,
            provider=provider,
            is_gpu=provider == "CUDAExecutionProvider",
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
        )

    def run(self, request: DenoiseRequest) -> DenoiseResult:
        request.validate()
        height, width, _ = request.packed.shape
        if height % 16 or width % 16:
            raise GpuRuntimeError("PMRID 输入高宽必须能被 16 整除")
        raw_shape = (1, 4, height, width)
        if self._raw_buffer is None or self._raw_buffer.shape != raw_shape:
            self._raw_buffer = np.empty(raw_shape, dtype=np.float32)
        raw = self._raw_buffer
        np.copyto(raw[0], request.packed.transpose(2, 0, 1), casting="same_kind")
        low, high = self.manifest.noise_input.value_range
        self._noise_buffer[0, 0, 0, 0] = np.clip(request.effective_iso, low, high)
        effective_iso = self._noise_buffer
        started = time.perf_counter()
        try:
            values = self.session.run(
                [self.manifest.output.name],
                {self.manifest.input.name: raw, self.manifest.noise_input.name: effective_iso},
            )
        except Exception as exc:
            raise GpuRuntimeError(f"ONNX RAW 推理失败：{exc}") from exc
        if self.require_cuda and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise GpuRuntimeError("ONNX Runtime 在推理时回退到 CPU，GPU 自检失败")
        elapsed = time.perf_counter() - started
        if len(values) != 1:
            raise GpuRuntimeError("ONNX 模型输出数量无效")
        prediction = np.asarray(values[0])
        if prediction.shape != raw.shape or not np.isfinite(prediction).all():
            raise GpuRuntimeError("ONNX 模型输出形状无效或包含 NaN/无穷值")
        model_output = prediction[0].transpose(1, 2, 0).astype(np.float32, copy=False)
        return DenoiseResult(
            packed=np.clip(model_output, 0.0, 1.0).astype(np.float32, copy=False),
            engine=self.info,
            stats=EngineRunStats(inference_seconds=elapsed, tile_size=height),
        )
