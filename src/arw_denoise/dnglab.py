from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .dngwrite import (
    remove_pixel_dependent_tags,
    replace_cfa_pixels_in_place,
    snapshot_dng_metadata,
    validate_processed_dng,
)
from .domain import ExternalToolError, RawMetadata
from .task_control import CancellationToken, ProcessingCancelled


@dataclass(frozen=True)
class DngLabResult:
    output: Path
    version: str
    analysis: dict | None


class DngLabClient:
    def __init__(
        self,
        executable: Path | str | None = None,
        timeout_seconds: int = 300,
        *,
        poll_seconds: float = 0.1,
    ):
        discovered = self._discover(executable)
        if not discovered:
            raise ExternalToolError("未找到 dnglab，请安装或在设置中指定 dnglab.exe")
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("dnglab 超时和轮询间隔必须大于零")
        self.executable = Path(discovered)
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.popen_factory = subprocess.Popen

    @staticmethod
    def _discover(explicit: Path | str | None) -> str | None:
        if explicit:
            return str(explicit)
        environment = os.environ.get("ARW_DENOISE_DNGLAB")
        if environment:
            return environment
        candidates = (
            Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) / "tools" / "dnglab.exe",
            Path(sys.executable).resolve().parent / "tools" / "dnglab.exe",
            Path(__file__).resolve().parent / "bin" / "dnglab.exe",
            Path.cwd() / "vendor" / "dnglab" / "dnglab.exe",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return shutil.which("dnglab")

    @staticmethod
    def _terminate_process(process: object) -> None:
        try:
            process.terminate()  # type: ignore[attr-defined]
            process.communicate(timeout=2.0)  # type: ignore[attr-defined]
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()  # type: ignore[attr-defined]
                process.communicate(timeout=2.0)  # type: ignore[attr-defined]
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _run(
        self,
        *args: str,
        cancellation: CancellationToken | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(self.executable), *args]
        if cancellation is not None:
            cancellation.check()
        try:
            process = self.popen_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ExternalToolError(f"无法启动 dnglab：{exc}") from exc
        started = time.monotonic()
        while True:
            try:
                if cancellation is not None:
                    cancellation.check()
            except ProcessingCancelled:
                self._terminate_process(process)
                raise
            remaining = self.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                self._terminate_process(process)
                raise ExternalToolError(f"dnglab 运行超过 {self.timeout_seconds} 秒")
            try:
                stdout, stderr = process.communicate(timeout=min(self.poll_seconds, remaining))
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue

    def version(self, cancellation: CancellationToken | None = None) -> str:
        result = self._run("--version", cancellation=cancellation)
        text = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            raise ExternalToolError(f"无法读取 dnglab 版本：{text}")
        return text

    def analyze(self, path: Path, cancellation: CancellationToken | None = None) -> dict:
        result = self._run("analyze", "--structure", "--json", str(path), cancellation=cancellation)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise ExternalToolError(f"DNG 校验失败：{message}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError("dnglab 未返回有效 JSON 结构") from exc

    def metadata(self, path: Path, cancellation: CancellationToken | None = None) -> dict:
        result = self._run("analyze", "--meta", "--json", str(path), cancellation=cancellation)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise ExternalToolError(f"RAW 元数据读取失败：{message}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError("dnglab 未返回有效的 RAW 元数据 JSON") from exc

    def compatibility_convert(
        self,
        source: Path,
        output: Path,
        embed_raw: bool = False,
        *,
        cancellation: CancellationToken | None = None,
    ) -> DngLabResult:
        """Create an untouched compatibility DNG; this does not denoise pixels."""
        source = Path(source).resolve()
        output = Path(output).resolve()
        if not source.is_file():
            raise ExternalToolError(f"找不到输入文件：{source.name}")
        if output.exists():
            raise ExternalToolError(f"拒绝覆盖已有输出：{output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.dng")
        try:
            tool_version = self.version(cancellation)
            result = self._run(
                "convert",
                "--compression", "lossless",
                "--embed-raw", "true" if embed_raw else "false",
                str(source),
                str(temporary),
                cancellation=cancellation,
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                message = (result.stderr or result.stdout).strip()
                raise ExternalToolError(f"dnglab 转换失败：{message}")
            analysis = self.analyze(temporary, cancellation)
            if cancellation is not None:
                cancellation.check()
            self._publish_no_overwrite(temporary, output)
            return DngLabResult(output=output, version=tool_version, analysis=analysis)
        finally:
            temporary.unlink(missing_ok=True)

    def write_processed_cfa(
        self,
        source: Path,
        output: Path,
        processed_visible: "object",
        metadata: RawMetadata,
        *,
        cancellation: CancellationToken | None = None,
    ) -> DngLabResult:
        """Create a metadata-preserving uncompressed DNG and replace its CFA samples."""
        source = Path(source).resolve()
        output = Path(output).resolve()
        if not source.is_file():
            raise ExternalToolError(f"找不到输入文件：{source.name}")
        if output.exists():
            raise ExternalToolError(f"拒绝覆盖已有输出：{output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.processing.dng")
        try:
            tool_version = self.version(cancellation)
            result = self._run(
                "convert", "--compression", "uncompressed", "--embed-raw", "false",
                str(source), str(temporary), cancellation=cancellation,
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                message = (result.stderr or result.stdout).strip()
                raise ExternalToolError(f"dnglab 基础 DNG 转换失败：{message}")
            metadata_before = snapshot_dng_metadata(temporary)
            if cancellation is not None:
                cancellation.check()
            replace_cfa_pixels_in_place(temporary, processed_visible, metadata)
            if cancellation is not None:
                cancellation.check()
            remove_pixel_dependent_tags(temporary)
            if snapshot_dng_metadata(temporary) != metadata_before:
                raise ExternalToolError("写回 CFA 像素时 DNG 元数据发生变化")
            analysis = self.analyze(temporary, cancellation)
            validate_processed_dng(temporary, processed_visible, metadata)
            if cancellation is not None:
                cancellation.check()
            self._publish_no_overwrite(temporary, output)
            return DngLabResult(output=output, version=tool_version, analysis=analysis)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _publish_no_overwrite(temporary: Path, output: Path) -> None:
        """Atomically publish on the same volume without replacing another process's file."""
        try:
            if os.name == "nt":
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
                temporary.unlink()
        except FileExistsError as exc:
            raise ExternalToolError(f"发布时输出已存在：{output.name}") from exc
        except OSError as exc:
            raise ExternalToolError(f"无法原子发布 DNG：{exc}") from exc
