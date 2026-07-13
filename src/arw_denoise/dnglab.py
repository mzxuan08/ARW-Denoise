from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .domain import ExternalToolError, RawMetadata
from .dngwrite import replace_cfa_pixels_in_place, snapshot_dng_metadata, validate_processed_dng


@dataclass(frozen=True)
class DngLabResult:
    output: Path
    version: str
    analysis: dict | None


class DngLabClient:
    def __init__(self, executable: Path | str | None = None, timeout_seconds: int = 300):
        discovered = self._discover(executable)
        if not discovered:
            raise ExternalToolError("未找到 dnglab；请安装或在设置中指定 dnglab.exe")
        self.executable = Path(discovered)
        self.timeout_seconds = timeout_seconds

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

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [str(self.executable), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolError(f"dnglab 运行超过 {self.timeout_seconds} 秒") from exc
        except OSError as exc:
            raise ExternalToolError(f"无法启动 dnglab：{exc}") from exc

    def version(self) -> str:
        result = self._run("--version")
        text = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            raise ExternalToolError(f"无法读取 dnglab 版本：{text}")
        return text

    def analyze(self, path: Path) -> dict:
        result = self._run("analyze", "--structure", "--json", str(path))
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise ExternalToolError(f"DNG 校验失败：{message}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError("dnglab 未返回有效 JSON 结构") from exc

    def compatibility_convert(self, source: Path, output: Path, embed_raw: bool = False) -> DngLabResult:
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
            tool_version = self.version()
            result = self._run(
                "convert",
                "--compression", "lossless",
                "--embed-raw", "true" if embed_raw else "false",
                str(source),
                str(temporary),
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                message = (result.stderr or result.stdout).strip()
                raise ExternalToolError(f"dnglab 转换失败：{message}")
            analysis = self.analyze(temporary)
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
            tool_version = self.version()
            result = self._run(
                "convert", "--compression", "uncompressed", "--embed-raw", "false",
                str(source), str(temporary),
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                message = (result.stderr or result.stdout).strip()
                raise ExternalToolError(f"dnglab 基础 DNG 转换失败：{message}")
            metadata_before = snapshot_dng_metadata(temporary)
            replace_cfa_pixels_in_place(temporary, processed_visible, metadata)
            if snapshot_dng_metadata(temporary) != metadata_before:
                raise ExternalToolError("写回 CFA 像素时 DNG 元数据发生变化")
            analysis = self.analyze(temporary)
            validate_processed_dng(temporary, processed_visible, metadata)
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
