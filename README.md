# ARW Denoise 0.3.1

面向 Sony A7C II（ILCE-7CM2）的完整离线 Bayer RAW 批量降噪软件。使用开源 PMRID 神经网络和 ONNX Runtime CUDA，输出保留 CFA、白平衡、色彩元数据和传感器高光余量的可编辑 DNG。

## 使用发布包

1. 完整解压 `ArwDenoise-0.3.1-offline-win64.zip` 或体积更小的 `ArwDenoise-0.3.1-offline-win64-solid.7z`，不要只单独拖出 EXE。
2. 双击 `ArwDenoise.exe`。首次启动会自动执行一次真实 GPU 推理自检。
3. 在设置中选择默认导入目录，并选择“源目录 / DNG_Denoised”或固定导出目录。
4. 添加任意数量的 ARW 或整个文件夹，保持“全自动（推荐）”后开始处理。
5. 完成后点“打开导出目录”，或选中任务后点“定位选中 DNG”。
6. 选中已完成任务后可打开“对比预览”，在完全离线的独立窗口中同步缩放、拖动和分割对比源 ARW / 降噪 DNG。

## 缓存与磁盘空间

设置中的“缓存存放位置”可以选择 F 盘等数据盘。软件只会在所选位置下创建带安全标记的 `ArwDenoiseCache` 子目录，切换位置时自动清掉旧预览缓存；“打开缓存目录”和“清理缓存”按钮可随时查看或释放空间。清理操作不会删除同级的 ARW、DNG 或其他用户文件。

DNG 处理中的临时文件与最终导出文件放在同一目录，以保证写入成功后原子发布；任务失败或取消时会自动清理，不会长期占用空间。任务数据库和设置仍保存在 Windows 用户数据目录，但只包含路径、队列状态和少量文本信息。

高级设置默认折叠。只有勾选“启用手动高级参数”时，四个滑块才会覆盖基于 ISO 和 RAW 噪声估计的自动结果。自动模式优先使用 NVIDIA GPU；GPU 不可用时会记录原因并回退到保守 CPU 引擎。

0.3.0 增加当前照片/整队列进度、阶段、已用时间和 ETA。点击取消会终止当前 tile 或 dnglab 子进程，清理本任务临时文件并停止后续队列。“清理已完成记录”只清理历史，不会删除已导出 DNG。

0.3.1 增加可配置缓存位置、打开缓存目录和带安全标记的一键清理，适合把预览缓存放到非系统盘。

## 运行要求

- Windows 10/11 x64。
- 支持 CUDA 12 运行时的 NVIDIA 驱动。不需要安装 Python、CUDA Toolkit、cuDNN 或联网。
- 当前正式校验机型为 Sony A7C II；其他相机不会被默认猜测处理。
- 完整解压后约需 2.7 GB 空间。

PMRID 是通用 Bayer RAW 基线，不是 Sony 或 Adobe 的商业模型。请先用随包的 A7C II 验收 DNG 在 Adobe Camera Raw/Lightroom 或像素蛋糕中确认你的工作流。

## 开发与验证

```powershell
python -m pip install -e ".[dev,raw,gui,gpu]"
arw-denoise gpu-probe
python -m pytest
python scripts/stress_queue.py --count 150
```

构建和隔离验证：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/verify_offline_bundle.ps1 -Distribution dist/ArwDenoise
powershell -ExecutionPolicy Bypass -File scripts/package_release.ps1
```

源 ARW 始终只读；输出先写临时文件再发布，不静默覆盖旧文件。任务队列持久化，单任务失败不会阻断其他照片。
