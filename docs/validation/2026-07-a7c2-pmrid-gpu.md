# Sony A7C II / PMRID GPU 验证记录

验证日期：2026-07-14  
设备：Sony ILCE-7CM2（A7C II）、NVIDIA GeForce RTX 3070 Ti Laptop GPU 8 GB  
驱动：581.57  
模型：PMRID general Bayer RAW `1.0.0`，ONNX FP16  
执行器：ONNX Runtime GPU `1.23.2` / `CUDAExecutionProvider`

## 整图结果

| 源文件 | ISO | 自动强度 | GPU 推理 | tile | 输出 SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `DSC01959.ARW` | 400 | 0.48 | 3.868 s | 1024 | `09e0e9cb2ff1a7eb8316422fa72523b5d103c6396708dc26fda9047eb9b62edb` |
| `DSC01928.ARW` | 100 | 0.35 | 3.756 s | 1024 | `3cfc51ee4bd4d3860441645e196d93f78e6f1ba0977d1a7a3bb60090de63070c` |

总处理还包括 ARW 解码、后处理、DNG 无损写入与重新校验；GPU 纯推理时间不等于单张总耗时。

## RAW 保真检查

- 两张输出均为 `4688 x 7032` 的 Bayer CFA DNG，可由 LibRaw/rawpy 重新打开。
- 保留原 CFA 排列、黑电平 512、标称白电平 15360、相机色彩/白平衡元数据和裁切尺寸。
- 低于黑电平或高于标称白电平的传感器数据 100% 原样写回，不截断 Sony 高光余量。
- ISO 400 样片平均像素偏移 `0.0004` 个 RAW 码值，平均绝对变化 `5.061`，最大绝对变化 `198`。
- ISO 100 样片平均像素偏移 `0.0103` 个 RAW 码值，平均绝对变化 `2.210`，最大绝对变化 `131`。
- dnglab `0.7.2` 完成写入后结构校验与像素面回读对比。

## 离线发布验证

发布目录在清空 `PATH`、`PYTHONPATH`、模型路径和 dnglab 路径后，独立完成 CUDA 真实推理自检，并通过 GUI 启动冒烟测试。包内自带 PySide6、rawpy/LibRaw、ONNX Runtime、CUDA/cuDNN/cuFFT 必需 DLL、PMRID ONNX 模型和 dnglab；运行时仅要求兼容的 NVIDIA 驱动。

Adobe Camera Raw/Lightroom 和像素蛋糕的最终主观画质/可编辑性确认需由用户用本页两张验收 DNG 完成。
