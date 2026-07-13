# ARW Denoise

面向 Sony A7C II 的开源 Bayer RAW 批量降噪桌面软件。项目当前处于首个可运行垂直切片：已经包含持久化任务队列、RAW 探测接口、Bayer 数值管线、CPU 小波基线、dnglab 兼容转换适配器和 PySide6 GUI 骨架。

> 重要：当前版本已实现保守的“无压缩基础 DNG + CFA 像素平面安全替换”路径，但尚未用真实 A7C II ARW 在 Adobe 与像素蛋糕中完成外部验收。`dng-convert` 只用于建立未经降噪的兼容基线；`process-cpu` 才会执行 CPU Bayer 降噪。软件不会把 RGB 图像伪装成 RAW DNG。

## 开发运行

```powershell
python -m pip install -e ".[dev,raw,gui]"
python -m arw_denoise --help
python -m arw_denoise gui
```

探测一张 ARW：

```powershell
python -m arw_denoise probe path\to\image.ARW
```

使用已安装的 dnglab 建立兼容 DNG：

```powershell
python -m arw_denoise dng-convert input.ARW output.DNG
```

安装 dnglab Windows 辅助程序并运行 CPU 降噪：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_dnglab.ps1
python -m arw_denoise process-cpu input.ARW output_DN.DNG --dnglab vendor/dnglab/dnglab.exe
```

## 安全约束

- 源 ARW 只读。
- 输出先写临时文件，再原子改名。
- 不静默覆盖已有文件。
- 缺少关键 RAW 元数据时拒绝处理。
- 单张任务失败不会破坏队列数据库。

完整设计和实施计划位于 `docs/superpowers/`。
