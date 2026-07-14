# Sony ARW 通用能力检测与 A7C II 回归

## 自动化覆盖

- 0.4.0 完整测试：164 passed，2 skipped。
- 新增 22 个相机兼容测试，覆盖 A1、A9、A7、A6xxx、NEX、ZV、FX、RX、SLT、DSLR、未知未来 Sony 型号、Make 字段变体、非 Sony 拒绝和无效 CFA 拒绝。
- 相机型号不参与允许/拒绝判断；实际 2×2 Bayer、尺寸、位深、黑白电平和活动区域决定是否兼容。

## Sony A7C II / RTX 3070 Ti Laptop 实机回归

源文件：`DSC01959.ARW`

- Make / Model：Sony / ILCE-7CM2
- 可见 CFA：7032 × 4688，共 32,966,016 个采样
- CFA pattern：`[0, 1, 3, 2]`，color description：`RGBG`
- 黑电平：`[512, 512, 512, 512]`，白电平：15360
- 自动参数：ISO 400，strength 0.48
- 推理：PMRID 1.0.0 / CUDAExecutionProvider / tile 1024
- GPU 推理耗时：5.665 秒
- CPU fallback：无

0.4.0 输出与已验收的 0.3.0 DNG 逐 CFA 采样比较：

- 尺寸一致：4688 × 7032
- 不同采样数：0
- 最大绝对差值：0

这证明移除 A7C II 型号白名单没有改变既有相机的处理结果。
