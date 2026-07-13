# Sony ARW 批量降噪与 CFA DNG 导出软件设计

日期：2026-07-13  
状态：已由用户逐段确认

## 1. 目标

开发一款可安装、可分发的开源 Windows 桌面软件，批量读取 Sony A7C II（ILCE-7CM2）ARW，在未去马赛克的 Bayer RAW 域执行降噪，并导出可在 Adobe Lightroom/Camera Raw 和像素蛋糕中继续进行 RAW 域编辑的 DNG。

软件面向一次 100 张以上的常规批处理，但不设置照片数量上限。默认体验为自动降噪，同时提供少量高级参数。首要运行硬件为 RTX 3070 Ti Laptop（按 8 GB 显存约束设计），无合适 GPU 时提供 CPU 回退。

## 2. 非目标

首版不包含账号、云端、在线图库、付费、自动综合修图、插件系统或移动端。首版不承诺 A7C II 以外机型的画质与 DNG 兼容性，也不覆盖 Sony ARQ。

输出不是已经去马赛克的 RGB 图像，也不以 Linear DNG 冒充原始 Bayer RAW。原始 ARW 永远只读，软件不覆盖或修改源文件。

## 3. 技术路线

采用 Python 桌面应用与独立 DNG 辅助核心的混合架构：

- Python、PySide6：GUI、配置、队列和应用编排。
- SQLite：持久化任务、参数、状态和恢复信息。
- LibRaw/rawpy：A7C II ARW 解码、Bayer 数据和元数据读取。
- PyTorch 或 ONNX Runtime CUDA：GPU Bayer 域模型推理。
- 方差稳定化、BM3D/小波：CPU 基准与回退。
- 基于 dnglab/rawler 的开源辅助进程：写入、压缩和校验 CFA DNG。

dnglab/rawler 辅助进程与 Python 主程序通过版本化 JSON 请求和文件路径通信。辅助进程崩溃不得拖垮 GUI 或任务数据库。

不采用纯 Python DNG 写入作为默认路线，因为 PiDNG 一类库主要面向简单 Bayer 数据，尚不能直接视为完整覆盖 A7C II 元数据与 Adobe 兼容性。也不采用 RGB 降噪后输出 Linear DNG 的路线，因为该路线会固化去马赛克、白平衡和部分高光恢复决策。

## 4. 架构与组件

### 4.1 `desktop`

提供 PySide6 主窗口、设置、预览和通知。GUI 只提交命令并订阅状态，不直接解码或处理图像，长任务不得阻塞 UI 线程。

### 4.2 `jobs`

使用 SQLite 保存任务及其状态。支持任意长度队列、暂停、继续、取消、失败重试和程序重启恢复。每张图是独立任务，单张失败不得中止整批。

任务状态至少包含：`queued`、`decoding`、`denoising`、`writing`、`validating`、`completed`、`failed`、`cancelled`。重启时，中断于运行态的任务回到可重试状态，而不是被当作完成。

### 4.3 `raw_pipeline`

负责 ARW 解码、数值归一化、噪声估计、模型选择、分块、推理、拼接、范围恢复及安全检查。内部数据必须携带 CFA 排列、active area、masked area、黑电平、白电平和位深，禁止只传裸数组而丢失语义。

### 4.4 `models`

为普通 GPU 模型、极暗 GPU 模型和 CPU 回退提供相同接口。模型包包含权重、版本、哈希、许可证、训练数据来源摘要和兼容的推理运行时版本。模型加载失败必须产生可操作的错误并允许切换 CPU。

### 4.5 `dng_worker`

接收降噪后的 CFA 数据及源元数据，写入同目录临时文件，重新读取并校验后原子改名。默认输出名为 `<原名>_DN.dng`；冲突时追加序号，不静默覆盖。

### 4.6 数据流

```text
ARW 文件/文件夹
  -> 只读扫描与 SQLite 任务队列
  -> LibRaw 解码 Bayer 与元数据
  -> 自动噪声估计和模式选择
  -> GPU Bayer 降噪
       -> 显存不足：缩小分块重试
       -> 仍失败或无 GPU：CPU 回退
  -> 数值范围、黑电平和坏点安全处理
  -> dng_worker 写入 CFA、元数据和 JPEG 预览
  -> 结构与元数据校验
  -> 原子发布最终 DNG
```

## 5. RAW 降噪算法

### 5.1 预处理

读取可见 Bayer 区域及必要的 masked area，不执行白平衡、去马赛克、色彩空间转换、锐化或风格处理。按每通道黑电平和白电平把有效信号归一化，并将 Bayer 打包为 R、G1、G2、B 四通道。

自动模式结合 ISO、曝光时间、黑区或低信号区域统计，估计 Poisson-Gaussian 噪声参数并生成噪声图。估计器必须给出置信度；低置信度时采用保守强度，避免过度平滑。

### 5.2 GPU 模型

主模型采用 Bayer 域轻量 NAFNet/PMRID 类架构，输入为四通道 Bayer 与噪声图，输出仍为四通道 Bayer。普通模式优先速度和细节；极暗模式使用更大感受野与更保守的纹理恢复，面向欠曝后提亮 3–5 档的素材。

33 MP 图像采用带重叠边界的分块推理，以权重窗融合重叠区域，禁止直接硬拼接。分块大小根据显存探测动态选择；发生 CUDA OOM 时，清理缓存并按更小分块重试一次，然后回退 CPU。

现成 PMRID 权重来自手机传感器，不作为 A7C II 最终质量模型。项目必须提供可复现训练管线，利用许可证兼容的公开 RAW 数据、Sony 低照度数据和物理 Poisson-Gaussian 噪声合成训练通用权重。发布物不得把跨传感器权重描述为 A7C II 专用模型。

### 5.3 CPU 回退

CPU 引擎对四个 Bayer 平面进行方差稳定化、噪声自适应 BM3D/小波处理和颜色一致性约束，再执行逆变换。它既是无 NVIDIA 环境的可用方案，也是 GPU 模型的质量基准和故障回退。

### 5.4 后处理边界

恢复源文件黑电平和白电平编码范围，保持 CFA 排列、active area 与位深一致。允许保守坏点修复，但禁止锐化、局部对比度、色调映射、生成式补纹理和改变画幅。

免校准通用模型不保证从严重欠曝素材中恢复真实细节。默认策略是保留少量自然噪声，而不是产生不存在的纹理。

## 6. DNG 输出

DNG 使用 CFA 图像数据，并保留或正确映射以下信息：

- Make、Model、UniqueCameraModel；
- CFARepeatPatternDim、CFAPattern；
- BitsPerSample、BlackLevel、WhiteLevel；
- ActiveArea、DefaultCropOrigin、DefaultCropSize；
- ColorMatrix、CalibrationIlluminant、AsShotNeutral；
- ISO、快门、光圈、焦距、时间、方向和镜头 EXIF；
- 嵌入式 JPEG 预览与软件/模型版本标识。

不复制与新 CFA 数据矛盾的原厂私有偏移或校验字段。无法可靠解释的 MakerNotes 可保存在独立元数据块或旁车记录中，但不得为了“完整复制”制造损坏 DNG。

每个临时 DNG 必须重新读取并验证尺寸、CFA、位深、黑白电平、active area 和关键元数据。验证失败时删除临时文件，保留源 ARW，并记录结构化错误。

## 7. GUI 与交互

主窗口采用三栏加固定底栏：

- 左栏：添加文件、添加文件夹、最近任务、预设。
- 中栏：缩略图队列，显示等待、处理中、完成、失败，支持筛选、排序、多选和重新处理。
- 右栏：自动、极暗、CPU 兼容三种模式。默认自动；高级区含降噪强度、彩色噪点、细节保护、伪影抑制四个滑块。
- 底栏：输出目录、完成/剩余数量、预计时间、显存占用、开始、暂停、继续、取消。

双击进入同位置对比预览，支持按住切换、1:1 放大和暗部检查。预览只处理当前视口邻近 RAW 分块，参数变化采用防抖更新，并取消已过时的预览任务。

首次启动检测 GPU、驱动、运行时、模型完整性和可写输出位置，不要求用户拍摄校准素材。高级参数可保存为预设。关闭程序时保存未完成队列；再次启动后显示恢复提示，不未经用户确认自动占用 GPU。

## 8. 错误处理与恢复

- ARW 损坏：标记单张失败并显示解码原因。
- 不支持机型或 RAW 模式：拒绝处理并保留任务记录，不猜测 CFA 或黑白电平。
- 显存不足：缩小分块重试，随后 CPU 回退。
- 模型缺失或哈希不符：禁止加载，提示修复或切换 CPU。
- 磁盘空间不足：在写入前估算空间并暂停相关任务。
- DNG 写入或校验失败：删除临时文件，不发布不完整结果。
- 程序或系统中断：依赖 SQLite 事务和临时文件规则恢复，不把半成品标记完成。

日志默认不包含完整个人路径或图像内容。诊断包由用户主动导出，并允许在导出前查看内容。

## 9. 测试与验收

### 9.1 平台与格式

- Windows 10/11 64 位安装与卸载。
- 无需预装 Python。
- A7C II 无压缩、压缩和无损压缩 ARW 样本。
- NVIDIA CUDA 路径、CPU 路径和 GPU 到 CPU 回退路径。

其他 LibRaw 支持机型可显示为实验性导入，但 GUI 必须明确“未验证”，不得暗示同等画质保证。

### 9.2 DNG 兼容性

- 自动结构校验通过。
- 当前版 Lightroom Classic 与 Adobe Camera Raw 可导入、显示正确方向和色彩，并可继续调整白平衡、曝光及相机配置。
- 像素蛋糕完成导入、预览、RAW 域调色和导出人工测试。
- 若像素蛋糕拒绝默认 DNG 编码，保留一个经过验证的兼容编码配置，而不是降级成伪 RAW Linear DNG。

### 9.3 画质

- 在许可证兼容的成对 RAW 测试集上报告 PSNR/SSIM，并保留与 CPU 基准的固定版本对比。
- A7C II 真实样片覆盖常规高 ISO 和欠曝提亮 3–5 档。
- 人工检查暗部色斑、细发、布纹、星点、边缘光晕、分块接缝、坏点和高光裁切。
- 自动模式不得整体弱于 CPU 基准；极暗模式不得以明显虚假纹理换取指标。

### 9.4 稳定性

使用至少 150 张队列测试暂停、恢复、取消、失败重试、程序强制退出、磁盘满和显存不足。失败文件不得阻塞后续任务。测试前后源 ARW 哈希必须一致。

### 9.5 性能目标

以 RTX 3070 Ti Laptop 和 A7C II 33 MP 输入为基准：

- 普通模式平均不超过 45 秒/张；
- 极暗模式平均不超过 120 秒/张；
- 首次局部预览目标在 5 秒内出现。

性能报告必须注明 GPU 功耗模式、驱动、模型版本、分块大小和输入 RAW 类型。

## 10. 打包、开源与发布

使用适合 Windows 的冻结/打包工具生成安装包，并捆绑 Python 运行时、GUI 依赖、DNG 辅助程序和许可证文件。CUDA 推理由应用检测可用 provider；不把完整开发环境打进安装包。

项目采用与所选 GPL 依赖兼容的开源许可证。第三方代码、模型和数据集分别记录许可证与来源；模型权重只有在其训练数据与权重许可证允许再分发时才进入正式安装包。

首版提供未签名安装包，并明确 Windows 信誉提示。代码签名属于后续发布基础设施，不作为首版完成条件。

## 11. 里程碑

1. 建立 ARW 解码、元数据快照和未经降噪的 CFA DNG 往返原型，先验证 Adobe 与像素蛋糕兼容性。
2. 实现 SQLite 队列、CPU 基准引擎和批量 DNG 导出，形成可靠的端到端版本。
3. 实现 GPU 分块推理接口、模型包格式和通用 Bayer 模型训练管线。
4. 集成普通与极暗权重，完成画质、稳定性和性能验收。
5. 完成 PySide6 GUI、恢复体验、安装包和开源发布材料。

第 1 个里程碑是技术闸门：如果无法生成同时被 Adobe 与像素蛋糕接受的 CFA DNG，必须先解决格式兼容性，不继续投入模型训练或完整 GUI。

## 12. 参考依据

- LibRaw 0.22 支持列表明确包含 Sony ILCE-7CM2：https://www.libraw.org/supported-cameras
- Sony 将 A7C II RAW 标为 ARW 4.0：https://www.sony.com/electronics/support/e-mount-body-ilce-7-series/ilce-7cm2/specifications
- Adobe DNG 与 Camera Raw 说明：https://helpx.adobe.com/camera-raw/using/introduction-camera-raw.html
- Adobe 对 CFA RAW、Linear DNG 和非 RAW 的能力区分：https://helpx.adobe.com/camera-raw/using/enhance.html
- 像素蛋糕说明其调色可基于 RAW 域：https://www.pixcakes.com/guide
- dnglab/rawler 为开源 RAW 到 DNG 路线：https://dng.neoanaloglab.com/en/about/
- PMRID 提供 Bayer RAW 降噪代码、模型与 Apache-2.0 许可证：https://github.com/MegEngine/PMRID
- PiDNG 提供 Python Bayer DNG 写入参考：https://github.com/schoolpost/PiDNG
