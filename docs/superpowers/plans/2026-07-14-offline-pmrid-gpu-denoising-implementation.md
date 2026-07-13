# 离线 PMRID GPU RAW 降噪实施计划

日期：2026-07-14  
依据：`docs/superpowers/specs/2026-07-14-offline-pmrid-gpu-denoising-design.md`

## 实施原则

采用测试先行和纵向切片。每项任务先建立可复现的失败测试，再实现最小改动并提交。GPU 不可用的开发环境必须能运行全部非 CUDA 测试；CUDA 集成测试单独标记，并在 RTX 3070 Ti Laptop 上完成真实验收。现有 CPU 与 DNG 路径在 GPU 功能完成前始终保持可用。

## 阶段 1：模型来源、清单与转换

### 任务 1.1 固定 PMRID 上游资产

涉及文件：

- 新增 `vendor/pmrid/UPSTREAM.md`
- 新增 `vendor/pmrid/LICENSE`
- 新增 `scripts/fetch_pmrid_model.ps1`
- 更新 `THIRD_PARTY_NOTICES.md`

步骤：

1. 固定 PMRID 官方仓库提交和 `torch_pretrained.ckp` 的来源 URL。
2. 下载后验证预期 SHA-256，拒绝未知或变更的 checkpoint。
3. 记录 Apache-2.0 许可、上游提交、原始权重哈希和转换产物关系。
4. 测试下载脚本在已有正确文件、损坏文件和网络失败时的行为。

验收：来源、许可和哈希可追溯；损坏权重不会进入构建。

### 任务 1.2 实现模型清单

涉及文件：

- 新增 `src/arw_denoise/model_manifest.py`
- 新增 `tests/test_model_manifest.py`
- 新增 `models/pmrid/manifest.json`

步骤：

1. 定义清单数据结构和严格 JSON 校验。
2. 校验模型名称、版本、SHA-256、输入输出、NCHW 排列、四通道 Bayer、FP16/FP32、数值范围、许可和运行时约束。
3. 实现打包环境与源码环境的模型目录发现。
4. 在加载模型前流式计算哈希。

验收：缺字段、未知精度、错误通道、错误哈希和路径逃逸均被拒绝。

### 任务 1.3 转换 PMRID 为 ONNX

涉及文件：

- 新增 `tools/pmrid_to_onnx.py`
- 新增 `tools/pmrid_net.py`
- 新增 `tests/test_pmrid_export.py`
- 生成 `models/pmrid/pmrid-fp16.onnx`

步骤：

1. 以独立、标注来源的适配文件复现官方 PyTorch 网络定义。
2. 载入官方权重，处理 state-dict 名称映射并严格检查缺失或多余参数。
3. 导出动态高宽、NCHW 四通道 ONNX；模型内部保持与官方预处理一致。
4. 使用固定随机输入比较 PyTorch、ONNX CPU FP32 和最终 FP16 结果。
5. 将导出命令、工具版本和产物哈希写入清单。

验收：FP32 最大误差不超过 `1e-5`，FP16 误差阈值根据固定样本设为 `2e-3`；输出尺寸和数值范围正确。

## 阶段 2：GPU 推理核心

### 任务 2.1 定义统一降噪接口

涉及文件：

- 新增 `src/arw_denoise/engines.py`
- 更新 `src/arw_denoise/denoise.py`
- 新增 `tests/test_engines.py`

步骤：

1. 定义引擎能力、实际 provider、模型信息和单次运行统计。
2. 让现有 Haar CPU 引擎实现统一接口。
3. 保持旧调用路径兼容并增加清晰类型边界。

验收：CPU 基线行为不变；处理器不再依赖具体降噪类。

### 任务 2.2 实现 ONNX Runtime CUDA 会话

涉及文件：

- 新增 `src/arw_denoise/onnx_engine.py`
- 新增 `tests/test_onnx_engine.py`
- 更新 `pyproject.toml`

步骤：

1. 延迟导入 ONNX Runtime，保证无 GPU 环境仍可启动。
2. 从离线运行库目录预加载 CUDA/cuDNN/MSVC DLL。
3. 创建仅优先 CUDA、显式允许 CPU fallback 的 provider 列表，并记录实际节点 provider。
4. 实现 NCHW/HWC、FP16/FP32 转换和输入输出契约检查。
5. 用假会话覆盖 provider 缺失、DLL 加载失败、形状错误、NaN 和运行异常。

验收：不会把 CPU provider 误报为 CUDA；错误包含可复制的修复信息。

### 任务 2.3 GPU 自检与设备信息

涉及文件：

- 新增 `src/arw_denoise/gpu_probe.py`
- 新增 `tests/test_gpu_probe.py`

步骤：

1. 查询 provider、驱动可见设备和模型兼容性。
2. 使用固定小 Bayer 块执行真实推理并同步完成。
3. 返回设备名、provider、模型版本、自检耗时和失败原因。
4. 缓存同一进程内的成功结果，但允许用户手动重新检测。

验收：检测到显卡但推理失败时返回失败；无 CUDA 环境测试可稳定运行。

### 任务 2.4 显存自适应分块与 OOM 回退

涉及文件：

- 更新 `src/arw_denoise/pipeline.py`
- 新增 `src/arw_denoise/tile_scheduler.py`
- 新增 `tests/test_tile_scheduler.py`
- 更新 `tests/test_pipeline.py`

步骤：

1. 把现有重叠融合扩展为可报告进度的调度器。
2. 根据设备显存与模型推荐值选择初始分块。
3. 定义有限缩块序列；仅捕获明确的 CUDA OOM，其他错误直接上报。
4. 每次 OOM 后释放可回收资源并重建必要会话。
5. 全部 GPU 尺寸失败时返回结构化回退原因。

验收：无硬接缝；OOM 顺序和次数可预测；不会无限重试或吞掉非 OOM 错误。

## 阶段 3：自动控制与 RAW 安全后处理

### 任务 3.1 自动噪声配置

涉及文件：

- 新增 `src/arw_denoise/auto_tune.py`
- 新增 `tests/test_auto_tune.py`
- 更新 `src/arw_denoise/domain.py`

步骤：

1. 组合 ISO、曝光时间、四通道盲噪声估计和置信度。
2. 输出模型噪声条件、混合强度及四个高级参数的解析值。
3. 对缺失 EXIF、极端 ISO 和低置信度采用保守边界。
4. 固定自动策略版本并存入任务参数快照。

验收：ISO 或噪声升高时强度总体单调；低置信度不会增强处理。

### 任务 3.2 高级参数与结果混合

涉及文件：

- 新增 `src/arw_denoise/postprocess.py`
- 新增 `tests/test_postprocess.py`

步骤：

1. 实现原始 RAW 与模型输出的稳定混合。
2. 用边缘权重实现细节保护，不执行锐化。
3. 用绿色一致性和跨颜色残差实现保守彩噪控制。
4. 用有限邻域规则抑制异常点和分块伪影。
5. 对所有输出执行有限值和 `[0, 1]` 范围检查。

验收：恒定图不产生纹理；边缘不出现明显光晕；滑块影响单调且不改变整体曝光。

### 任务 3.3 集成 GPU/CPU 处理器

涉及文件：

- 更新 `src/arw_denoise/processor.py`
- 更新 `src/arw_denoise/cli.py`
- 更新 `tests/test_processor.py`
- 新增 `tests/test_gpu_processor.py`

步骤：

1. 增加 `auto`、`gpu`、`cpu` 引擎策略。
2. 默认自动选择通过自检的 GPU；失败时记录原因并回退 CPU。
3. 把实际引擎、模型、provider、分块、耗时写入处理结果。
4. 新增 CLI GPU 自检和单图 GPU 处理入口，便于诊断。

验收：现有 CPU 命令保持可用；GPU 处理完成后继续走同一 DNG 安全发布路径。

## 阶段 4：队列、设置与 GUI

### 任务 4.1 设置持久化和目录策略

涉及文件：

- 新增 `src/arw_denoise/settings.py`
- 更新 `src/arw_denoise/config.py`
- 新增 `tests/test_settings.py`
- 更新 `tests/test_jobs.py`

步骤：

1. 持久化默认导入目录、默认导出目录、导出策略、引擎模式和高级参数。
2. 解析“固定目录”和“源目录/DNG_Denoised”两种输出策略。
3. 加入任务时保存最终输出路径及完整参数快照。
4. 对不存在目录、不可写目录和迁移旧设置做兼容处理。

验收：修改全局设置不改变已有任务；旧版数据库可无损迁移。

### 任务 4.2 队列运行统计

涉及文件：

- 更新 `src/arw_denoise/jobs.py`
- 更新 `src/arw_denoise/domain.py`
- 更新 `tests/test_jobs.py`

步骤：

1. 增加实际引擎、模型版本、provider、分块、耗时和回退原因字段。
2. 提供幂等迁移并保留旧任务。
3. 保持中断恢复、失败重试和状态机约束。

验收：新旧数据库都能启动；运行统计不会破坏任务恢复。

### 任务 4.3 GUI GPU 状态与高级面板

涉及文件：

- 更新 `src/arw_denoise/gui.py`
- 新增 `tests/test_gui_settings.py`

步骤：

1. 显示 GPU 名称、实际 provider、模型版本、分块、显存和耗时。
2. 增加 GPU 自检按钮和可复制错误详情。
3. 实现默认折叠的四个高级滑块及自动模式。
4. 保持耗时工作在线程中执行，避免阻塞 UI。

验收：GPU 自检和批处理期间界面可响应；CPU 回退有清晰提示。

### 任务 4.4 GUI 导入导出目录操作

涉及文件：

- 更新 `src/arw_denoise/gui.py`
- 更新 `tests/test_gui_settings.py`

步骤：

1. 设置页增加默认导入、默认导出和导出策略。
2. 文件选择器从默认或最近导入目录打开。
3. 主界面增加“打开导出目录”。
4. 已完成任务增加“定位文件”，使用参数化系统调用而非拼接 shell 命令。
5. 处理缺失目录、未完成任务和 Explorer 启动失败。

验收：一键打开正确目录；定位操作选中实际 DNG；路径含中文和空格时正常。

## 阶段 5：离线打包与真实验收

### 任务 5.1 固定 GPU 构建环境

涉及文件：

- 新增 `requirements-gpu-lock.txt`
- 更新 `scripts/build_windows.ps1`
- 新增 `scripts/verify_offline_bundle.ps1`
- 更新打包配置文件

步骤：

1. 固定 `onnxruntime-gpu 1.23.2` 及解析出的 CUDA 12/cuDNN 9 wheel 版本和哈希。
2. 把 provider DLL、CUDA/cuDNN DLL、模型、清单和许可收集进发行目录。
3. PyInstaller 显式包含 ONNX Runtime provider 动态库与模型资产。
4. 在隔离 PATH、无 Python、无 CUDA Toolkit、无网络环境执行启动与 GPU 自检。

验收：仅依赖兼容 NVIDIA 驱动；离线包不读取开发机 site-packages 或系统 CUDA。

### 任务 5.2 自动化回归

涉及文件：

- 更新 `README.md`
- 更新 `THIRD_PARTY_NOTICES.md`
- 更新全部相关测试

步骤：

1. 运行全部 CPU 单元测试。
2. 运行 ONNX CPU 转换一致性测试。
3. 在 RTX 机器运行 CUDA 集成测试、OOM 注入和 DNG 回归。
4. 检查源码树无测试输出、未许可权重或临时文件。

验收：所有测试通过，第三方声明完整，CPU 旧功能无回归。

### 任务 5.3 A7C II 整图性能与兼容性

涉及文件：

- 新增 `docs/validation/2026-07-a7c2-pmrid-gpu.md`
- 生成用户验收 DNG 到 `outputs/`

步骤：

1. 使用已提供的 A7C II ARW 执行完整 GPU 处理。
2. 记录 GPU、驱动、运行时、模型、分块、峰值显存和每阶段耗时。
3. 检查 CFA、黑白电平、裁切、色彩矩阵和 DNG 可重新读取。
4. 生成原始、CPU、GPU 的固定暗部裁切供人工比较。
5. 用户在 Adobe/像素蛋糕完成最终可编辑性与画质确认。

验收：完整 DNG 可编辑、无可见分块接缝、无非预期曝光或色偏；普通模式达到或明确报告与 45 秒目标的差距。

### 任务 5.4 150 张队列和发布包

涉及文件：

- 新增 `scripts/stress_queue.py`
- 更新 `README.md`
- 生成最终离线 ZIP 到 `outputs/`

步骤：

1. 用至少 150 项队列测试暂停、恢复、失败隔离、重试和强制退出恢复。
2. 验证源 ARW 哈希不变、输出无覆盖、临时文件正确清理。
3. 对最终 ZIP、EXE、ONNX 模型和许可文件生成 SHA-256。
4. 在干净 Windows 用户目录完成最终冒烟测试。

验收：队列无数量限制，恢复可靠；离线包开箱可用并附完整哈希。

## 提交策略

每个任务至少形成一个独立提交。模型二进制与大型运行库不直接混入普通源码提交；其来源、哈希和可复现获取脚本先提交，经过许可与体积检查后再决定 Git LFS 或仅进入构建产物。任何真实 ARW 不进入版本库。

