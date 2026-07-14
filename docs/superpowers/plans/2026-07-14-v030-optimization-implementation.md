# ARW Denoise 0.3.0 优化实施计划

日期：2026-07-14  
规格：`docs/superpowers/specs/2026-07-14-v030-batch-performance-preview-packaging-design.md`  
分支：`feat/v030-optimization`

## 实施原则

- 每项任务先建立失败测试，再实现最小改动，通过后独立提交。
- 现有 PMRID、RAW 安全后处理、高光保护和 DNG 发布路径在全过程中必须保持可用。
- 不以改变模型、降低数值精度或缩小图像换取性能。
- 每个阶段结束都运行完整单元测试；真实 CUDA/A7C II 测试在关键纵向切片后运行。
- 只在四个阶段和发布门槛全部通过后生成 0.3.0 交付物。

## 阶段 1：立即取消、进度与 ETA

### 任务 1.1 定义统一任务控制契约

涉及：

- 新增 `src/arw_denoise/task_control.py`
- 新增 `tests/test_task_control.py`

步骤：

1. 为取消令牌、取消异常、阶段事件和单调进度快照写失败测试。
2. 实现线程安全的 `CancellationToken`，支持无锁读取和幂等取消。
3. 实现阶段权重与总进度聚合，拒绝倒退、超界和未知阶段。
4. 保持组件与 Qt、SQLite 和推理引擎无关。

验收：任一检查点取消都抛出专用非错误结果；进度严格单调且可序列化。

### 任务 1.2 让分块推理和后处理可取消

涉及：

- 更新 `src/arw_denoise/pipeline.py`
- 更新 `src/arw_denoise/tile_scheduler.py`
- 更新 `src/arw_denoise/postprocess.py`
- 更新 `tests/test_pipeline.py`
- 更新 `tests/test_tile_scheduler.py`
- 更新 `tests/test_postprocess.py`

步骤：

1. 在首个 tile、相邻 tile、OOM 重试前和后处理通道之间注入取消，建立精确的调用次数测试。
2. 将 tile 进度转换为统一阶段事件，保留现有回调兼容层。
3. 使取消异常跳过 OOM 尺寸重试和 CPU 回退。
4. 将后处理拆为可检查的有限阶段，不改变数值公式。

验收：取消不启动下一个 tile；OOM+取消最终为取消；未取消输出与基线一致。

### 任务 1.3 定向终止 dnglab 子进程与清理临时文件

涉及：

- 更新 `src/arw_denoise/dnglab.py`
- 更新 `tests/test_dnglab.py`

步骤：

1. 用可轮询的 `Popen` 适配器替换不可中断的 `subprocess.run`，先以假进程测试正常、超时、取消和终止失败。
2. 仅终止当前 dnglab 进程，不使用模糊进程名杀死。
3. 在 `finally` 中删除当前 UUID 临时 DNG，并保留无覆盖原子发布。
4. 将取消令牌传入元数据读取、转换、分析和写入过程。

验收：写入中取消时不产生最终 DNG，不留临时文件，不影响其他子进程。

### 任务 1.4 集成处理器、队列状态与 ETA

涉及：

- 新增 `src/arw_denoise/eta.py`
- 更新 `src/arw_denoise/processor.py`
- 更新 `src/arw_denoise/jobs.py`
- 新增 `tests/test_eta.py`
- 更新 `tests/test_smart_processor.py`
- 更新 `tests/test_jobs.py`

步骤：

1. 为阶段进度、取消结果和旧数据库幂等迁移写测试。
2. 增加阶段、阶段/总进度、耗时、RAM/VRAM 峰值和取消时间列。
3. 处理器接受统一控制器，将解码、降噪、后处理、写入和校验映射到阶段事件。
4. 实现按引擎和像素数归一化的滚动中位 ETA，覆盖样本不足和异常值。

验收：0.2.0 数据库无损启动；取消保留 `cancelled`；ETA 不在低信心时显示伪精确数字。

### 任务 1.5 GUI 即时取消、队列进度与清理

涉及：

- 更新 `src/arw_denoise/gui.py`
- 更新 `src/arw_denoise/gui_helpers.py`
- 更新 `tests/test_gui_settings.py`

步骤：

1. 将工作线程中的延迟取消替换为当前控制器立即取消。
2. 增加当前文件进度条、整队列进度、阶段、已用时间和 ETA。
3. 增加“清理已完成”，仅删除历史记录，不删除 DNG。
4. 使重试取消任务与重试失败任务分开，界面文字明确。

验收：GUI 不阻塞；进度不倒退；取消当前照片不继续队列；清理记录不碰文件。

## 阶段 2：性能与内存

### 任务 2.1 建立可复现的分阶段基准

涉及：

- 新增 `src/arw_denoise/metrics.py`
- 新增 `scripts/benchmark_a7c2.py`
- 新增 `tests/test_metrics.py`
- 更新 `src/arw_denoise/domain.py`
- 更新 `pyproject.toml`

步骤：

1. 引入固定版本 `psutil`，为进程 RSS 峰值、阶段耗时和 `nvidia-smi` VRAM 采样写可注入测试。
2. 在独立监控线程中有界轮询，完成后必须终止并回收。
3. 基准脚本运行预热与多次样本，输出 JSON 和 Markdown，不修改源 ARW。
4. 先记录 0.2.0 两张 A7C II 的解码、推理、后处理、写入、校验和总耗时与峰值。

验收：重复运行样本结构稳定，监控失败不影响正式处理，基准保留完整环境信息。

### 任务 2.2 减少整图浮点副本

涉及：

- 更新 `src/arw_denoise/pipeline.py`
- 更新 `src/arw_denoise/postprocess.py`
- 更新 `src/arw_denoise/processor.py`
- 更新 `tests/test_pipeline.py`
- 更新 `tests/test_postprocess.py`
- 更新 `tests/test_smart_processor.py`

步骤：

1. 为预分配输出、就地剪裁/缩放和缓冲区别名安全性写测试。
2. `tiled_inference` 支持调用方提供输出/权重缓冲，不必内部再分配同尺寸数组。
3. 后处理在语义安全处使用 `out=` 和就地操作，阶段结束立即释放中间缓冲。
4. 在解包前删除不再使用的模型输出，避免原始 packed、模型 packed、后处理 packed 和恢复 CFA 长时间同时存活。
5. 对随机、高反差、饱和和两张真实 A7C II 与 0.2.0 输出对比。

验收：每样本差不超过 1 个 RAW 码值；高光保护精确；RAM 峰值目标降低 25%。

### 任务 2.3 优化会话、分块与 I/O 路径

涉及：

- 更新 `src/arw_denoise/onnx_engine.py`
- 更新 `src/arw_denoise/tile_scheduler.py`
- 更新 `src/arw_denoise/dnglab.py`
- 更新 `tests/test_onnx_engine.py`
- 更新 `tests/test_tile_scheduler.py`
- 更新 `tests/test_dnglab.py`

步骤：

1. 测量会话初始化、HWC/NCHW 转换、tile padding 和 dnglab 子进程各自成本。
2. 保留单个经验证 CUDA 会话跨照片复用，对 OOM 才定向重建。
3. 仅在基准证明有收益时引入 I/O binding 或 pinned buffer；无收益的复杂度不合入。
4. 优化并发只用于可并行的预取/预览，同一 GPU 不并行跑两张整图。

验收：总耗时中位数目标降低 15%；显存无累积增长；150 张队列会话复用稳定。

## 阶段 3：源/降噪对比预览

### 任务 3.1 实现确定性预览显影与缓存

涉及：

- 新增 `src/arw_denoise/preview.py`
- 新增 `src/arw_denoise/preview_cache.py`
- 新增 `tests/test_preview.py`
- 新增 `tests/test_preview_cache.py`
- 更新 `src/arw_denoise/config.py`

步骤：

1. 为同一显影配置、旋转/尺寸一致、缓存命中/失效和损坏重建写测试。
2. 用 rawpy 对 ARW 和 DNG 应用固定的快速 sRGB 8-bit 显影，两边共用白平衡与曝光配置。
3. 将适应窗口和 100% 所需数据分层缓存，使首屏不等待不必要的全分辨率转换。
4. 缓存原子写入 AppData，以文件身份和显影版本命名，提供有界 LRU 清理。

验收：冷启动适应预览目标 5 秒内，缓存命中 1 秒内；失败不改变 DNG 任务状态。

### 任务 3.2 实现同步对比视图数学

涉及：

- 新增 `src/arw_denoise/compare_view.py`
- 新增 `tests/test_compare_view.py`

步骤：

1. 将适应窗口、100%、缩放中心、平移边界和分割位置建模为不依赖 Qt 的纯函数，先写可逆性与边界测试。
2. 保证左右图像共用唯一视口变换，不出现漂移或不同步像素。
3. 分割线仅改变剪裁区，不触发图像重缩放。

验收：同步变换在任意窗口尺寸和 DPI 缩放下保持一致。

### 任务 3.3 集成独立预览窗口

涉及：

- 更新 `src/arw_denoise/gui.py`
- 新增 `src/arw_denoise/preview_window.py`
- 更新 `tests/test_gui_settings.py`

步骤：

1. 只为已完成且源/输出均存在的任务启用“对比预览”。
2. 在专用 QThread 中生成/读取预览，用任务身份丢弃过时回调。
3. 窗口提供适应、100%、同步缩放/拖动、左右分割线和重新生成。
4. 错误细节可复制，关闭窗口可取消未完成预览工作而不影响处理队列。

验收：主界面持续响应；快速切换任务不显示错图；预览完全离线。

## 阶段 4：发布精简与双格式交付

### 任务 4.1 生成受控发布清单

涉及：

- 新增 `scripts/release_manifest.py`
- 新增 `tests/test_release_manifest.py`
- 更新 `scripts/build_windows.ps1`
- 更新 `scripts/verify_offline_bundle.ps1`

步骤：

1. 为相对路径规范化、哈希、重名和额外/缺失文件写测试。
2. 对构建目录生成稳定排序的 JSON 与 `SHA256SUMS.txt`。
3. 隔离验证先校验清单，再执行 CUDA/GUI 冒烟测试。

验收：任一文件被替换、删除或额外添加都导致验证失败。

### 任务 4.2 删除未使用运行资产

涉及：

- 新增 `scripts/audit_bundle.py`
- 新增 `tests/test_audit_bundle.py`
- 更新 `scripts/build_windows.ps1`
- 更新 `requirements-gpu-lock.txt`
- 更新 `THIRD_PARTY_NOTICES.md`

步骤：

1. 记录 0.2.0 各顶层目录、Qt 插件和 CUDA DLL 体积。
2. 排除未导入 Qt 模块/插件、ONNX 工具和 TensorRT provider，保留 Windows 平台、图像解码与必需样式资产。
3. 每删除一类 CUDA/cuDNN/cuFFT DLL 都在清空 PATH 的解压副本上重跑真实 PMRID CUDA 推理；失败即恢复。
4. 生成可追溯的体积报告，禁止依赖开发机环境的“假精简”。

验收：ZIP 不大于 0.2.0 的 1.84 GB；离线 CUDA 与 GUI 均通过。

### 任务 4.3 生成兼容 ZIP 与固实高压缩包

涉及：

- 新增 `scripts/package_release.ps1`
- 更新 `scripts/verify_offline_bundle.ps1`
- 更新 `README.md`

步骤：

1. 固定一个可再分发、带许可文件和 SHA-256 的 7-Zip 命令行版本，不调用未知系统压缩器。
2. 从同一已验证的发布目录生成 ZIP 与固实 LZMA2 包。
3. 将两个包分别解压到新目录，比对发布清单并各自执行离线自检。
4. 记录 ZIP、固实包、EXE、ONNX 和验收 DNG 哈希。

验收：固实包目标不超过 1.50 GB 且至少比 ZIP 小 15%；若熵限制阻止绝对目标，仍必须满足相对目标并提供体积报告。

## 阶段 5：最终回归与 0.3.0 发布

### 任务 5.1 自动化回归

1. 运行全部 pytest，包括取消、ETA、内存、预览、数据迁移和发布测试。
2. 运行 CUDA 自检、OOM 注入、取消竞争和 150 项持久化队列。
3. 检查工作树无真实 ARW、缓存、未许可二进制或临时输出。

### 任务 5.2 A7C II 画质、性能与取消验收

1. 用两张真实 A7C II ARW 运行预热和多次 0.3.0 处理，记录中位耗时与峰值。
2. 与 0.2.0 验收 DNG 比较像素、CFA、黑白电平、裁切、色彩矩阵和高光保护。
3. 在解码、tile、后处理和写入阶段分别取消，测量延迟并检查临时文件。
4. 生成新验收 DNG 与预览对比供 Adobe/像素蛋糕人工确认。

### 任务 5.3 离线交付

1. 构建 0.3.0，生成发布清单、ZIP 和固实包。
2. 对两个包运行 CRC/哈希验证、新目录解压、隔离 CUDA 自检和 GUI 冒烟测试。
3. 将交付物、验收 DNG、`SHA256SUMS-0.3.0.txt`、性能报告和发布说明写入 `outputs/`。
4. 使用开发分支收尾流程处理合并/保留选择。

## 提交与检查点

每个编号任务至少一个独立提交。阶段 1、2、3、4 结束后分别运行完整测试并检查实际输出；阶段 2 和 5 必须使用 RTX 3070 Ti Laptop 与真实 A7C II ARW。任何阶段如果导致 DNG 安全回归，立即停止后续优化并修复根因。

