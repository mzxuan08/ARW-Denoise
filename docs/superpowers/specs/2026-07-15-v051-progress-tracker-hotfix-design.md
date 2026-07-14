# ARW Denoise 0.5.1 进度跟踪启动故障修复设计

## 问题

0.5.0 的 GUI 工作线程使用位置参数调用只接受关键字参数的
`ProgressTracker`，导致队列任务在开始解码前以
`ProgressTracker.__init__() takes 1 positional argument ...` 失败。
故障与 ARW 文件、相机型号、GPU 和输出目录无关。

## 修复范围

- GUI 使用 `ProgressTracker(job_id=job.id, on_progress=save_progress)`。
- 提取一个轻量、可直接单元测试的 GUI 进度控制器构造函数，生产工作线程与测试共用同一入口。
- 回归测试必须触发一次进度事件，并验证任务 ID、阶段和回调数据正确。
- 不改变降噪算法、DNG 像素、自动参数、队列数据库结构或用户设置。

## 验证

- 运行完整 pytest 测试集。
- 使用真实 A7C II ARW 运行冻结版 GPU 降噪，验证生成可读取、可编辑 DNG。
- 执行完全离线 GPU/GUI 验包。
- 失败任务由用户点击“重试失败任务”恢复，不自动删除或重建队列。

## 发布

- 版本提升为 0.5.1，保留 0.5.0 Release 作为历史记录。
- 构建 ZIP 与高压缩 7z，复算 SHA256，重新解压验证。
- 提交并推送 `master`，创建 `v0.5.1` GitHub Release，并设为 Latest。
