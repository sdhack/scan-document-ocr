---
name: scan-document-ocr
description: 使用 GPU 识别中文扫描 PDF 中的正文、表格、公式、页内插图和整页照片，并输出 Markdown、图片资源和监测 JSON。适用于扫描书籍、教材、报告、表格、图文混排 PDF，以及需要批量解析、断点续跑、硬件监控、区分扫描底图与照片页或避免 CPU 回退的 OCR 任务。
---

# 扫描文档 GPU 识别

仅在 WSL/Linux 中执行 OCR。Windows 端只使用 `scripts/run_scan.ps1` 转发任务，不得使用 Windows Python 或静默回退 CPU。批量或长任务优先使用 `scripts/persistent_ocr.py`。

## 核心流程

1. 确认输入文件、页数和独立输出目录，不覆盖原 PDF 或已有知识库。
2. 启动前运行真实 ONNX Runtime Session 验证，要求首个 Provider 为 `CUDAExecutionProvider`；不能只检查 `get_available_providers()`。
3. 使用 RapidOCR、ONNX Runtime CUDA 和 PP-OCR 识别正文。
4. 使用独立表格 worker，仅把候选页送入 RapidTable；表格模型使用单页推理，避免阻塞普通正文页。
5. 使用独立输入馈送线程向有界任务队列持续提交页面；主线程只消费结果，避免超大 PDF 因“先塞满队列、后消费结果”发生死锁。
6. 用 OCR 文本框遮罩正文，再检测较大的非文字区域。结合文字量、非白区域、中间灰度占比和色调标准差区分普通扫描底图与整页照片页；保留照片页，不得把普通文字扫描页整页导出为插图。
7. 将正文、Markdown 表格和插图引用按原页码有序写入 `book.md`，将监测信息写入 `monitor.json`。
8. 校验页数、空文本页、乱码、表格、图片路径、错误数和 CPU 回退状态；通过后再生成完成标记并清理临时分片。

## 推荐运行基线

- WSL：Ubuntu 22.04；Python：3.10。
- 只安装 `onnxruntime-gpu`，禁止与 CPU 版 `onnxruntime` 混装。
- 通过 `LD_LIBRARY_PATH` 提供 CUDA 动态库；复杂命令和中文路径使用 stdin、Base64 或直接脚本入口传递，避免 `wsl.exe` 引号拆解。
- 限制 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 和 OpenCV 线程数为 2。
- 16GB 显存默认使用 2 个独立 PDF 渲染进程、3 个持久化文字 GPU worker和 1 个独立表格 worker；模型只初始化一次。
- 使用有界渲染、文字和表格队列形成流水线，限制内存占用并让 PDF 解码与 GPU 推理重叠。
- 对数百页及以上文档保持“馈送线程—有界队列—结果消费”并行；不得在主线程消费结果前同步提交全部页面。
- 普通页采用单页并发。RapidOCR 3.9.2 没有原生多图 batch，不得伪造无收益的页面 batch。
- 默认保持 `OCR_REC_BATCH=6`。只有在同一批代表性页面上确认速度更快且文本一致时才提高；本机测试中 16 和 32 均慢于 6，并出现少量字符差异。
- 页面渲染后直接把 NumPy/BGR 图像传入 OCR，避免 PNG 编码和解码中转。
- 对低文字页执行整页照片分类；默认阈值可通过 `OCR_FULL_PAGE_PHOTO_MAX_CHARS`、`OCR_FULL_PAGE_PHOTO_NONWHITE`、`OCR_FULL_PAGE_PHOTO_MIDTONE` 和 `OCR_FULL_PAGE_PHOTO_STD` 调整。修改阈值后必须在整本书上核对误报与漏报。
- 默认使用 `OCR_ENGINE=onnxruntime`。仅在 TensorRT Python 运行时、动态库和 FP16 引擎构建均验证成功后使用 `OCR_ENGINE=tensorrt`；TensorRT 请求失败时立即停止，不得静默改用 CUDA 或 CPU。
- 显存超过 80%、温度达到 78°C 或 CPU 超过 70% 时降低并发；83°C 时暂停派发新任务。
- 每 60 秒报告书籍数、页数、速度、预计剩余时间、GPU、显存、温度、CPU 和 worker 状态。

## 运行

首次安装先在WSL/Linux中执行：

```bash
bash scripts/install_wsl.sh
```

Windows 转发：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_scan.ps1 -InputPdf <pdf> -OutputDir <dir> -Workers 3
```

WSL/Linux 单本兼容入口：

```bash
bash scripts/run_linux.sh input.pdf output_dir
```

持久化 worker：

```bash
OCR_RENDER_WORKERS=2 OCR_WORKERS=3 OCR_TABLE_WORKERS=1 OCR_REC_BATCH=6 \
python scripts/persistent_ocr.py input.pdf output_dir
```

验证完成后可显式启用 TensorRT FP16：

```bash
OCR_ENGINE=tensorrt OCR_TRT_CACHE=/path/to/cache \
python scripts/persistent_ocr.py input.pdf output_dir
```

## 验收

`monitor.json` 必须记录 `requested_provider`、`actual_provider`、`cuda_available` 和 `fallback_to_cpu`。要求：

- `actual_provider` 为 `CUDAExecutionProvider`
- `fallback_to_cpu` 为 `false`
- `errors` 为 0，或所有错误页均已单独重试并明确记录
- `book.md` 页码完整、中文可读、图片引用有效
- `monitor.json` 记录 `full_page_photos` 和 `full_page_photo_pages`；抽样确认这些页面是真实照片或图版，而不是普通文字扫描底图
- 临时分片可删除，合并后的 Markdown、图片资源和汇总监测文件必须保留

复杂公式需要独立 LaTeX 公式识别模型复核，不得把普通 OCR 输出当作公式真值。
