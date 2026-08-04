# Scan Document OCR Skill

一个基于GPU面向批量中文PDF OCR扫描的技能。它使用 WSL/Linux、RapidOCR、ONNX Runtime CUDA 和 RapidTable，把正文、表格与有效插图整理为 Markdown，并明确拒绝静默回退 CPU。

## 解决的痛点

| 常见痛点 | 本项目的处理方式 |
|---|---|
| 明明指定GPU，实际却在CPU上运行 | 创建真实ONNX模型Session验证CUDA；加载失败立即报错，不允许静默回退 |
| 长PDF识别慢，GPU利用率低 | 持久化GPU worker配合独立渲染、表格和结果消费队列，让各阶段重叠执行 |
| CPU长时间满载、设备温度过高 | 限制数学库和OpenCV线程，并监测CPU、GPU、显存与温度；达到保护阈值后减缓派发 |
| 数百页任务中途失败后只能从头重跑 | 按页写入Markdown并保存`checkpoint.json`，支持从已完成页继续 |
| 每一页都跑表格模型，速度被严重拖慢 | 先做轻量表格候选检测，只把候选页交给独立表格worker |
| 扫描页整张被误判成插图 | 先遮罩OCR文字区域，再检测非文字区域，并单独判断整页照片 |
| 中文文件名、空格路径和WSL引号经常报错 | 统一转换Windows/WSL路径，并用Base64传递输入输出参数 |
| 多worker合并后页码或图片路径错乱 | 按原页号顺序消费结果，写入统一Markdown并生成相对图片引用 |
| 任务显示“完成”，却不知道是否漏页、乱码或回退CPU | 输出`monitor.json`，记录页数、错误、实际Provider、worker、耗时和图片统计 |
| 原始PDF或旧结果容易被覆盖 | 强制使用独立输出目录，源PDF始终保持只读 |

## 核心优点

- **拒绝假GPU**：不仅检查Provider列表，还创建真实ONNX模型Session；CUDA动态库加载失败就立即停止，不会静默回退CPU后继续慢跑。
- **兼顾速度与稳定性**：采用独立PDF渲染队列、持久化文字GPU worker、独立表格worker和有界任务队列，让渲染、识别、表格处理与写入形成流水线。
- **适合长文档和批量任务**：模型只初始化一次，支持数百页PDF、断点续跑和按原页码有序合并，减少中途失败后的重复计算。
- **正文、表格、插图一次输出**：正文写入Markdown；仅在表格候选页调用表格模型；插图保存到`assets/`并自动生成相对引用。
- **减少整页误提取**：先用OCR文本框遮罩正文，再识别非文字区域；结合文字量、灰度覆盖和色调变化，区分普通扫描底图与整页照片。
- **对中文路径更友好**：Windows到WSL通过路径转换和Base64参数传递，降低空格、中文文件名和命令行引号造成的失败概率。
- **资源使用可观察**：每分钟报告页数、速度、GPU利用率、显存、温度和CPU占用；高温或显存压力下暂停派发新任务。
- **结果可以验收**：`monitor.json`记录实际Provider、CPU回退、错误页、worker、耗时、插图和整页照片页，便于确认任务是否真的完成。
- **保护原始资料**：使用独立输出目录，不覆盖源PDF；Markdown、图片资源和监测结果分开保存，便于后续知识库、检索或人工校对。

## 环境要求

- Windows 10/11 + WSL2，或原生 Linux。
- Ubuntu 22.04（推荐）。
- NVIDIA GPU，建议 8GB 以上显存。
- Windows NVIDIA 驱动支持 WSL CUDA。
- Python 3.10。

不要求在 Windows 安装 CUDA Toolkit。GPU运行时库安装在 WSL Python 虚拟环境中。

## 安装

在 WSL 中执行：

```bash
cd skills/scan-document-ocr
bash scripts/install_wsl.sh
```

安装脚本默认创建：

```text
$HOME/.venvs/scan-document-ocr
```

如需自定义路径：

```bash
OCR_VENV_WSL=/your/venv/path bash scripts/install_wsl.sh
```

## Windows调用

在 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\skills\scan-document-ocr\scripts\run_scan.ps1 `
  -InputPdf "D:\books\example.pdf" `
  -OutputDir "D:\ocr-output\example" `
  -Workers 3
```

如果虚拟环境不是默认位置：

```powershell
$env:OCR_VENV_WSL='/your/venv/path'
```

## Linux/WSL直接调用

```bash
source "$HOME/.venvs/scan-document-ocr/bin/activate"
bash scripts/run_linux.sh input.pdf output_dir
```

## 输出

```text
output_dir/
├── book.md
├── monitor.json
├── checkpoint.json
└── assets/
```

完成后确认：

- `actual_provider` 是 `CUDAExecutionProvider`；
- `fallback_to_cpu` 是 `false`；
- `errors` 为0，或错误页已经复核；
- `book.md` 页码完整，图片引用存在；
- `full_page_photo_pages` 抽样后确实是照片或图版。

复杂数学公式仍需专用 LaTeX 公式识别模型复核。

## 真实识别案例：《中国居民膳食指南（2022年版）》

测试对象是一份约 156.6 MB、共 374 页的中文扫描 PDF，包含正文、表格、食物图片和整页照片。Windows 端使用 PowerShell 转发到 WSL，配置 2 个渲染进程、3 个持久化文字 GPU worker 和 1 个表格 worker：

```powershell
powershell -ExecutionPolicy Bypass -File .\skills\scan-document-ocr\scripts\run_scan.ps1 `
  -InputPdf "D:\books\中国居民膳食指南（2022年版）.pdf" `
  -OutputDir "D:\ocr-output\中国居民膳食指南（2022年版）" `
  -Workers 3
```

实际结果：

| 验收项 | 结果 |
|---|---:|
| PDF页数 | 374 |
| Markdown页码标题 | 374 |
| OCR错误 | 0 |
| 实际推理引擎 | ONNX Runtime CUDA |
| 实际Provider | `CUDAExecutionProvider` |
| CPU回退 | `false` |
| 提取插图 | 265张 |
| 缺失图片引用 | 0 |
| 整页照片 | 3页（第14、190、302页） |
| UTF-8替代字符 `�` | 0 |
| `book.md` 大小 | 约1.76 MB |
| 总耗时 | 约3分14秒 |

`monitor.json` 中的关键结果示例：

```json
{
  "pages": 374,
  "errors": 0,
  "illustrations": 265,
  "full_page_photos": 3,
  "full_page_photo_pages": [14, 190, 302],
  "requested_provider": "CUDAExecutionProvider",
  "actual_provider": "CUDAExecutionProvider",
  "actual_engine": ["onnxruntime-cuda"],
  "cuda_available": true,
  "fallback_to_cpu": false,
  "workers": {
    "render": 2,
    "text": 3,
    "table": 1
  },
  "rec_batch_num": 6
}
```

完成后额外检查了Markdown页码、乱码替代字符和全部图片引用，并人工抽查3个整页照片候选；三页均为真实照片页，不是普通文字扫描底图。测试数据只用于说明识别效果，仓库不提供原PDF、识别全文或书中插图。

## 作为Agent技能安装

把 `skills/scan-document-ocr` 整个目录复制到目标Agent的技能目录，或从GitHub仓库安装该子目录。不同Agent只要支持 `SKILL.md` 技能格式，即可读取工作流；实际OCR仍必须在WSL/Linux GPU环境中运行。

