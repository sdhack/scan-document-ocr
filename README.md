# Scan Document OCR Skill

面向中文扫描 PDF 的 GPU OCR 技能：一次完成正文、表格与有效插图提取，输出可检索、可校验、可继续加工的 Markdown，并拒绝 CUDA 失败后静默回退 CPU。

适合扫描书籍、教材、报告、图文混排资料及批量 PDF 数字化，也可作为支持 `SKILL.md` 的 Agent 技能使用。

## 它解决什么问题

| 实际问题 | 解决方式 |
|---|---|
| 指定了 GPU，任务却悄悄在 CPU 上运行 | 创建真实 ONNX 模型 Session 验证 CUDA；运行库加载失败立即停止 |
| 长 PDF 处理慢、GPU 利用率低 | 持久化 GPU worker，并行组织渲染、文字识别、表格识别和结果写入 |
| CPU 长时间满载、设备温度过高 | 限制数学库与 OpenCV 线程，按 CPU、GPU、显存和温度动态调节任务派发 |
| 每页都调用表格模型，浪费大量时间 | 先检测表格候选页，仅对候选页运行独立表格 worker |
| 扫描页背景被当成整页插图 | 遮罩 OCR 文字区域后检测非文字内容，并单独判断整页照片 |
| 中途失败后只能从头开始 | 按页保存检查点，支持断点续跑 |
| 多 worker 合并后页码或图片路径错乱 | 按原页码有序写入，统一生成相对图片引用 |
| 中文文件名、空格和 WSL 引号导致命令失败 | 自动转换 Windows/WSL 路径，并用 Base64 安全传递参数 |
| 完成后无法判断是否漏页、乱码或回退 CPU | 生成 `monitor.json`，记录页数、错误、Provider、耗时与资源统计 |
| 原 PDF 或旧结果容易被覆盖 | 源文件只读，所有产物写入独立输出目录 |

## 工作流程

```text
扫描 PDF
   ↓
独立渲染队列
   ↓
持久化 GPU 文字识别 ──→ 表格候选检测 ──→ 独立表格 worker
   ↓                                      ↓
正文与位置框 ──→ 文字区域遮罩 ──→ 插图/整页照片判断
   ↓
按页有序写入 Markdown + assets + monitor.json + checkpoint.json
```

模型只初始化一次。渲染、推理、表格处理和写入能够重叠执行；有界队列与温度保护用于控制长时间批处理的负载。

## 核心能力

- **真实 GPU 校验**：验证实际推理 Session，而不只检查 Provider 名单。
- **长文档与批量处理**：支持持久化 worker、断点续跑、顺序合并和独立输出目录。
- **混合内容提取**：正文写入 Markdown，表格按候选页识别，插图保存到 `assets/`。
- **减少无效图片**：结合文字遮罩、灰度覆盖和色调变化，区分普通扫描底图、局部插图与整页照片。
- **资源自适应**：监测 CPU、GPU、显存和温度，在压力过高时减缓新任务派发。
- **结果可验收**：通过 `monitor.json` 检查真实 Provider、CPU 回退、错误页、页数、图片和耗时。
- **中文路径友好**：兼容包含中文、空格和括号的 Windows 路径。
- **保护原始资料**：不覆盖源 PDF；Markdown、图片和监测数据分开保存。

## GPU 与 CPU 实测

在同一台机器、同一 OCR 模型、同一 PDF 页面和相同渲染参数下，强制使用 GPU 与 CPU 进行单页对照：

| 测试 | GPU | CPU | GPU 相对速度 |
|---|---:|---:|---:|
| 同页测试 A | 0.221 秒/页 | 1.199 秒/页 | 约 5.4 倍 |
| 同页测试 B | 0.165 秒/页 | 1.073 秒/页 | 约 6.5 倍 |

按测试 A 粗略换算，374 页纯文字 OCR 推理约需：GPU 83 秒，CPU 448 秒（约 7 分 28 秒）。

> 以上数据只比较文字 OCR 推理，不包含 PDF 渲染、表格识别、插图裁剪、磁盘写入和模型初始化。GPU 主要提升速度并减轻 CPU 的推理负担，不会自动提高同一模型的识别准确率。实际性能取决于硬件、页面复杂度、渲染倍率和模型版本。

## 真实案例：中国居民膳食指南（2022 年版）

测试对象为一份约 156.6 MB、共 374 页的中文扫描 PDF，包含正文、表格、食物图片和整页照片。配置为 2 个渲染进程、3 个持久化文字 GPU worker 和 1 个表格 worker。

| 验收项 | 结果 |
|---|---:|
| PDF 页数 | 374 |
| Markdown 页码标题 | 374 |
| OCR 错误 | 0 |
| 实际推理引擎 | ONNX Runtime CUDA |
| 实际 Provider | `CUDAExecutionProvider` |
| CPU 回退 | `false` |
| 提取插图 | 265 张 |
| 缺失图片引用 | 0 |
| 整页照片 | 3 页（第 14、190、302 页） |
| UTF-8 替代字符 `�` | 0 |
| `book.md` 大小 | 约 1.76 MB |
| 完整流程耗时 | 约 3 分 14 秒 |

完成后检查了 Markdown 页码、乱码替代字符和全部图片引用，并人工复核 3 个整页照片候选，均为真实照片页。测试数据仅用于说明识别效果；仓库不提供原 PDF、识别全文或书中插图。

## 快速开始

### 1. 环境要求

- Windows 10/11 + WSL2，或原生 Linux
- Ubuntu 22.04（推荐）
- NVIDIA GPU，建议 8 GB 以上显存
- 支持 WSL CUDA 的 Windows NVIDIA 驱动
- Python 3.10

不要求在 Windows 单独安装 CUDA Toolkit；GPU 运行库安装在 WSL/Linux 的 Python 虚拟环境中。

### 2. 安装

在 WSL/Linux 中进入仓库：

```bash
cd skills/scan-document-ocr
bash scripts/install_wsl.sh
```

默认虚拟环境：

```text
$HOME/.venvs/scan-document-ocr
```

如需自定义：

```bash
OCR_VENV_WSL=/your/venv/path bash scripts/install_wsl.sh
```

### 3. 从 Windows PowerShell 运行

```powershell
powershell -ExecutionPolicy Bypass -File .\skills\scan-document-ocr\scripts\run_scan.ps1 `
  -InputPdf "D:\books\example.pdf" `
  -OutputDir "D:\ocr-output\example" `
  -Workers 3
```

虚拟环境不在默认位置时：

```powershell
$env:OCR_VENV_WSL='/your/venv/path'
```

### 4. 从 Linux/WSL 直接运行

```bash
source "$HOME/.venvs/scan-document-ocr/bin/activate"
bash scripts/run_linux.sh input.pdf output_dir
```

## 输出结构

```text
output_dir/
├── book.md          # 按页整理的正文、表格和图片引用
├── monitor.json     # 运行环境、进度、错误和资源统计
├── checkpoint.json  # 断点续跑状态
└── assets/          # 提取的有效插图
```

`monitor.json` 的关键字段示例：

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

## 完成后如何验收

- `actual_provider` 为 `CUDAExecutionProvider`。
- `fallback_to_cpu` 为 `false`。
- `errors` 为 0，或错误页已经人工复核。
- `book.md` 页码完整，且所有图片引用都存在。
- `full_page_photo_pages` 抽样后确实是照片或图版。
- 文本中不存在异常数量的 UTF-8 替代字符 `�`。

复杂数学公式建议使用专用 LaTeX 公式识别模型进行二次复核。

## 作为 Agent Skill 使用

将 `skills/scan-document-ocr` 整个目录复制到目标 Agent 的技能目录，或从 GitHub 安装该子目录。只要 Agent 支持 `SKILL.md` 技能格式，即可读取并执行工作流。

OCR 运行环境仍须满足本项目要求：在 WSL/Linux 中运行，并通过真实 ONNX Session 验证 CUDA。Agent 位于 Windows 并不意味着 OCR 会在 Windows Python 中执行。

## 适用边界

- 识别准确度受扫描清晰度、倾斜、字体、版式和渲染倍率影响。
- GPU 加速改善的是推理速度；同一模型下不等于识别准确率自动提高。
- 表格、复杂公式、手写内容和严重破损页面仍可能需要人工复核或专用模型。
- 请仅处理你有权使用的文档，并遵守版权、隐私和数据合规要求。

## 许可证

本项目采用 [MIT License](LICENSE)。
