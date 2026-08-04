# Scan Document OCR

一个基础GPU面向批量中文PDF OCR扫描的技能。它使用 WSL/Linux、RapidOCR、ONNX Runtime CUDA 和 RapidTable，把正文、表格与有效插图整理为 Markdown，并明确拒绝静默回退 CPU。

## 主要能力

- 识别中文扫描 PDF 正文并保留原页码。
- 只在表格候选页调用表格模型，输出 Markdown 表格。
- 遮罩文字后提取页内插图，区分普通扫描底图与整页照片。
- 使用持久化 GPU worker、独立渲染队列和独立表格 worker。
- 支持断点续跑、每分钟资源报告和 `monitor.json` 验收。
- 用真实 ONNX Runtime session 验证 CUDA，加载失败时立即停止。

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

## 作为Agent技能安装

把 `skills/scan-document-ocr` 整个目录复制到目标Agent的技能目录，或从GitHub仓库安装该子目录。不同Agent只要支持 `SKILL.md` 技能格式，即可读取工作流；实际OCR仍必须在WSL/Linux GPU环境中运行。

