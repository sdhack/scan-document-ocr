# Scan Document OCR

一个基于GPU面向批量中文PDF OCR扫描的技能。它使用 WSL/Linux、RapidOCR、ONNX Runtime CUDA 和 RapidTable，把正文、表格与有效插图整理为 Markdown，并明确拒绝静默回退 CPU。

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

