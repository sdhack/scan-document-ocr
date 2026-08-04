import json
import base64
import sys
import time
import importlib.metadata as metadata
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
from rapidocr import RapidOCR
from rapid_table import RapidTable, RapidTableInput, EngineType, ModelType


def require_cuda_runtime():
    ort = __import__('onnxruntime')
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        raise RuntimeError(f'CUDAExecutionProvider unavailable; providers={providers}')
    installed = {d.metadata.get('Name', '').lower() for d in metadata.distributions()}
    if 'onnxruntime' in installed and 'onnxruntime-gpu' in installed:
        raise RuntimeError('onnxruntime and onnxruntime-gpu are installed together; remove the CPU package')
    import rapidocr
    model_dir = Path(rapidocr.__file__).resolve().parent / 'models'
    models = sorted(model_dir.glob('*.onnx'))
    if not models:
        raise RuntimeError(f'No OCR ONNX model found in {model_dir}')
    session = ort.InferenceSession(str(models[0]), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    actual = session.get_providers()
    if not actual or actual[0] != 'CUDAExecutionProvider':
        raise RuntimeError(f'CUDA runtime load failed; actual_session_providers={actual}; check LD_LIBRARY_PATH and libcublasLt.so.12')
    return providers, actual


def table_candidate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, img.shape[1] // 24), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, img.shape[0] // 24)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    hs = sum(cv2.boundingRect(c)[2] > img.shape[1] * .25 for c in cv2.findContours(horizontal, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
    vs = sum(cv2.boundingRect(c)[3] > img.shape[0] * .25 for c in cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
    return hs >= 3 and vs >= 2


def figure_regions(img, boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for box in boxes:
        x, y, w, h = cv2.boundingRect(np.asarray(box, np.int32))
        cv2.rectangle(mask, (max(0, x - 12), max(0, y - 12)), (min(gray.shape[1] - 1, x + w + 12), min(gray.shape[0] - 1, y + h + 12)), 255, -1)
    ink = ((gray < 242).astype(np.uint8) * 255)
    ink[mask > 0] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    page_area = gray.shape[0] * gray.shape[1]
    regions = []
    for contour in cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        if not .025 <= (w * h) / page_area <= .65 or w < 120 or h < 80 or y < 40 or y + h > gray.shape[0] - 40:
            continue
        if any(abs(x - a) < 20 and abs(y - b) < 20 for a, b, _, _ in regions):
            continue
        regions.append((x, y, w, h))
    return regions


def main(pdf_arg, output_arg):
    requested, actual = require_cuda_runtime()
    pdf = Path(pdf_arg).resolve()
    output = Path(output_arg).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    ocr = RapidOCR(params={'EngineConfig.onnxruntime.use_cuda': True})
    table = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, engine_type=EngineType.ONNXRUNTIME, engine_cfg={'use_cuda': True}, use_ocr=True))
    doc = fitz.open(pdf)
    started = time.perf_counter()
    stats = {'pages': len(doc), 'scanned_pages': 0, 'table_pages': 0, 'illustration_pages': 0, 'illustrations': 0, 'errors': 0, 'chars': 0, 'requested_provider': 'CUDAExecutionProvider', 'actual_provider': actual[0], 'cuda_available': True, 'fallback_to_cpu': False}
    records = []
    with (output / 'book.md').open('w', encoding='utf-8') as markdown:
        for page_no, page in enumerate(doc, 1):
            page_started = time.perf_counter()
            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                data = pixmap.tobytes('png')
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                result = ocr(data)
                text = '\n'.join(getattr(result, 'txts', ()) or ())
                stats['chars'] += len(text)
                stats['scanned_pages'] += len(text) > 50
                markdown.write(f'\n\n## Page {page_no}\n\n{text}\n')
                boxes = getattr(result, 'boxes', None)
                regions = figure_regions(image, boxes if boxes is not None else ())
                stats['illustration_pages'] += bool(regions)
                for x, y, w, h in regions:
                    name = f'page-{page_no:04d}-figure-{stats["illustrations"] + 1:03d}.png'
                    cv2.imwrite(str(output / 'assets' / name), image[y:y + h, x:x + w])
                    markdown.write(f'\n![Page {page_no} figure](assets/{name})\n')
                    stats['illustrations'] += 1
                if table_candidate(image):
                    stats['table_pages'] += 1
                    for html in getattr(table(image), 'pred_htmls', ()) or ():
                        try:
                            for frame in pd.read_html(html):
                                markdown.write('\n### Table\n\n' + frame.to_markdown(index=False) + '\n')
                        except Exception:
                            markdown.write('\n### Table HTML\n\n' + str(html) + '\n')
                records.append({'page': page_no, 'seconds': round(time.perf_counter() - page_started, 3), 'figures': len(regions)})
            except Exception as exc:
                stats['errors'] += 1
                records.append({'page': page_no, 'error': f'{type(exc).__name__}: {exc}'})
            elapsed = time.perf_counter() - started
            print(f'PROGRESS page={page_no}/{len(doc)} elapsed={elapsed:.1f}s avg={elapsed/page_no:.2f}s/page figures={stats["illustrations"]} tables={stats["table_pages"]} errors={stats["errors"]}', flush=True)
    stats['seconds'] = round(time.perf_counter() - started, 2)
    stats['seconds_per_page'] = round(stats['seconds'] / len(doc), 3)
    (output / 'monitor.json').write_text(json.dumps({'summary': stats, 'pages': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULT ' + json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python scripts/scan_pdf.py input.pdf output_dir')
    a, b = sys.argv[1], sys.argv[2]
    if a.startswith('b64:'): a = base64.b64decode(a[4:]).decode()
    if b.startswith('b64:'): b = base64.b64decode(b[4:]).decode()
    main(a, b)
