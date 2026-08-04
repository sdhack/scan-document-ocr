"""Bounded, persistent GPU OCR pipeline for scanned PDFs."""
import base64
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderTask:
    pdf: str
    page: int


@dataclass
class RenderedPage:
    pdf: str
    page: int
    image: object
    render_seconds: float


@dataclass
class TableTask:
    pdf: str
    page: int
    image: object


def resource_state():
    state = {'gpu_util': 0, 'gpu_memory_pct': 0.0, 'gpu_temp': 0, 'cpu_pct': 0.0}
    try:
        raw = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits',
        ], text=True).strip().split(',')
        state.update(
            gpu_util=int(raw[0]),
            gpu_memory_pct=int(raw[1]) / int(raw[2]) * 100,
            gpu_temp=int(raw[3]),
        )
    except Exception:
        pass
    try:
        import psutil
        state['cpu_pct'] = psutil.cpu_percent(interval=.1)
    except Exception:
        pass
    return state


def load_helpers():
    helper_path = Path(os.environ.get(
        'OCR_ENGINE_SCRIPT',
        str(Path(__file__).resolve().with_name('scan_pdf.py')),
    ))
    spec = importlib.util.spec_from_file_location('scan_helpers', helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def is_full_page_photo(image, text):
    """Separate photo plates from blank or mostly textual scanned pages."""
    import cv2
    import numpy as np

    if len(''.join(text.split())) >= int(os.environ.get(
            'OCR_FULL_PAGE_PHOTO_MAX_CHARS', '20')):
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ignore a narrow scanner margin before measuring tonal coverage.
    height, width = gray.shape
    margin_y = max(1, int(height * 0.03))
    margin_x = max(1, int(width * 0.03))
    core = gray[margin_y:height - margin_y, margin_x:width - margin_x]
    nonwhite_ratio = float(np.mean(core < 242))
    midtone_ratio = float(np.mean((core > 20) & (core < 225)))
    tonal_std = float(np.std(core))
    return (
        nonwhite_ratio >= float(os.environ.get(
            'OCR_FULL_PAGE_PHOTO_NONWHITE', '0.32'))
        and midtone_ratio >= float(os.environ.get(
            'OCR_FULL_PAGE_PHOTO_MIDTONE', '0.16'))
        and tonal_std >= float(os.environ.get(
            'OCR_FULL_PAGE_PHOTO_STD', '30'))
    )


def render_worker(task_queue, ocr_queue, result_queue):
    import cv2
    import fitz
    import numpy as np

    docs = {}
    result_queue.put({'type': 'renderer_ready', 'pid': os.getpid()})
    while True:
        task = task_queue.get()
        if task is None:
            break
        started = time.perf_counter()
        try:
            doc = docs.get(task.pdf)
            if doc is None:
                doc = fitz.open(task.pdf)
                docs[task.pdf] = doc
            pixmap = doc[task.page].get_pixmap(
                matrix=fitz.Matrix(float(os.environ.get('OCR_RENDER_SCALE', '1.25')),
                                   float(os.environ.get('OCR_RENDER_SCALE', '1.25'))),
                alpha=False,
            )
            pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n == 4:
                image = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
            elif pixmap.n == 3:
                image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            ocr_queue.put(RenderedPage(
                task.pdf, task.page, image,
                round(time.perf_counter() - started, 4),
            ))
        except Exception as exc:
            result_queue.put({
                'type': 'error', 'stage': 'render', 'pdf': task.pdf,
                'page': task.page, 'error': f'{type(exc).__name__}: {exc}',
            })


def create_text_engine(helpers):
    from rapidocr import RapidOCR

    requested = os.environ.get('OCR_ENGINE', 'onnxruntime').lower()
    rec_batch = int(os.environ.get('OCR_REC_BATCH', '6'))
    if requested == 'tensorrt':
        try:
            import tensorrt  # noqa: F401
            from rapidocr.utils.typings import EngineType
        except ImportError as exc:
            raise RuntimeError(
                'OCR_ENGINE=tensorrt requested, but TensorRT Python runtime is unavailable'
            ) from exc
        params = {
            'Det.engine_type': EngineType.TENSORRT,
            'Cls.engine_type': EngineType.TENSORRT,
            'Rec.engine_type': EngineType.TENSORRT,
            'EngineConfig.tensorrt.use_fp16': True,
            'EngineConfig.tensorrt.use_int8': False,
            'EngineConfig.tensorrt.cache_dir': os.environ.get(
                'OCR_TRT_CACHE', '/tmp/rapidocr-tensorrt-cache'),
            'Rec.rec_batch_num': rec_batch,
        }
        return RapidOCR(params=params), 'TensorRTExecutionProvider', 'tensorrt-fp16'

    _, actual = helpers.require_cuda_runtime()
    params = {
        'EngineConfig.onnxruntime.use_cuda': True,
        'EngineConfig.onnxruntime.intra_op_num_threads': 2,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        'Rec.rec_batch_num': rec_batch,
    }
    return RapidOCR(params=params), actual[0], 'onnxruntime-cuda'


def text_worker(ocr_queue, table_queue, result_queue):
    import cv2

    helpers = load_helpers()
    try:
        ocr, actual_provider, actual_engine = create_text_engine(helpers)
    except Exception as exc:
        result_queue.put({
            'type': 'worker_start_error', 'worker': 'text', 'pid': os.getpid(),
            'error': f'{type(exc).__name__}: {exc}',
        })
        return
    result_queue.put({
        'type': 'text_worker_ready', 'pid': os.getpid(),
        'actual_provider': actual_provider, 'actual_engine': actual_engine,
        'rec_batch_num': int(os.environ.get('OCR_REC_BATCH', '6')),
    })
    while True:
        item = ocr_queue.get()
        if item is None:
            break
        started = time.perf_counter()
        try:
            result = ocr(item.image)
            text = '\n'.join(getattr(result, 'txts', ()) or ())
            boxes = getattr(result, 'boxes', None)
            full_page_photo = is_full_page_photo(item.image, text)
            if full_page_photo:
                height, width = item.image.shape[:2]
                regions = [(0, 0, width, height)]
            else:
                regions = helpers.figure_regions(
                    item.image, boxes if boxes is not None else ())
            assets = []
            for index, (x, y, width, height) in enumerate(regions, 1):
                ok, encoded = cv2.imencode(
                    '.png', item.image[y:y + height, x:x + width])
                if ok:
                    assets.append((index, encoded.tobytes()))
            is_table = helpers.table_candidate(item.image)
            if is_table:
                table_queue.put(TableTask(item.pdf, item.page, item.image))
            result_queue.put({
                'type': 'page', 'pdf': item.pdf, 'page': item.page,
                'render_seconds': item.render_seconds,
                'ocr_seconds': round(time.perf_counter() - started, 4),
                'text': text, 'assets': assets, 'table_pending': is_table,
                'full_page_photo': full_page_photo,
            })
        except Exception as exc:
            result_queue.put({
                'type': 'error', 'stage': 'ocr', 'pdf': item.pdf,
                'page': item.page, 'error': f'{type(exc).__name__}: {exc}',
            })


def table_worker(table_queue, result_queue):
    from io import StringIO

    import pandas as pd
    from rapid_table import RapidTable, RapidTableInput, EngineType, ModelType

    helpers = load_helpers()
    try:
        _, actual = helpers.require_cuda_runtime()
        table = RapidTable(RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            engine_type=EngineType.ONNXRUNTIME,
            engine_cfg={'use_cuda': True},
            use_ocr=True,
        ))
    except Exception as exc:
        result_queue.put({
            'type': 'worker_start_error', 'worker': 'table', 'pid': os.getpid(),
            'error': f'{type(exc).__name__}: {exc}',
        })
        return
    result_queue.put({
        'type': 'table_worker_ready', 'pid': os.getpid(),
        'actual_provider': actual[0],
    })
    while True:
        task = table_queue.get()
        if task is None:
            break
        started = time.perf_counter()
        try:
            tables = []
            for html in getattr(table(task.image), 'pred_htmls', ()) or ():
                try:
                    tables.extend(
                        frame.to_markdown(index=False)
                        for frame in pd.read_html(StringIO(html))
                    )
                except Exception:
                    tables.append(str(html))
            result_queue.put({
                'type': 'table', 'pdf': task.pdf, 'page': task.page,
                'table_seconds': round(time.perf_counter() - started, 4),
                'tables': tables,
            })
        except Exception as exc:
            result_queue.put({
                'type': 'table_error', 'stage': 'table', 'pdf': task.pdf,
                'page': task.page, 'error': f'{type(exc).__name__}: {exc}',
            })


def start_pipeline(text_count=3, render_count=2, table_count=1):
    render_queue = mp.Queue(maxsize=max(4, render_count * 4))
    ocr_queue = mp.Queue(maxsize=max(4, text_count * 3))
    table_queue = mp.Queue(maxsize=max(2, table_count * 2))
    results = mp.Queue()
    renderers = [
        mp.Process(target=render_worker, args=(render_queue, ocr_queue, results),
                   daemon=True)
        for _ in range(render_count)
    ]
    text_workers = [
        mp.Process(target=text_worker, args=(ocr_queue, table_queue, results),
                   daemon=True)
        for _ in range(text_count)
    ]
    table_workers = [
        mp.Process(target=table_worker, args=(table_queue, results), daemon=True)
        for _ in range(table_count)
    ]
    processes = renderers + text_workers + table_workers
    for process in processes:
        process.start()
    return (render_queue, ocr_queue, table_queue, results,
            renderers, text_workers, table_workers)


def stop_pipeline(render_queue, ocr_queue, table_queue,
                  renderers, text_workers, table_workers):
    for _ in renderers:
        render_queue.put(None)
    for process in renderers:
        process.join(timeout=10)
    for _ in text_workers:
        ocr_queue.put(None)
    for process in text_workers:
        process.join(timeout=10)
    for _ in table_workers:
        table_queue.put(None)
    for process in table_workers:
        process.join(timeout=10)


def decode_arg(value):
    if value.startswith('b64:'):
        return base64.b64decode(value[4:]).decode()
    return value


def write_page(markdown, output, page_number, current, figure_count):
    markdown.write(f'\n\n## Page {page_number + 1}\n\n')
    if current.get('type') == 'error':
        markdown.write('[OCR ERROR]\n')
        return figure_count
    markdown.write(current['text'] + '\n')
    for _, payload in current.get('assets', ()):
        figure_count += 1
        name = f'page-{page_number + 1:04d}-figure-{figure_count:04d}.png'
        (output / 'assets' / name).write_bytes(payload)
        markdown.write(f'\n![Page {page_number + 1} figure](assets/{name})\n')
    for table_markdown in current.get('tables', ()):
        markdown.write('\n### Table\n\n' + table_markdown + '\n')
    return figure_count


def main():
    import fitz

    text_count = int(os.environ.get('OCR_WORKERS', '3'))
    render_count = int(os.environ.get('OCR_RENDER_WORKERS', '2'))
    table_count = int(os.environ.get('OCR_TABLE_WORKERS', '1'))
    pipeline = start_pipeline(text_count, render_count, table_count)
    (render_queue, ocr_queue, table_queue, results,
     renderers, text_workers, table_workers) = pipeline

    expected = {
        'renderer_ready': render_count,
        'text_worker_ready': text_count,
        'table_worker_ready': table_count,
    }
    ready = {key: 0 for key in expected}
    worker_info = []
    deadline = time.time() + 180
    while ready != expected and time.time() < deadline:
        try:
            message = results.get(timeout=1)
        except queue.Empty:
            continue
        if message.get('type') == 'worker_start_error':
            raise RuntimeError(json.dumps(message, ensure_ascii=False))
        if message.get('type') in ready:
            ready[message['type']] += 1
            worker_info.append(message)
            print(json.dumps(message, ensure_ascii=False), flush=True)
    if ready != expected:
        raise RuntimeError(f'pipeline startup incomplete: {ready} expected {expected}')

    if len(sys.argv) <= 2:
        stop_pipeline(render_queue, ocr_queue, table_queue,
                      renderers, text_workers, table_workers)
        print(json.dumps({'status': 'pipeline_ready', 'workers': ready},
                         ensure_ascii=False))
        return

    input_pdf = str(Path(decode_arg(sys.argv[1])).resolve())
    output = Path(decode_arg(sys.argv[2])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    with fitz.open(input_pdf) as doc:
        page_count = len(doc)
    checkpoint = output / 'checkpoint.json'
    resume = {'completed_pages': 0, 'figures': 0, 'full_page_photo_pages': []}
    if checkpoint.exists() and (output / 'book.md').exists():
        resume.update(json.loads(checkpoint.read_text(encoding='utf-8')))
    start_page = min(int(resume['completed_pages']), page_count)

    def feed_pages():
        for page_index in range(start_page, page_count):
            render_queue.put(RenderTask(input_pdf, page_index))

    feeder = threading.Thread(target=feed_pages, name='pdf-page-feeder',
                              daemon=True)
    feeder.start()

    pending = {}
    errors = []
    timings = {'render': [], 'ocr': [], 'table': []}
    figure_count = int(resume['figures'])
    full_page_photo_pages = set(resume.get('full_page_photo_pages', []))
    next_page = start_page
    mode = 'a' if start_page else 'w'
    last_report = time.time()
    with (output / 'book.md').open(mode, encoding='utf-8') as markdown:
        while next_page < page_count:
            message = results.get(timeout=180)
            message_type = message.get('type')
            page_number = message.get('page')
            if message_type == 'page':
                entry = pending.setdefault(page_number, {})
                entry['page'] = message
                if not message['table_pending']:
                    entry['table_done'] = True
                timings['render'].append(message['render_seconds'])
                timings['ocr'].append(message['ocr_seconds'])
                if message.get('full_page_photo'):
                    full_page_photo_pages.add(page_number + 1)
            elif message_type == 'table':
                entry = pending.setdefault(page_number, {})
                entry['tables'] = message['tables']
                entry['table_done'] = True
                timings['table'].append(message['table_seconds'])
            elif message_type == 'table_error':
                errors.append(message)
                entry = pending.setdefault(page_number, {})
                entry['tables'] = []
                entry['table_done'] = True
            elif message_type == 'error':
                errors.append(message)
                pending[page_number] = {
                    'page': message, 'table_done': True, 'tables': [],
                }

            while next_page in pending:
                entry = pending[next_page]
                if 'page' not in entry or not entry.get('table_done'):
                    break
                current = entry['page']
                current['tables'] = entry.get('tables', [])
                figure_count = write_page(
                    markdown, output, next_page, current, figure_count)
                del pending[next_page]
                next_page += 1
                if next_page % 25 == 0 or next_page == page_count:
                    markdown.flush()
                    checkpoint.write_text(json.dumps({
                        'completed_pages': next_page, 'pages': page_count,
                        'errors': len(errors), 'figures': figure_count,
                        'full_page_photo_pages': sorted(full_page_photo_pages),
                    }, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps({
                        'progress': next_page, 'pages': page_count,
                        'errors': len(errors),
                    }, ensure_ascii=False), flush=True)

            if time.time() - last_report >= 60:
                state = resource_state()
                completed = next_page - start_page
                elapsed = max(0.001, time.time() - last_report)
                print(json.dumps({
                    'type': 'resources', 'completed': completed,
                    'pages': page_count - start_page,
                    'pages_per_second_window': round(completed / elapsed, 3),
                    **state,
                }, ensure_ascii=False), flush=True)
                last_report = time.time()
                if state['gpu_temp'] >= 83 or state['gpu_memory_pct'] >= 90:
                    time.sleep(5)

    actual_engines = sorted({
        info.get('actual_engine') for info in worker_info
        if info.get('actual_engine')
    })
    providers = sorted({
        info.get('actual_provider') for info in worker_info
        if info.get('actual_provider')
    })
    summary = {
        'pages': page_count,
        'errors': len(errors),
        'illustrations': figure_count,
        'full_page_photos': len(full_page_photo_pages),
        'full_page_photo_pages': sorted(full_page_photo_pages),
        'requested_provider': (
            'TensorRTExecutionProvider'
            if os.environ.get('OCR_ENGINE', 'onnxruntime').lower() == 'tensorrt'
            else 'CUDAExecutionProvider'
        ),
        'actual_provider': providers[0] if len(providers) == 1 else providers,
        'actual_engine': actual_engines,
        'cuda_available': 'CUDAExecutionProvider' in providers,
        'fallback_to_cpu': any(provider == 'CPUExecutionProvider'
                               for provider in providers),
        'workers': {
            'render': render_count, 'text': text_count, 'table': table_count,
        },
        'rec_batch_num': int(os.environ.get('OCR_REC_BATCH', '6')),
        'timings': {
            key: round(sum(values) / len(values), 4) if values else 0
            for key, values in timings.items()
        },
    }
    (output / 'monitor.json').write_text(json.dumps(
        {'summary': summary, 'workers': worker_info, 'errors': errors},
        ensure_ascii=False, indent=2,
    ), encoding='utf-8')
    feeder.join(timeout=30)
    stop_pipeline(render_queue, ocr_queue, table_queue,
                  renderers, text_workers, table_workers)
    print(json.dumps({'status': 'pipeline_complete', 'summary': summary},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
