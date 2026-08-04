"""Fail fast unless a real ONNX model session loads on CUDA."""
import importlib.metadata as metadata
from pathlib import Path


def main():
    import onnxruntime as ort
    import rapidocr

    installed = {
        dist.metadata.get("Name", "").lower()
        for dist in metadata.distributions()
    }
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        raise SystemExit(
            "ERROR: onnxruntime and onnxruntime-gpu are installed together. "
            "Remove the CPU package."
        )
    model_dir = Path(rapidocr.__file__).resolve().parent / "models"
    models = sorted(model_dir.glob("*.onnx"))
    if not models:
        raise SystemExit(f"ERROR: no ONNX model found in {model_dir}")
    session = ort.InferenceSession(
        str(models[0]),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    actual = session.get_providers()
    if not actual or actual[0] != "CUDAExecutionProvider":
        raise SystemExit(
            "ERROR: CUDA runtime failed to load; "
            f"actual_session_providers={actual}"
        )
    print({"actual_provider": actual[0], "cuda_verified": True})


if __name__ == "__main__":
    main()

