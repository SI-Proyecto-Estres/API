from pathlib import Path
from typing import Any, Optional
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


class UniversalModelAdapter:
    def __init__(
        self,
        model: Any,
        backend: str,
        source_path: Path,
        ort_input_name: Optional[str] = None,
    ):
        self.model = model
        self.backend = backend
        self.source_path = source_path
        self.ort_input_name = ort_input_name

    def predict_proba(self, vector: np.ndarray, num_classes: int = 3) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        if self.backend == "keras":
            arr = np.asarray(self.model.predict(x, verbose=0), dtype=np.float32)[0]
            if np.all(arr >= 0) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
                return arr / np.sum(arr)
            return _softmax(arr)

        if self.backend == "tflite":
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            expected_dtype = input_details[0]["dtype"]
            self.model.set_tensor(input_details[0]["index"], x.astype(expected_dtype))
            self.model.invoke()
            arr = np.asarray(self.model.get_tensor(output_details[0]["index"])[0], dtype=np.float32)
            if np.all(arr >= 0) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
                return arr / np.sum(arr)
            return _softmax(arr)

        if self.backend == "sklearn":
            if hasattr(self.model, "predict_proba"):
                probs = np.asarray(self.model.predict_proba(x), dtype=np.float32)[0]
                return probs / np.sum(probs)
            pred = int(np.asarray(self.model.predict(x))[0])
            probs = np.zeros(num_classes, dtype=np.float32)
            probs[np.clip(pred, 0, num_classes - 1)] = 1.0
            return probs

        if self.backend == "torch":
            import torch

            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.from_numpy(x))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                arr = out.detach().cpu().numpy()[0]
            if np.all(arr >= 0) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
                return arr.astype(np.float32)
            return _softmax(arr)

        if self.backend == "onnx":
            outputs = self.model.run(None, {self.ort_input_name: x})
            arr = np.asarray(outputs[0], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            if np.all(arr >= 0) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
                return arr / np.sum(arr)
            return _softmax(arr)

        raise ValueError(f"Backend no soportado: {self.backend}")


def load_model_from_dir(models_dir: str = "modelos") -> Optional[UniversalModelAdapter]:
    base = Path(models_dir)
    if not base.exists():
        return None

    preferred_keras = base / "modelo_estres.keras"
    keras_files = [preferred_keras] if preferred_keras.exists() else sorted(
        list(base.glob("*.keras")) + list(base.glob("*.h5"))
    )

    preferred_tflite = base / "modelo_estres.tflite"
    tflite_files = [preferred_tflite] if preferred_tflite.exists() else sorted(base.glob("*.tflite"))

    candidates = [
        ("keras", keras_files),
        ("tflite", tflite_files),
        ("joblib", sorted(list(base.glob("*.joblib")) + list(base.glob("*.pkl")))),
        ("torch", sorted(list(base.glob("*.pt")) + list(base.glob("*.pth")))),
        ("onnx", sorted(base.glob("*.onnx"))),
    ]

    for kind, files in candidates:
        if not files:
            continue
        path = files[0]

        if kind == "keras":
            try:
                import tensorflow as tf
            except (ImportError, ModuleNotFoundError):
                continue

            try:
                model = tf.keras.models.load_model(path)
                return UniversalModelAdapter(model, backend="keras", source_path=path)
            except Exception:
                continue

        if kind == "tflite":
            try:
                import tensorflow as tf
            except (ImportError, ModuleNotFoundError):
                continue

            try:
                interpreter = tf.lite.Interpreter(model_path=str(path))
                interpreter.allocate_tensors()
                return UniversalModelAdapter(interpreter, backend="tflite", source_path=path)
            except Exception:
                continue

        if kind == "joblib":
            import joblib

            return UniversalModelAdapter(joblib.load(path), backend="sklearn", source_path=path)

        if kind == "torch":
            import torch

            try:
                model = torch.jit.load(path, map_location="cpu")
            except Exception:
                model = torch.load(path, map_location="cpu", weights_only=False)
            return UniversalModelAdapter(model, backend="torch", source_path=path)

        if kind == "onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            return UniversalModelAdapter(
                session, backend="onnx", source_path=path, ort_input_name=input_name
            )

    return None