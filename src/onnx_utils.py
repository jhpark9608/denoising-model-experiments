"""Small ONNX Runtime boundary for the public demo."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_session(path: str | Path) -> ort.InferenceSession:
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise ValueError("invalid or unsupported ONNX model") from error
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("model must have exactly one input and one output")
    return session


def run_inference(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    if input_data.dtype != np.float32 or input_data.ndim != 4:
        raise ValueError("input must be an NCHW float32 array")
    try:
        output = session.run(None, {session.get_inputs()[0].name: input_data})[0]
    except Exception as error:
        raise ValueError("ONNX inference failed") from error
    if output.shape != input_data.shape:
        raise ValueError("model output must preserve the input shape")
    if not np.isfinite(output).all():
        raise ValueError("model output must contain only finite values")
    return output
