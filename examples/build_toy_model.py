"""Build the independent synthetic input and averaging-filter ONNX graph."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public toy ONNX assets.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    synthetic = (np.sin(3 * np.pi * x) + np.cos(2 * np.pi * y)).astype(np.float32)
    np.save(output_dir / "synthetic_input.npy", synthetic)

    weights = numpy_helper.from_array(np.full((1, 1, 3, 3), 1 / 9, dtype=np.float32), "weights")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weights"], ["output"], pads=[1, 1, 1, 1])],
        "toy_average_filter",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 32, 32])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_dir / "toy_denoiser.onnx")
    print(f"Saved synthetic input and toy model to {output_dir}")


if __name__ == "__main__":
    main()
