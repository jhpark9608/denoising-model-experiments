"""Command-line entry point for the clean-room ONNX demo."""

import argparse
from pathlib import Path

import numpy as np

from onnx_utils import load_session, run_inference
from preprocessing import load_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public toy denoising model.")
    parser.add_argument("--input", required=True, type=Path, help="2D NumPy input file")
    parser.add_argument("--model", required=True, type=Path, help="ONNX model file")
    parser.add_argument("--output", type=Path, default=Path("examples/output.npy"))
    args = parser.parse_args()

    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("output must differ from input")
        input_data = load_input(args.input)
        output = run_inference(load_session(args.model), input_data)[0, 0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved {output.shape} output to {args.output}")


if __name__ == "__main__":
    main()
