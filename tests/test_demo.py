"""End-to-end checks for the public clean-room demo."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DemoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = ROOT / f".test_tmp_{os.getpid()}"
        self.work.mkdir()
        self.addCleanup(self.cleanup_work)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "build_toy_model.py"),
                "--output-dir",
                str(self.work),
            ],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )
        self.model = self.work / "toy_denoiser.onnx"
        self.input = self.work / "synthetic_input.npy"

    def cleanup_work(self) -> None:
        for path in self.work.iterdir():
            path.unlink()
        self.work.rmdir()

    def run_demo(self, input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "inference_demo.py"),
                "--input",
                str(input_path),
                "--model",
                str(self.model),
                "--output",
                str(output_path),
            ],
            cwd=self.work,
            capture_output=True,
            text=True,
        )

    def test_end_to_end_and_invalid_inputs(self) -> None:
        output_path = self.work / "output.npy"
        result = self.run_demo(self.input, output_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = np.load(self.input, allow_pickle=False)
        output = np.load(output_path, allow_pickle=False)
        self.assertEqual(output.ndim, 2)
        self.assertEqual(output.shape, source.shape)
        self.assertTrue(np.isfinite(output).all())

        for name, invalid in (
            ("non_2d.npy", np.zeros((3, 3, 1), dtype=np.float32)),
            ("undersized.npy", np.zeros((2, 3), dtype=np.float32)),
            ("non_finite.npy", np.full((3, 3), np.nan, dtype=np.float32)),
            ("overflow_on_cast.npy", np.full((3, 3), 1e300, dtype=np.float64)),
            ("complex.npy", np.full((3, 3), 1 + 2j, dtype=np.complex64)),
            ("non_numeric.npy", np.full((3, 3), "x")),
            ("object.npy", np.full((3, 3), object(), dtype=object)),
        ):
            invalid_path = self.work / name
            np.save(invalid_path, invalid)
            result = self.run_demo(invalid_path, self.work / f"{name}.out.npy")
            self.assertNotEqual(result.returncode, 0)

        malformed = self.work / "malformed.onnx"
        malformed.write_bytes(b"not an ONNX model")
        self.model = malformed
        result = self.run_demo(self.input, output_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)

        self.model = self.work / "toy_denoiser.onnx"
        wrong_shape = self.work / "wrong_shape.npy"
        np.save(wrong_shape, np.zeros((3, 3), dtype=np.float32))
        result = self.run_demo(wrong_shape, output_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)

        result = self.run_demo(self.input, self.input)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
