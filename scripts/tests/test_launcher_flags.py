#!/usr/bin/env python3
"""
Class A cover for the launcher's backend-flag validation (T12.1d). No accelerator, no
TornadoVM SDK, no model: the validation runs before any of those are touched, which is
the point of testing it here rather than through a launch.

Run with: python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    """Import `llama-tornado` by path -- its name is not a valid module identifier."""
    spec = importlib.util.spec_from_loader(
        "llama_tornado_launcher",
        importlib.machinery.SourceFileLoader(
            "llama_tornado_launcher", str(REPO_ROOT / "llama-tornado")
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


class Args:
    """Only the two fields the validation reads."""

    def __init__(self, backend_override, use_gpu):
        self.backend_override = backend_override
        self.use_gpu = use_gpu


class BackendFlagValidation(unittest.TestCase):

    def _reject(self, backend):
        args = Args(backend, use_gpu=False)
        out = io.StringIO()
        with self.assertRaises(SystemExit) as exit_ctx, redirect_stdout(out):
            launcher.validate_backend_selection(args)
        return exit_ctx.exception.code, out.getvalue()

    def test_every_backend_flag_without_gpu_is_rejected(self):
        for backend in launcher.Backend:
            with self.subTest(backend=backend.value):
                code, message = self._reject(backend)
                self.assertEqual(1, code)
                # The message must name --gpu: that is the fix the user has to apply.
                self.assertIn("--gpu", message)
                self.assertIn(f"--{backend.value}", message)

    def test_the_message_says_the_model_would_run_on_the_cpu(self):
        # The failure this guards against is silent and looks like slow GPU execution,
        # so naming the CPU is the part that makes it recognizable.
        _, message = self._reject(launcher.Backend.CUDA)
        self.assertIn("CPU", message)

    def test_backend_flag_with_gpu_is_accepted(self):
        for backend in launcher.Backend:
            with self.subTest(backend=backend.value):
                launcher.validate_backend_selection(Args(backend, use_gpu=True))

    def test_no_backend_flag_is_accepted_either_way(self):
        launcher.validate_backend_selection(Args(None, use_gpu=False))
        launcher.validate_backend_selection(Args(None, use_gpu=True))


if __name__ == "__main__":
    unittest.main()
