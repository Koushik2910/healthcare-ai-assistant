"""Adversarial evaluation suite.

Exports the shared test-case definitions used by both the live runner
(``scripts/eval.py``) and the hermetic pytest harness
(``tests/test_eval_harness.py``).
"""
from src.eval.cases import EVAL_CASES, EvalCase

__all__ = ["EVAL_CASES", "EvalCase"]
