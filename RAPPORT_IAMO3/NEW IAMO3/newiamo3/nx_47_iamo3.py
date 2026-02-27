# %% [code]
"""
NX_47_IAMO3 - Kaggle inference server compatible script.
Format aligned with competitor notebook submission method:
- Uses kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer
- Exposes predict(id_, question, answer=None) -> pl.DataFrame({'id', 'answer'})
"""

import gc
import math
import os
import re
from typing import Optional

import polars as pl

import kaggle_evaluation.aimo_3_inference_server


# %% [code]
class NX47AIMO3Solver:
    """Lightweight deterministic fallback solver for AIMO3 runtime stability."""

    _mod_pattern = re.compile(
        r"(?:mod(?:ulo)?\s*(\d+))|(?:\b(\d+)\s*mod(?:ulo)?\b)",
        flags=re.IGNORECASE,
    )
    _number_pattern = re.compile(r"-?\d+")

    @staticmethod
    def _extract_numbers(question: str) -> list[int]:
        return [int(x) for x in NX47AIMO3Solver._number_pattern.findall(question)]

    def solve_problem(self, question: str) -> int:
        text = (question or "").strip().lower()
        nums = self._extract_numbers(text)

        if not nums:
            return 0

        if any(k in text for k in ("sum", "somme", "total", "add", "+")):
            return sum(nums) % 1000

        if any(k in text for k in ("product", "produit", "multiply", "times", "*")):
            out = 1
            for n in nums:
                out *= n
                out %= 1000
            return out

        if any(k in text for k in ("prime", "premier")):
            n = nums[0]
            if n < 2:
                return 0
            root = int(math.isqrt(n))
            for i in range(2, root + 1):
                if n % i == 0:
                    return 0
            return 1

        mod_match = self._mod_pattern.search(text)
        if mod_match and len(nums) >= 2:
            mod_base = int(next(g for g in mod_match.groups() if g))
            if mod_base != 0:
                return nums[0] % mod_base

        # Stable fallback for competition format [0, 999]
        return nums[0] % 1000


solver = NX47AIMO3Solver()


# %% [code]
def predict(
    id_: pl.DataFrame,
    question: pl.DataFrame,
    answer: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Kaggle gateway callback: returns exactly columns ['id', 'answer']."""

    id_value = id_.item(0)
    question_text = question.item(0)

    gc.disable()
    pred = int(solver.solve_problem(question_text)) % 1000
    gc.enable()
    gc.collect()

    return pl.DataFrame({"id": [id_value], "answer": [pred]})


# %% [code]
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",)
    )
