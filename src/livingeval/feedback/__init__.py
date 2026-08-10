"""The write-back half of the loop.

    from livingeval import feedback

    result = feedback.promote(store, "support", max_cases=400)   # confirmed -> suite
    print(result.summary())

    feedback.export_finetune_data(result.suite, "data/judge.jsonl", fmt="chat")

Two jobs, deliberately separate: confirmed reviews go back into the suite, and
human-confirmed labels go out as training data. Only reviewed cases move in either
direction — training a scorer on the judge's own labels and then measuring it against
the judge is a closed loop that reports success regardless of whether the judge is any
good.
"""

from livingeval.feedback.writer import (
    DEFAULT_INSTRUCTION,
    FORMATS,
    PromotionResult,
    export_finetune_data,
    promote,
)

__all__ = [
    "DEFAULT_INSTRUCTION",
    "FORMATS",
    "PromotionResult",
    "export_finetune_data",
    "promote",
]
