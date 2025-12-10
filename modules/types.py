from dataclasses import dataclass
from typing import Tuple


@dataclass
class PredictionResult:
    label: str
    score: float


def human_readable_shape(shape: Tuple) -> str:
    try:
        return "x".join("?" if s is None else str(s) for s in shape)
    except Exception:
        return str(shape)
