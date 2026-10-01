"""Limits for a short public-problem smoke test."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeConfig:
    problem: str = "cvoyager"
    method: str = "adam"
    seconds: float = 120.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.problem not in {"cvoyager", "uifo"}:
            raise ValueError("problem must be cvoyager or uifo")
        if self.method not in {"adam", "random"}:
            raise ValueError("method must be adam or random")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
            raise ValueError("seconds must be a number")
        if not 1 <= self.seconds <= 300:
            raise ValueError("seconds must be between 1 and 300")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer between 0 and 2147483647")
