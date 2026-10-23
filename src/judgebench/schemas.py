from typing import Literal

from pydantic import BaseModel, Field


CRITERIA = [
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone_clarity",
    "safety",
]


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=3)
    evidence: str = Field(min_length=1)


class Verdict(BaseModel):
    case_id: str

    winner: Literal["A", "B", "tie"]

    scores: dict[str, dict[str, CriterionScore]]

    overall_score: dict[str, float]

    decision_rationale: str = Field(min_length=3)

    confidence: float = Field(ge=0.0, le=1.0)

    flags: list[str] = Field(default_factory=list)
