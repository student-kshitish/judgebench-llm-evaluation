import pytest
from pydantic import ValidationError

from judgebench.parser import extract_json
from judgebench.schemas import CriterionScore


def test_extract_plain_json():
    result = extract_json('{"status": "ok"}')
    assert result["status"] == "ok"


def test_extract_fenced_json():
    result = extract_json(
        '```json\n{"status": "ok"}\n```'
    )

    assert result["status"] == "ok"


def test_extract_json_from_text():
    result = extract_json(
        'Result follows: {"status": "ok"}'
    )

    assert result["status"] == "ok"


def test_invalid_json():
    with pytest.raises(ValueError):
        extract_json("No JSON here")


def test_invalid_criterion_score():
    with pytest.raises(ValidationError):
        CriterionScore(
            score=10,
            rationale="Invalid score",
            evidence="Test evidence",
        )
