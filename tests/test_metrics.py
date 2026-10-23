import pytest

from judgebench.metrics import (
    agreement_rate,
    flip_rate,
    mean,
    pass_rate,
    winner_counts,
)


def test_mean():
    assert mean([3, 4, 5]) == 4


def test_pass_rate():
    assert pass_rate([4, 5, 2], threshold=3.5) == pytest.approx(
        2 / 3
    )


def test_winner_counts():
    assert winner_counts(["A", "B", "A", "tie"]) == {
        "A": 2,
        "B": 1,
        "tie": 1,
    }


def test_flip_rate():
    assert flip_rate([True, False, True]) == pytest.approx(
        2 / 3
    )


def test_agreement():
    assert agreement_rate(
        ["A", "B", "A"],
        ["A", "A", "A"],
    ) == pytest.approx(2 / 3)


def test_mismatched_agreement_lengths():
    with pytest.raises(ValueError):
        agreement_rate(["A"], ["A", "B"])
