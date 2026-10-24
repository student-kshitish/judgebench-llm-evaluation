import pytest

from judgebench.metrics import cohen_kappa
from judgebench.validation import validate_judgments


def test_perfect_agreement():
    labels = ["A", "B", "tie", "A"]

    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_for_imperfect_agreement():
    predicted = ["A", "A", "B", "B"]
    reference = ["A", "B", "A", "B"]

    assert cohen_kappa(predicted, reference) == pytest.approx(0.0)


def test_kappa_length_mismatch():
    with pytest.raises(ValueError):
        cohen_kappa(["A"], ["A", "B"])


def test_validation_report():
    report = validate_judgments(
        ["A", "B", "A"],
        ["A", "B", "B"],
    )

    assert report["sample_count"] == 3
    assert report["exact_agreement"] == pytest.approx(2 / 3)
    assert report["reference_labels_reviewed"] is False


def test_empty_labels():
    assert cohen_kappa([], []) is None
