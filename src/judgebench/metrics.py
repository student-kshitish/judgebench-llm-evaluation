from collections import Counter


def mean(values):
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def pass_rate(scores, threshold=3.5):
    """Fraction of scores meeting the pass threshold."""
    if not scores:
        return 0.0

    return sum(score >= threshold for score in scores) / len(scores)


def winner_counts(winners):
    """Count A wins, B wins, and ties."""
    counts = Counter(winners)

    return {
        "A": counts.get("A", 0),
        "B": counts.get("B", 0),
        "tie": counts.get("tie", 0),
    }


def flip_rate(flips):
    """Fraction of cases whose decision changed."""
    if not flips:
        return 0.0

    return sum(bool(flip) for flip in flips) / len(flips)


def agreement_rate(predicted, reference):
    """Exact agreement between two equally sized label lists."""
    if len(predicted) != len(reference):
        raise ValueError("Label lists must have equal length.")

    if not predicted:
        return 0.0

    return sum(
        prediction == actual
        for prediction, actual in zip(predicted, reference)
    ) / len(predicted)


def cohen_kappa(predicted, reference):
    """
    Calculate unweighted Cohen's kappa for categorical labels.

    Returns None if chance agreement is 1.0, because the
    standard kappa denominator is then zero.
    """
    if len(predicted) != len(reference):
        raise ValueError("Label lists must have equal length.")

    if not predicted:
        return None

    observed = agreement_rate(predicted, reference)

    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)
    n = len(predicted)

    categories = set(predicted) | set(reference)

    expected = sum(
        (predicted_counts[label] / n)
        * (reference_counts[label] / n)
        for label in categories
    )

    denominator = 1.0 - expected

    if abs(denominator) < 1e-12:
        return None

    return (observed - expected) / denominator
