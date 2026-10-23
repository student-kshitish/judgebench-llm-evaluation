def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def pass_rate(scores, threshold=3.5):
    if not scores:
        return 0.0

    passed = sum(
        score >= threshold
        for score in scores
    )

    return passed / len(scores)


def winner_counts(winners):
    return {
        "A": winners.count("A"),
        "B": winners.count("B"),
        "tie": winners.count("tie"),
    }


def flip_rate(flip_flags):
    if not flip_flags:
        return 0.0

    return sum(bool(flag) for flag in flip_flags) / len(
        flip_flags
    )


def agreement_rate(predictions, references):
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have equal length"
        )

    if not predictions:
        return 0.0

    matches = sum(
        predicted == reference
        for predicted, reference in zip(
            predictions,
            references,
        )
    )

    return matches / len(predictions)
