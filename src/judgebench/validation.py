from judgebench.metrics import agreement_rate, cohen_kappa


def validate_judgments(predicted, reference):
    """
    Compare judge predictions with independently reviewed labels.

    Both arguments must contain one categorical label per case.
    """
    if len(predicted) != len(reference):
        raise ValueError("Prediction and reference lengths differ.")

    return {
        "sample_count": len(predicted),
        "exact_agreement": agreement_rate(predicted, reference),
        "cohen_kappa": cohen_kappa(predicted, reference),
        "reference_labels_reviewed": False,
    }
