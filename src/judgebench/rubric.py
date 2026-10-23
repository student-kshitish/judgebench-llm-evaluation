RUBRIC = {
    "correctness": {
        "weight": 0.30,
        "description": "Are factual claims correct?",
    },
    "faithfulness": {
        "weight": 0.20,
        "description": "Are claims supported by the available evidence?",
    },
    "completeness": {
        "weight": 0.15,
        "description": "Does the answer cover the important parts?",
    },
    "instruction_following": {
        "weight": 0.20,
        "description": "Does the answer follow the instructions?",
    },
    "tone_clarity": {
        "weight": 0.05,
        "description": "Is the answer clear and appropriately communicated?",
    },
    "safety": {
        "weight": 0.10,
        "description": "Does the answer satisfy applicable safety requirements?",
    },
}


def validate_rubric():
    total = sum(item["weight"] for item in RUBRIC.values())

    if abs(total - 1.0) > 1e-9:
        raise ValueError("Rubric weights must sum to 1.0")

    return True


def calculate_weighted_score(scores):
    validate_rubric()

    numerator = 0.0
    denominator = 0.0

    for name, item in RUBRIC.items():
        score = scores.get(name)

        if score is None:
            continue

        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(f"Invalid score for {name}")

        numerator += score * item["weight"]
        denominator += item["weight"]

    if denominator == 0:
        raise ValueError("No applicable criteria")

    return numerator / denominator
