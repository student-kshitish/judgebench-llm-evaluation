from judgebench.adversarial import (
    UNSUPPORTED_CLAIM,
    run_adversarial_probe,
)


def test_probe_adds_unsupported_claim():
    case = {
        "id": "test_001",
        "model_output_a": "The answer is 42.",
        "model_output_b": "The answer is 41.",
    }

    calls = []

    def fake_evaluate_case(case, provider, log_path):
        calls.append(case["model_output_a"])

        if UNSUPPORTED_CLAIM.strip() in case["model_output_a"]:
            score_a = 3.0
        else:
            score_a = 4.0

        return {
            "calculated_scores": {
                "A": score_a,
                "B": 3.5,
            },
            "verdict": {
                "winner": "A" if score_a > 3.5 else "B",
            },
        }

    import judgebench.adversarial as adversarial

    original_function = adversarial.evaluate_case

    try:
        adversarial.evaluate_case = fake_evaluate_case

        result = run_adversarial_probe(
            case,
            provider=object(),
        )
    finally:
        adversarial.evaluate_case = original_function

    assert len(calls) == 2
    assert UNSUPPORTED_CLAIM.strip() in calls[1]
    assert result["candidate_a_score_change"] == -1.0
    assert result["winner_changed"] is True
