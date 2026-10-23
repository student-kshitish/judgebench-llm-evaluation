from copy import deepcopy

from judgebench.evaluator import evaluate_case


def reverse_candidate_order(case):
    """
    Swap answers A and B without changing the underlying answer content.
    """

    swapped = deepcopy(case)

    swapped["model_output_a"], swapped["model_output_b"] = (
        case["model_output_b"],
        case["model_output_a"],
    )

    return swapped


def mapped_back_winner(winner):
    """
    Convert the reversed-order winner to the original answer identity.
    """

    if winner == "A":
        return "B"

    if winner == "B":
        return "A"

    return "tie"


def run_position_probe(case, provider, log_path):
    original = evaluate_case(
        case,
        provider,
        log_path=log_path,
    )

    reversed_case = reverse_candidate_order(case)

    reversed_result = evaluate_case(
        reversed_case,
        provider,
        log_path=log_path,
    )

    first_winner = original["verdict"]["winner"]

    second_winner = mapped_back_winner(
        reversed_result["verdict"]["winner"]
    )

    return {
        "case_id": case["id"],
        "original_winner": first_winner,
        "reversed_winner_mapped_back": second_winner,
        "position_flip": first_winner != second_winner,
    }


def make_padded_answer(answer):
    """
    Add repetitive wording without adding substantive information.
    """

    return (
        answer
        + " This answer provides the requested information. "
        + "The same information is being restated for emphasis. "
        + "No additional factual information is being introduced."
    )


def run_verbosity_probe(case, provider, log_path):
    baseline = evaluate_case(
        case,
        provider,
        log_path=log_path,
    )

    padded_case = deepcopy(case)

    padded_case["model_output_a"] = make_padded_answer(
        case["model_output_a"]
    )

    padded = evaluate_case(
        padded_case,
        provider,
        log_path=log_path,
    )

    baseline_score = baseline["calculated_scores"]["A"]
    padded_score = padded["calculated_scores"]["A"]

    return {
        "case_id": case["id"],
        "baseline_score": baseline_score,
        "padded_score": padded_score,
        "score_change": padded_score - baseline_score,
    }
