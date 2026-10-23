from judgebench.bias import (
    make_padded_answer,
    mapped_back_winner,
    reverse_candidate_order,
)


def test_reverse_candidate_order():
    case = {
        "id": "test",
        "model_output_a": "Answer A",
        "model_output_b": "Answer B",
    }

    reversed_case = reverse_candidate_order(case)

    assert reversed_case["model_output_a"] == "Answer B"
    assert reversed_case["model_output_b"] == "Answer A"


def test_map_reversed_winner():
    assert mapped_back_winner("A") == "B"
    assert mapped_back_winner("B") == "A"
    assert mapped_back_winner("tie") == "tie"


def test_padding_does_not_remove_original_answer():
    answer = "The answer is 42."
    padded = make_padded_answer(answer)

    assert padded.startswith(answer)
    assert len(padded) > len(answer)
