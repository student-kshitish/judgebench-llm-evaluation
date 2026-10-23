import pytest

from judgebench.consistency import mode_or_none, run_consistency_probe


def test_mode_returns_most_common_winner():
    assert mode_or_none(["A", "B", "A"]) == "A"


def test_mode_returns_none_for_empty_list():
    assert mode_or_none([]) is None


def test_mode_resolves_ties_deterministically():
    assert mode_or_none(["B", "A"]) == "A"


class FakeProvider:
    model = "fake-model"
    temperature = 0.0
    call_count = 0
    total_duration_ns = 0


def test_consistency_probe_repeats_each_case(monkeypatch, tmp_path):
    import judgebench.consistency as consistency

    calls = []

    def fake_evaluate_case(case, provider, log_path):
        calls.append(case["id"])

        return {
            "verdict": {"winner": "A"},
            "calculated_scores": {"A": 4.5, "B": 3.0},
        }

    monkeypatch.setattr(
        consistency,
        "evaluate_case",
        fake_evaluate_case,
    )

    report = run_consistency_probe(
        cases=[{"id": "case_1"}],
        provider=FakeProvider(),
        repeats=3,
        log_path=str(tmp_path / "judge.jsonl"),
    )

    assert calls == ["case_1", "case_1", "case_1"]
    assert report["successful_judgments"] == 3
    assert report["failed_judgments"] == 0
    assert report["mean_winner_consistency_vs_first_run"] == 1.0
    assert report["case_results"][0]["score_statistics"]["A_mean"] == 4.5


def test_consistency_probe_rejects_single_repeat():
    with pytest.raises(ValueError, match="at least 2"):
        run_consistency_probe(
            cases=[],
            provider=FakeProvider(),
            repeats=1,
        )
