import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from judgebench.config import load_config
from judgebench.evaluator import evaluate_case
from judgebench.providers import OllamaProvider
from judgebench.suite import load_suite


def create_provider(config):
    """Create the configured local judge provider."""
    ollama_config = config["ollama"]
    judge_config = config["judge"]

    if judge_config.get("provider") != "ollama":
        raise ValueError("Only the Ollama provider is supported.")

    return OllamaProvider(
        model=judge_config["model"],
        base_url=ollama_config["base_url"],
        timeout_seconds=ollama_config["timeout_seconds"],
        temperature=judge_config["temperature"],
    )


def mode_or_none(values):
    """Return the most common value, or None for empty input."""
    if not values:
        return None

    counts = Counter(values)
    highest_count = max(counts.values())

    # Resolve ties deterministically.
    return sorted(
        value for value, count in counts.items()
        if count == highest_count
    )[0]


def run_consistency_probe(
    cases,
    provider,
    repeats=3,
    log_path="logs/consistency_judge.jsonl",
):
    """
    Re-evaluate identical cases and measure judgment consistency.

    A failed repetition is recorded and excluded from score metrics.
    """
    if repeats < 2:
        raise ValueError("Use at least 2 repetitions.")

    case_reports = []
    errors = []
    all_winners = []
    all_scores = {"A": [], "B": []}
    exact_winner_rates = []
    modal_winner_rates = []

    for case in cases:
        case_id = case["id"]
        print(f"Testing consistency for {case_id}...")

        run_records = []

        for repetition in range(1, repeats + 1):
            try:
                result = evaluate_case(
                    case,
                    provider,
                    log_path=log_path,
                )

                verdict = result["verdict"]
                scores = result["calculated_scores"]

                run_records.append({
                    "repetition": repetition,
                    "winner": verdict["winner"],
                    "score_a": scores["A"],
                    "score_b": scores["B"],
                })

            except Exception as error:
                errors.append({
                    "case_id": case_id,
                    "repetition": repetition,
                    "error": str(error),
                })

        if not run_records:
            case_reports.append({
                "case_id": case_id,
                "successful_repetitions": 0,
                "requested_repetitions": repeats,
                "winner_consistency": None,
                "score_statistics": None,
                "runs": [],
            })
            continue

        winners = [record["winner"] for record in run_records]
        scores_a = [record["score_a"] for record in run_records]
        scores_b = [record["score_b"] for record in run_records]

        modal_winner = mode_or_none(winners)

        # Exact agreement with the first successful repetition.
        first_winner_rate = sum(
            winner == winners[0] for winner in winners
        ) / len(winners)

        # Fraction matching the most common winner.
        modal_rate = sum(
            winner == modal_winner for winner in winners
        ) / len(winners)

        exact_winner_rates.append(first_winner_rate)
        modal_winner_rates.append(modal_rate)

        all_winners.extend(winners)
        all_scores["A"].extend(scores_a)
        all_scores["B"].extend(scores_b)

        case_reports.append({
            "case_id": case_id,
            "requested_repetitions": repeats,
            "successful_repetitions": len(run_records),
            "failed_repetitions": repeats - len(run_records),
            "winner_counts": dict(Counter(winners)),
            "modal_winner": modal_winner,
            "winner_consistency_vs_first_run": first_winner_rate,
            "winner_consistency_vs_modal_winner": modal_rate,
            "score_statistics": {
                "A_mean": mean(scores_a),
                "A_population_stddev": pstdev(scores_a),
                "B_mean": mean(scores_b),
                "B_population_stddev": pstdev(scores_b),
            },
            "runs": run_records,
        })

    successful_runs = sum(
        report["successful_repetitions"]
        for report in case_reports
    )

    requested_runs = len(cases) * repeats

    report = {
        "experiment": "Repeated-Judgment Consistency",
        "model": provider.model,
        "judge_temperature": provider.temperature,
        "cases_requested": len(cases),
        "repetitions_per_case": repeats,
        "requested_judgments": requested_runs,
        "successful_judgments": successful_runs,
        "failed_judgments": len(errors),
        "judgment_success_rate": (
            successful_runs / requested_runs if requested_runs else 0.0
        ),
        "overall_winner_counts": dict(Counter(all_winners)),
        "mean_winner_consistency_vs_first_run": (
            mean(exact_winner_rates) if exact_winner_rates else None
        ),
        "mean_winner_consistency_vs_modal_winner": (
            mean(modal_winner_rates) if modal_winner_rates else None
        ),
        "score_statistics": {
            candidate: {
                "mean": mean(scores) if scores else None,
                "population_stddev": pstdev(scores) if scores else None,
                "observations": len(scores),
            }
            for candidate, scores in all_scores.items()
        },
        "errors": errors,
        "case_results": case_reports,
        "model_calls": provider.call_count,
        "total_duration_ns": provider.total_duration_ns,
        "interpretation_note": (
            "Consistency measures repeatability, not correctness. "
            "Repeated calls may share deterministic settings or correlated "
            "errors. Results are limited by the size and composition of "
            "the test suite."
        ),
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Measure repeated LLM-judge consistency."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--suite",
        default="data/suite.json",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        default="reports/consistency_results.json",
    )
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    config = load_config(args.config)
    cases = load_suite(args.suite)
    provider = create_provider(config)

    report = run_consistency_probe(
        cases=cases,
        provider=provider,
        repeats=args.repeats,
        log_path="logs/consistency_judge.jsonl",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nConsistency experiment complete.")
    print(json.dumps({
        "cases_requested": report["cases_requested"],
        "repetitions_per_case": report["repetitions_per_case"],
        "successful_judgments": report["successful_judgments"],
        "failed_judgments": report["failed_judgments"],
        "judgment_success_rate": report["judgment_success_rate"],
        "mean_winner_consistency_vs_first_run": (
            report["mean_winner_consistency_vs_first_run"]
        ),
        "mean_winner_consistency_vs_modal_winner": (
            report["mean_winner_consistency_vs_modal_winner"]
        ),
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
