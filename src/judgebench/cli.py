import argparse
import json
from pathlib import Path
from judgebench.suite import load_suite
from judgebench.bias import (
    run_position_probe,
    run_verbosity_probe,
)
from judgebench.config import load_config
from judgebench.evaluator import evaluate_case
from judgebench.metrics import (
    agreement_rate,
    flip_rate,
    cohen_kappa,
    mean,
    pass_rate,
    winner_counts,
)
from judgebench.providers import OllamaProvider


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_cases(path):
    data = read_json(path)

    if not isinstance(data, dict):
        raise ValueError("Test suite must be a JSON object")

    cases = data.get("cases")

    if not isinstance(cases, list) or not cases:
        raise ValueError("Test suite must contain a nonempty cases list")

    ids = [case.get("id") for case in cases]

    if any(not case_id for case_id in ids):
        raise ValueError("Every test case must have an id")

    if len(ids) != len(set(ids)):
        raise ValueError("Test case IDs must be unique")

    return cases


def make_provider(config, role):
    ollama_config = config["ollama"]
    role_config = config[role]

    if role_config.get("provider") != "ollama":
        raise ValueError("Only the Ollama provider is implemented")

    return OllamaProvider(
        model=role_config["model"],
        base_url=ollama_config["base_url"],
        timeout_seconds=ollama_config["timeout_seconds"],
        temperature=role_config["temperature"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM-as-Judge Evaluation Pipeline"
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
        "--output",
        default=None,
    )

    parser.add_argument(
        "--mode",
        choices=["evaluate", "position", "verbosity", "all"],
        default="all",
    )

    args = parser.parse_args()

    default_outputs = {
        "evaluate": "reports/evaluation_results.json",
        "position": "reports/position_results.json",
        "verbosity": "reports/verbosity_results.json",
        "all": "reports/all_results.json",
    }

    output_path = args.output or default_outputs[args.mode]

    config = load_config(args.config)
    cases = load_suite(args.suite)
    provider = make_provider(config, "judge")

    results = []
    positions = []
    verbosities = []
    errors = []

    log_path = "logs/judge.jsonl"

    for case in cases:
        case_id = case["id"]
        print(f"Evaluating {case_id}...")

        if args.mode in ("evaluate", "all"):
            try:
                results.append(
                    evaluate_case(
                        case,
                        provider,
                        log_path=log_path,
                    )
                )
            except Exception as error:
                errors.append({
                    "case_id": case_id,
                    "mode": "evaluate",
                    "error": str(error),
                })
                print(f"Evaluation failed: {error}")

        if args.mode in ("position", "all"):
            try:
                positions.append(
                    run_position_probe(
                        case,
                        provider,
                        log_path=log_path,
                    )
                )
            except Exception as error:
                errors.append({
                    "case_id": case_id,
                    "mode": "position",
                    "error": str(error),
                })
                print(f"Position probe failed: {error}")

        if args.mode in ("verbosity", "all"):
            try:
                verbosities.append(
                    run_verbosity_probe(
                        case,
                        provider,
                        log_path=log_path,
                    )
                )
            except Exception as error:
                errors.append({
                    "case_id": case_id,
                    "mode": "verbosity",
                    "error": str(error),
                })
                print(f"Verbosity probe failed: {error}")

    # Count completed operations for the selected mode.
    if args.mode == "evaluate":
        successful_count = len(results)
        attempted_count = len(cases)
    elif args.mode == "position":
        successful_count = len(positions)
        attempted_count = len(cases)
    elif args.mode == "verbosity":
        successful_count = len(verbosities)
        attempted_count = len(cases)
    else:
        successful_count = (
            len(results) + len(positions) + len(verbosities)
        )
        attempted_count = len(cases) * 3

    candidate_scores = [
        score
        for result in results
        for score in result["calculated_scores"].values()
    ]

    winners = [
        result["verdict"]["winner"]
        for result in results
    ]

    labels_data = read_json("data/gold_labels.json")

    gold_by_case = {
        item["case_id"]: item["human_winner"]
        for item in labels_data.get("labels", [])
    }

    predicted = []
    reference = []

    for result in results:
        case_id = result["case_id"]

        if case_id in gold_by_case:
            predicted.append(result["verdict"]["winner"])
            reference.append(gold_by_case[case_id])

    score_changes = [
        item["score_change"]
        for item in verbosities
    ]

    report = {
        "model": provider.model,
        "suite": args.suite,
        "mode": args.mode,
        "total_cases": len(cases),
        "attempted_operations": attempted_count,
        "successful_evaluations": successful_count,
        "failed_operations": errors,
        "operation_success_rate": (
            successful_count / attempted_count
            if attempted_count
            else 0.0
        ),
        "mean_candidate_score": mean(candidate_scores),
        "pass_rate": pass_rate(
            candidate_scores,
            config["evaluation"]["pass_threshold"],
        ),
        "winner_counts": winner_counts(winners),
        "position_flip_rate": flip_rate(
            [item["position_flip"] for item in positions]
        ),
        "position_cases_tested": len(positions),
        "position_results": positions,
        "mean_verbosity_score_change": mean(score_changes),
        "verbosity_cases_tested": len(verbosities),
        "verbosity_results": verbosities,
        "gold_label_agreement": (
            agreement_rate(predicted, reference)
            if predicted
            else None
        ),
        "gold_label_cohen_kappa": (
            cohen_kappa(predicted, reference)
            if predicted
            else None
        ),

        "gold_label_count": len(reference),
        "results": results,
        "model_calls": provider.call_count,
        "total_duration_ns": provider.total_duration_ns,
    }

    write_json(output_path, report)

    print("\nExperiment complete.")
    print(json.dumps({
        "mode": args.mode,
        "total_cases": len(cases),
        "successful_evaluations": successful_count,
        "failed_operations": len(errors),
        "operation_success_rate": report["operation_success_rate"],
        "position_flip_rate": report["position_flip_rate"],
        "position_cases_tested": report["position_cases_tested"],
        "mean_verbosity_score_change": (
            report["mean_verbosity_score_change"]
        ),
        "verbosity_cases_tested": report["verbosity_cases_tested"],
        "output": output_path,
    }, indent=2))


if __name__ == "__main__":
    main()
