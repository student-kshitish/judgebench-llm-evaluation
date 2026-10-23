import argparse
import json
from pathlib import Path
from statistics import mean
from judgebench.suite import load_suite
from judgebench.config import load_config
from judgebench.evaluator import evaluate_case
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
    """Load and validate a JSON test suite."""
    data = read_json(path)
    cases = data.get("cases") if isinstance(data, dict) else None

    if not isinstance(cases, list) or not cases:
        raise ValueError("The suite must contain a nonempty 'cases' list.")

    ids = [case.get("id") for case in cases]

    if any(not case_id for case_id in ids):
        raise ValueError("Every test case must have an id.")

    if len(ids) != len(set(ids)):
        raise ValueError("Test case IDs must be unique.")

    for case in cases:
        if not any(
            case.get(field)
            for field in ("input", "question", "prompt")
        ):
            raise ValueError(
                f"Case {case['id']} needs an 'input', 'question', or 'prompt'."
            )

    return cases


def make_provider(config, role):
    """Create a provider using the selected role's configuration."""
    ollama_config = config["ollama"]
    role_config = config[role]

    if role_config.get("provider") != "ollama":
        raise ValueError(
            f"Unsupported {role} provider: {role_config.get('provider')}"
        )

    return OllamaProvider(
        model=role_config["model"],
        base_url=ollama_config["base_url"],
        timeout_seconds=ollama_config["timeout_seconds"],
        temperature=role_config["temperature"],
    )


def build_generation_prompt(case):
    """Convert a test case into a prompt for the answer-generating model."""
    question = (
        case.get("input")
        or case.get("question")
        or case.get("prompt")
    )

    system_prompt = case.get(
        "system_prompt",
        "Answer the user's question accurately and follow all instructions.",
    )

    return (
        f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Provide your answer directly."
    )


def generate_answer(provider, prompt):
    """Generate an answer and retain basic usage information."""
    response = provider.generate(prompt)

    if not isinstance(response, dict) or not response.get("text"):
        raise ValueError("The generation provider returned an empty answer.")

    return {
        "text": response["text"],
        "model": response.get("model", provider.model),
        "prompt_tokens": response.get("prompt_tokens", 0),
        "completion_tokens": response.get("completion_tokens", 0),
        "duration_ns": response.get("duration_ns", 0),
    }


def decide_winner(score_a, score_b, margin=0.05):
    """Decide the aggregate winner using a predefined tie margin."""
    if abs(score_a - score_b) < margin:
        return "tie"

    return "A" if score_a > score_b else "B"


def run_comparison(
    cases,
    generator_a,
    generator_b,
    judge,
    log_path,
):
    results = []
    failures = []

    for case in cases:
        case_id = case["id"]
        print(f"Comparing configurations for {case_id}...")

        try:
            prompt = build_generation_prompt(case)

            answer_a = generate_answer(generator_a, prompt)
            answer_b = generate_answer(generator_b, prompt)

            # Give the judge the two freshly generated answers.
            comparison_case = dict(case)
            comparison_case["model_output_a"] = answer_a["text"]
            comparison_case["model_output_b"] = answer_b["text"]

            evaluation = evaluate_case(
                comparison_case,
                judge,
                log_path=log_path,
            )

            scores = evaluation["calculated_scores"]

            results.append({
                "case_id": case_id,
                "configuration_a": {
                    "model": answer_a["model"],
                    "answer": answer_a["text"],
                    "score": scores["A"],
                    "prompt_tokens": answer_a["prompt_tokens"],
                    "completion_tokens": answer_a["completion_tokens"],
                },
                "configuration_b": {
                    "model": answer_b["model"],
                    "answer": answer_b["text"],
                    "score": scores["B"],
                    "prompt_tokens": answer_b["prompt_tokens"],
                    "completion_tokens": answer_b["completion_tokens"],
                },
                "case_winner": evaluation["verdict"]["winner"],
                "judge_rationale": evaluation["verdict"]["decision_rationale"],
            })

        except Exception as error:
            failures.append({
                "case_id": case_id,
                "error": str(error),
            })
            print(f"Failed on {case_id}: {error}")

    scores_a = [
        item["configuration_a"]["score"]
        for item in results
    ]
    scores_b = [
        item["configuration_b"]["score"]
        for item in results
    ]

    mean_a = mean(scores_a) if scores_a else None
    mean_b = mean(scores_b) if scores_b else None

    if mean_a is None or mean_b is None:
        overall_winner = "undetermined"
    else:
        overall_winner = decide_winner(mean_a, mean_b)

    return {
        "experiment": "LLM-as-Judge A/B Configuration Comparison",
        "total_cases": len(cases),
        "successful_cases": len(results),
        "failed_cases": len(failures),
        "failures": failures,
        "configuration_a": {
            "model": generator_a.model,
            "mean_judge_score": mean_a,
            "answers_scored": len(scores_a),
        },
        "configuration_b": {
            "model": generator_b.model,
            "mean_judge_score": mean_b,
            "answers_scored": len(scores_b),
        },
        "overall_winner": overall_winner,
        "tie_margin": 0.05,
        "results": results,
        "generator_a_calls": generator_a.call_count,
        "generator_b_calls": generator_b.call_count,
        "judge_calls": judge.call_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two local LLM generation configurations."
    )
    parser.add_argument(
        "--config-a",
        default="configs/config_a.yaml",
    )
    parser.add_argument(
        "--config-b",
        default="configs/config_b.yaml",
    )
    parser.add_argument(
        "--judge-config",
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--suite",
        default="data/suite.json",
    )
    parser.add_argument(
        "--output",
        default="reports/ab_comparison.json",
    )
    args = parser.parse_args()

    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)
    judge_config = load_config(args.judge_config)

    cases = load_suite(args.suite)

    generator_a = make_provider(config_a, "generator")
    generator_b = make_provider(config_b, "generator")
    judge = make_provider(judge_config, "judge")

    report = run_comparison(
        cases=cases,
        generator_a=generator_a,
        generator_b=generator_b,
        judge=judge,
        log_path="logs/ab_judge.jsonl",
    )

    write_json(args.output, report)

    print("\nA/B experiment complete.")
    print(json.dumps({
        "total_cases": report["total_cases"],
        "successful_cases": report["successful_cases"],
        "failed_cases": report["failed_cases"],
        "configuration_a_mean_score": (
            report["configuration_a"]["mean_judge_score"]
        ),
        "configuration_b_mean_score": (
            report["configuration_b"]["mean_judge_score"]
        ),
        "overall_winner": report["overall_winner"],
        "report": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
