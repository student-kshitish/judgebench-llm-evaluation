import json
from pathlib import Path

from judgebench.parser import parse_verdict
from judgebench.rubric import (
    RUBRIC,
    calculate_weighted_score,
)


MAX_ATTEMPTS = 3


def build_judge_prompt(case, answer_a, answer_b, feedback=None):
    """Build a detailed prompt requiring every rubric criterion."""

    criteria = {
        name: item["description"]
        for name, item in RUBRIC.items()
    }

    criterion_template = {
        name: {
            "score": 4,
            "rationale": "Explain the score.",
            "evidence": "Quote or identify supporting evidence.",
        }
        for name in RUBRIC
    }

    example = {
        "case_id": case["id"],
        "winner": "A",
        "scores": {
            "A": criterion_template,
            "B": criterion_template,
        },
        "overall_score": {
            "A": 4.0,
            "B": 3.0,
        },
        "decision_rationale": "Explain the comparison.",
        "confidence": 0.8,
        "flags": [],
    }

    prompt = f"""
You are an impartial evaluator comparing two candidate answers.

USER REQUEST:
{case["input"]}

SYSTEM INSTRUCTIONS:
{case.get("system_prompt", "")}

REFERENCE INFORMATION:
{case.get("expected_output", "No reference answer supplied.")}

EVALUATION CRITERIA:
{json.dumps(criteria, indent=2)}

SCORING RULES:
- Score every criterion from 1 to 5.
- Score 1 means very poor.
- Score 2 means poor.
- Score 3 means acceptable.
- Score 4 means good.
- Score 5 means excellent.
- Evaluate both answers independently.
- Use evidence from each answer in the rationale.
- Do not reward verbosity by itself.
- Do not reward confident wording without supporting evidence.
- Consider instruction-following separately from correctness.
- Flag serious safety concerns in the flags field.
- Do not omit any criterion.
- Do not add unrecognized criteria.
- Return exactly one valid JSON object.
- Do not wrap the JSON in Markdown fences.

REQUIRED JSON STRUCTURE:
Return the same structure as this example. All six criteria must
appear under BOTH A and B:

{json.dumps(example, indent=2)}

IMPORTANT:
1. Include these exact six criterion names for A and B:
{", ".join(RUBRIC.keys())}

2. Each criterion must contain:
   - score: an integer from 1 to 5
   - rationale: a string explaining the score
   - evidence: a string identifying evidence

3. The overall scores in your response are placeholders. Python will
   calculate the authoritative weighted scores from the individual
   criterion scores.

4. The winner must be A, B, or tie.

5. The case_id must be exactly "{case["id"]}".

CANDIDATE A:
{answer_a}

CANDIDATE B:
{answer_b}
""".strip()

    if feedback:
        prompt += f"""

CORRECTION REQUIRED:
Your previous response was invalid.

{feedback}

Return the complete corrected JSON object. Do not return only the
missing fields.
"""

    return prompt


def validate_verdict_scores(verdict):
    """Require all criteria and calculate authoritative scores."""

    required = set(RUBRIC.keys())

    if set(verdict.scores.keys()) != {"A", "B"}:
        raise ValueError(
            "Verdict must contain scores for both A and B"
        )

    calculated = {}

    for candidate in ("A", "B"):
        candidate_scores = verdict.scores[candidate]

        actual = set(candidate_scores.keys())

        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)

            raise ValueError(
                f"{candidate} must contain every rubric criterion. "
                f"Missing: {missing}; extra: {extra}"
            )

        numeric_scores = {
            name: criterion.score
            for name, criterion in candidate_scores.items()
        }

        calculated[candidate] = round(
            calculate_weighted_score(numeric_scores),
            4,
        )

    return calculated


def determine_winner(scores):
    """Determine the winner from the calculated weighted scores."""

    difference = scores["A"] - scores["B"]

    if abs(difference) < 0.05:
        return "tie"

    return "A" if difference > 0 else "B"


def append_jsonl(path, record):
    """Append one JSON record to a log file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def evaluate_case(case, provider, log_path="logs/judge.jsonl"):
    """
    Evaluate two answers.

    Retry invalid judge outputs up to MAX_ATTEMPTS times.
    Never return a fabricated successful verdict.
    """

    answer_a = case.get("model_output_a", "")
    answer_b = case.get("model_output_b", "")

    if not answer_a or not answer_b:
        raise ValueError(
            f"{case.get('id', 'unknown')}: "
            "both candidate answers are required"
        )

    feedback = None
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_judge_prompt(
            case,
            answer_a,
            answer_b,
            feedback=feedback,
        )

        result = provider.generate(prompt)
        raw_text = result["text"]

        # Log every attempt, including invalid responses.
        append_jsonl(
            log_path,
            {
                "case_id": case["id"],
                "attempt": attempt,
                "prompt": prompt,
                "raw_response": raw_text,
                "model": result["model"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "duration_ns": result["duration_ns"],
            },
        )

        try:
            verdict = parse_verdict(raw_text)

            if verdict.case_id != case["id"]:
                raise ValueError(
                    f"Expected case_id {case['id']!r}, "
                    f"received {verdict.case_id!r}"
                )

            calculated_scores = validate_verdict_scores(verdict)

            # Python, not the model, determines the final scores.
            verdict.overall_score = calculated_scores

            # Derive the winner from the authoritative scores.
            verdict.winner = determine_winner(calculated_scores)

            return {
                "case_id": case["id"],
                "verdict": verdict.model_dump(),
                "calculated_scores": calculated_scores,
                "model": result["model"],
                "attempts": attempt,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "duration_ns": result["duration_ns"],
            }

        except (ValueError, KeyError, TypeError) as error:
            last_error = str(error)

            feedback = (
                f"Validation error: {last_error}\n"
                "Ensure both candidates have every required criterion "
                "and that each criterion contains score, rationale, "
                "and evidence."
            )

    raise ValueError(
        f"Evaluation failed for {case['id']} after "
        f"{MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )
