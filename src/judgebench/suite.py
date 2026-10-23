import json
from pathlib import Path

import yaml


def load_suite(path):
    """Load and validate a JSON or YAML evaluation test suite."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported suite format '{suffix}'. "
            "Use .json, .yaml, or .yml."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            if suffix == ".json":
                data = json.load(file)
            else:
                data = yaml.safe_load(file)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(
            f"Could not parse test suite {path}: {error}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"Could not read test suite {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError("Test suite must be a JSON/YAML object.")

    cases = data.get("cases")

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Test suite must contain a nonempty 'cases' list."
        )

    seen_ids = set()

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"Case at index {index} must be an object."
            )

        case_id = case.get("id")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"Case at index {index} needs a nonempty string ID."
            )

        if case_id in seen_ids:
            raise ValueError(f"Duplicate test case ID: {case_id}")

        seen_ids.add(case_id)

    return cases
