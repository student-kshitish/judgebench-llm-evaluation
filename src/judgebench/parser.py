import json
import re

from pydantic import ValidationError

from judgebench.schemas import Verdict


def extract_json(text):
    """
    Extract a JSON object from a raw model response.
    """

    text = text.strip()

    # Remove Markdown fences if present.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```\s*$", "", text)

    # First try parsing the entire response.
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Then search for a JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(text[index:])

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found")


def parse_verdict(text):
    """
    Parse and validate a judge verdict.
    """

    data = extract_json(text)

    try:
        return Verdict.model_validate(data)

    except ValidationError as error:
        raise ValueError(
            f"Judge verdict failed schema validation: {error}"
        ) from error
