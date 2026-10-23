import json

import pytest
import yaml

from judgebench.suite import load_suite


def test_load_json_suite(tmp_path):
    suite = {
        "cases": [
            {"id": "case_1", "input": "What is 2 + 2?"}
        ]
    }

    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    cases = load_suite(path)

    assert len(cases) == 1
    assert cases[0]["id"] == "case_1"


def test_load_yaml_suite(tmp_path):
    suite = {
        "cases": [
            {"id": "case_1", "input": "What is 2 + 2?"}
        ]
    }

    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(suite),
        encoding="utf-8",
    )

    cases = load_suite(path)

    assert len(cases) == 1
    assert cases[0]["id"] == "case_1"


def test_reject_empty_suite(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"cases": []}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_suite(path)


def test_reject_duplicate_ids(tmp_path):
    suite = {
        "cases": [
            {"id": "duplicate"},
            {"id": "duplicate"},
        ]
    }

    path = tmp_path / "duplicate.yaml"
    path.write_text(
        yaml.safe_dump(suite),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(path)


def test_reject_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse"):
        load_suite(path)


def test_reject_unsupported_extension(tmp_path):
    path = tmp_path / "suite.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        load_suite(path)
