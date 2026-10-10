"""Export helpers: replace vs append, CSV header, empty CSV."""

from __future__ import annotations

import json

from rl_crypto.export import write_csv, write_json, write_jsonl


def test_write_jsonl_replaces_by_default(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    write_jsonl(path, [{"a": 2}])
    assert [json.loads(line)["a"] for line in path.read_text().splitlines()] == [2]


def test_write_jsonl_appends(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    write_jsonl(path, [{"a": 2}], append=True)
    assert [json.loads(line)["a"] for line in path.read_text().splitlines()] == [1, 2]


def test_write_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    assert path.exists()


def test_write_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    text = path.read_text().splitlines()
    assert text[0] == "a,b"
    assert text[1] == "1,2"
    assert text[2] == "3,4"


def test_write_csv_empty_is_a_headerless_file(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv(path, [])
    assert path.read_text() == ""


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "doc.json"
    write_json(path, {"schema": "rl-crypto/1", "rows": []})
    assert json.loads(path.read_text())["schema"] == "rl-crypto/1"
