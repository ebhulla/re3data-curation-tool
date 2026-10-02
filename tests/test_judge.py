import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import judge  # noqa: E402


def test_parse_valid():
    text = '{"results": [{"id": 0, "reason": "x", "label": "Duplicate"}, {"id": 1, "reason": "y", "label": "hierarchy"}]}'
    assert [label for label, _ in judge.parse_response(text, 2)] == ["duplicate", "hierarchy"]


def test_parse_tolerates_surrounding_text():
    text = 'Sure! {"results": [{"id": 0, "reason": "", "label": "unrelated"}]} done'
    assert judge.parse_response(text, 1)[0][0] == "unrelated"


def test_parse_missing_and_invalid_become_error():
    text = '{"results": [{"id": 0, "reason": "", "label": "banana"}, {"id": 5, "reason": "", "label": "duplicate"}]}'
    assert [label for label, _ in judge.parse_response(text, 2)] == ["error", "error"]


def test_parse_garbage():
    assert judge.parse_response("not json", 3) == [("error", "")] * 3


def test_merge_swapped():
    assert judge.merge_swapped("duplicate", "duplicate") == "duplicate"
    assert judge.merge_swapped("duplicate", "hierarchy") == "disagree"


def test_build_messages_numbers_pairs():
    msgs = judge.build_messages([("A1", "B1"), ("A2", "B2")])
    assert '0. A: "A1" | B: "B1"' in msgs[1]["content"]
    assert '1. A: "A2" | B: "B2"' in msgs[1]["content"]
