import json

import pytest

from omni.utils.json import strict_json_loads


def test_strict_json_loads_preserves_unicode_and_nested_values() -> None:
    content = '{"消息":["你好",{"details":{"enabled":true,"count":0}}],"empty":null}'

    assert strict_json_loads(content) == {
        "消息": ["你好", {"details": {"enabled": True, "count": 0}}],
        "empty": None,
    }


@pytest.mark.parametrize(
    "content",
    [
        '{"key":1,"key":2}',
        '{"nested":{"key":1,"key":2}}',
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_strict_json_loads_rejects_duplicates_and_nonstandard_constants(
    content: str,
) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(content)


def test_strict_json_loads_does_not_swallow_decode_errors() -> None:
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads("{")
