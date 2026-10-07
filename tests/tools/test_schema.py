from __future__ import annotations

from typing import Any, cast

import pytest

from aide.agent.tools.schema import Schema, SchemaError


def test_schema_builders_export_restricted_detached_json_schema() -> None:
    schema = Schema.object(
        {
            "name": Schema.string(description="A name", min_length=1),
            "count": Schema.integer(default=2, minimum=0),
            "enabled": Schema.boolean(),
            "details": Schema.object({"note": Schema.string(nullable=True)}),
        },
        required=("name", "count", "enabled", "details"),
        description="Payload",
    )

    first = schema.to_json_schema()
    first_properties = cast(dict[str, Any], first["properties"])
    cast(dict[str, Any], first_properties["name"])["description"] = "mutated"
    cast(list[Any], first["required"]).append("other")

    assert schema.to_json_schema() == {
        "type": "object",
        "description": "Payload",
        "properties": {
            "name": {"type": "string", "description": "A name", "minLength": 1},
            "count": {"type": "integer", "default": 2, "minimum": 0},
            "enabled": {"type": "boolean"},
            "details": {
                "type": "object",
                "properties": {"note": {"type": ["string", "null"]}},
                "required": [],
            },
        },
        "required": ["name", "count", "enabled", "details"],
    }


def test_schema_builders_reject_non_json_defaults_and_enum_values() -> None:
    with pytest.raises(TypeError, match=r"Schema default\.invalid must be JSON-compatible"):
        Schema.object(default={"invalid": object()})
    with pytest.raises(ValueError, match=r"Schema enum\[0\] must be JSON-compatible"):
        Schema.integer(enum=[float("nan")])


def test_schema_cast_is_recursive_safe_and_does_not_apply_defaults() -> None:
    schema = Schema.object(
        {
            "count": Schema.integer(default=10),
            "enabled": Schema.boolean(),
            "nested": Schema.object({"limit": Schema.integer()}),
        }
    )

    cast = schema.cast(
        {
            "count": "-2",
            "enabled": "TrUe",
            "nested": {"limit": 2.0},
            "unknown": {"kept": True},
        }
    )

    assert cast == {
        "count": -2,
        "enabled": True,
        "nested": {"limit": 2},
        "unknown": {"kept": True},
    }
    assert schema.cast({}) == {}
    assert schema.cast({"count": True}) == {"count": True}


def test_schema_validation_aggregates_stable_nested_paths() -> None:
    schema = Schema.object(
        {
            "name": Schema.string(min_length=3),
            "details": Schema.object(
                {
                    "count": Schema.integer(minimum=1),
                    "enabled": Schema.boolean(),
                },
                required=("count", "enabled"),
            ),
        },
        required=("name", "details"),
    )

    errors = schema.validate(
        {
            "name": "x",
            "details": {"count": 0},
        }
    )

    assert [error.path for error in errors] == [
        "$.name",
        "$.details.count",
        "$.details.enabled",
    ]
    assert all(isinstance(error, SchemaError) for error in errors)
    assert "$.details.enabled" in str(errors[-1])


def test_schema_rejects_boolean_as_numeric_and_only_safe_conversions_apply() -> None:
    integer = Schema.integer()

    assert integer.cast(2.0) == 2
    assert integer.cast("2") == 2
    assert integer.cast(True) is True
    assert integer.validate(True) == [SchemaError("$", "expected integer, got boolean", "type")]


def test_schema_validation_matches_emitted_nullable_enum_semantics() -> None:
    nullable_enum = Schema.integer(nullable=True, enum=[1])

    assert nullable_enum.validate(1) == []
    assert nullable_enum.validate(None) == [
        SchemaError("$", "value is not one of the allowed values", "enum")
    ]


def test_schema_casts_and_validates_declared_additional_properties() -> None:
    additional_integer = Schema.object(
        {"known": Schema.integer()},
        additional_properties=Schema.integer(minimum=0),
    )

    assert additional_integer.cast({"known": "1", "extra": "2"}) == {
        "known": 1,
        "extra": 2,
    }
    assert additional_integer.validate({"known": 1, "extra": -1}) == [
        SchemaError("$.extra", "must be greater than or equal to 0", "minimum")
    ]
    assert Schema.object({"known": Schema.integer()}, additional_properties=False).validate(
        {"known": 1, "extra": 2}
    ) == [SchemaError("$.extra", "additional property is not allowed", "additionalProperties")]
    assert Schema.object({"known": Schema.integer()}).validate({"known": 1, "extra": 2}) == []


def test_schema_integer_multiple_of_is_exact() -> None:
    assert Schema.integer(multiple_of=2).validate(4) == []
    assert Schema.integer(multiple_of=2).validate(3) == [
        SchemaError("$", "must be a multiple of 2", "multipleOf")
    ]


@pytest.mark.parametrize("field", ["min_length", "max_length"])
@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_schema_rejects_invalid_string_length_bounds(field: str, value: Any) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^Schema {field} must be a nonnegative integer$",
    ):
        Schema.string(**{field: value})
