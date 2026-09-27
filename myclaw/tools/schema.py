"""Small, provider-neutral schemas for Tool declarations and arguments."""

from __future__ import annotations

import builtins
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from json import dumps
from typing import Any, Literal, TypeGuard, cast

from myclaw.utils.validation import require_nonnegative_int

type SchemaKind = Literal["string", "integer", "boolean", "object"]

_MISSING = object()
_INTEGER_TEXT = re.compile(r"^[+-]?[0-9]+$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ToolParam:
    """Scalar declaration metadata used by annotation-derived Tool schemas."""

    description: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    format: str | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("Tool parameter description must be a string")
        for string_name, string_value in (("format", self.format), ("pattern", self.pattern)):
            if string_value is not None and not isinstance(string_value, str):
                raise TypeError(f"Tool parameter {string_name} must be a string")
        for length_name, length_value in (
            ("min_length", self.min_length),
            ("max_length", self.max_length),
        ):
            if length_value is not None and (
                isinstance(length_value, bool)
                or not isinstance(length_value, int)
                or length_value < 0
            ):
                raise ValueError(f"Tool parameter {length_name} must be a nonnegative integer")
        for number_name, number_value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if number_value is not None and (
                isinstance(number_value, bool) or not isinstance(number_value, int)
            ):
                raise TypeError(f"Tool parameter {number_name} must be an integer")
        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError("Tool parameter min_length cannot exceed max_length")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Tool parameter minimum cannot exceed maximum")
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as error:
                raise ValueError(
                    "Tool parameter pattern must be a valid regular expression"
                ) from error


@dataclass(frozen=True, slots=True)
class SchemaError:
    """One stable, path-addressed schema validation error."""

    path: str
    message: str
    keyword: str | None = None

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class Schema:
    """An immutable restricted recursive JSON Schema fragment.

    The supported vocabulary is deliberately small.  A Schema only describes
    JSON values; it does not apply defaults or decide what to do with unknown
    object members during casting.
    """

    __slots__ = (
        "_additional_properties",
        "_default",
        "_description",
        "_enum",
        "_exclusive_maximum",
        "_exclusive_minimum",
        "_format",
        "_kind",
        "_max_length",
        "_maximum",
        "_min_length",
        "_minimum",
        "_multiple_of",
        "_nullable",
        "_pattern",
        "_properties",
        "_required",
    )

    def __init__(
        self,
        kind: SchemaKind,
        *,
        nullable: bool = False,
        description: str | None = None,
        default: builtins.object = _MISSING,
        enum: Sequence[builtins.object] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        format: str | None = None,
        minimum: int | float | None = None,
        maximum: int | float | None = None,
        exclusive_minimum: int | float | None = None,
        exclusive_maximum: int | float | None = None,
        multiple_of: int | float | None = None,
        properties: Mapping[str, Schema] | None = None,
        required: Iterable[str] = (),
        additional_properties: bool | Schema | None = None,
    ) -> None:
        if kind not in {"string", "integer", "boolean", "object"}:
            raise ValueError(f"unsupported Schema type: {kind}")
        if not isinstance(nullable, bool):
            raise TypeError("Schema nullable must be a boolean")
        if description is not None and not isinstance(description, str):
            raise TypeError("Schema description must be a string")
        if enum is not None:
            if isinstance(enum, (str, bytes)) or not isinstance(enum, Sequence):
                raise TypeError("Schema enum must be a sequence")
            enum_values = tuple(
                deepcopy(_require_json_value(value, f"enum[{index}]"))
                for index, value in enumerate(enum)
            )
        else:
            enum_values = None
        if min_length is not None:
            require_nonnegative_int(min_length, field="Schema min_length")
        if max_length is not None:
            require_nonnegative_int(max_length, field="Schema max_length")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("Schema min_length cannot exceed max_length")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise TypeError("Schema pattern must be a string")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError("Schema pattern must be a valid regular expression") from error
        if format is not None and not isinstance(format, str):
            raise TypeError("Schema format must be a string")
        for name, value in (
            ("minimum", minimum),
            ("maximum", maximum),
            ("exclusive_minimum", exclusive_minimum),
            ("exclusive_maximum", exclusive_maximum),
            ("multiple_of", multiple_of),
        ):
            if value is not None:
                _require_finite_number(value, name)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Schema minimum cannot exceed maximum")
        if multiple_of is not None and multiple_of <= 0:
            raise ValueError("Schema multiple_of must be greater than zero")
        if properties is not None:
            for name, property_schema in properties.items():
                if not isinstance(name, str) or not name:
                    raise TypeError("Schema object property names must be non-empty strings")
                if not isinstance(property_schema, Schema):
                    raise TypeError(f"Schema for object property {name} must be a Schema")
            property_values = tuple(
                (name, property_schema) for name, property_schema in properties.items()
            )
        else:
            property_values = ()
        required_values = tuple(required)
        if any(not isinstance(name, str) or not name for name in required_values):
            raise TypeError("Schema required names must be non-empty strings")
        if len(set(required_values)) != len(required_values):
            raise ValueError("Schema required names must be unique")
        property_names = {name for name, _ in property_values}
        missing = sorted(set(required_values) - property_names)
        if missing:
            raise ValueError(f"Schema required properties are not declared: {', '.join(missing)}")
        if additional_properties is not None and not isinstance(
            additional_properties, (bool, Schema)
        ):
            raise TypeError("Schema additional_properties must be a boolean or Schema")
        if kind == "object" and properties is None:
            property_values = ()

        self._kind = kind
        self._nullable = nullable
        self._description = description
        self._default = (
            _MISSING if default is _MISSING else deepcopy(_require_json_value(default, "default"))
        )
        self._enum = enum_values
        self._min_length = min_length
        self._max_length = max_length
        self._pattern = pattern
        self._format = format
        self._minimum = minimum
        self._maximum = maximum
        self._exclusive_minimum = exclusive_minimum
        self._exclusive_maximum = exclusive_maximum
        self._multiple_of = multiple_of
        self._properties = property_values
        self._required = required_values
        self._additional_properties = additional_properties

        if default is not _MISSING:
            errors = self.validate(default)
            if errors:
                raise TypeError(f"Schema default is invalid: {errors[0]}")

    @classmethod
    def string(
        cls,
        *,
        description: str | None = None,
        default: builtins.object = _MISSING,
        nullable: bool = False,
        enum: Sequence[builtins.object] | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        format: str | None = None,
    ) -> Schema:
        return cls(
            "string",
            description=description,
            default=default,
            nullable=nullable,
            enum=enum,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            format=format,
        )

    @classmethod
    def integer(
        cls,
        *,
        description: str | None = None,
        default: builtins.object = _MISSING,
        nullable: bool = False,
        enum: Sequence[builtins.object] | None = None,
        minimum: int | float | None = None,
        maximum: int | float | None = None,
        exclusive_minimum: int | float | None = None,
        exclusive_maximum: int | float | None = None,
        multiple_of: int | float | None = None,
    ) -> Schema:
        return cls(
            "integer",
            description=description,
            default=default,
            nullable=nullable,
            enum=enum,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
            exclusive_maximum=exclusive_maximum,
            multiple_of=multiple_of,
        )

    @classmethod
    def boolean(
        cls,
        *,
        description: str | None = None,
        default: builtins.object = _MISSING,
        nullable: bool = False,
        enum: Sequence[builtins.object] | None = None,
    ) -> Schema:
        return cls(
            "boolean",
            description=description,
            default=default,
            nullable=nullable,
            enum=enum,
        )

    @classmethod
    def object(
        cls,
        properties: Mapping[str, Schema] | None = None,
        *,
        required: Iterable[str] = (),
        description: str | None = None,
        default: builtins.object = _MISSING,
        nullable: bool = False,
        enum: Sequence[builtins.object] | None = None,
        additional_properties: bool | Schema | None = None,
        **named_properties: Schema,
    ) -> Schema:
        merged: dict[str, Schema] = {} if properties is None else dict(properties)
        overlap = set(merged).intersection(named_properties)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Schema object properties declared twice: {names}")
        merged.update(named_properties)
        return cls(
            "object",
            description=description,
            default=default,
            nullable=nullable,
            enum=enum,
            properties=merged,
            required=required,
            additional_properties=additional_properties,
        )

    @property
    def kind(self) -> SchemaKind:
        return self._kind

    @property
    def has_default(self) -> bool:
        return self._default is not _MISSING

    @property
    def default(self) -> builtins.object:
        if not self.has_default:
            raise AttributeError("Schema has no default")
        return deepcopy(self._default)

    @property
    def properties(self) -> dict[str, Schema]:
        return {name: schema for name, schema in self._properties}

    def to_json_schema(self) -> dict[str, Any]:
        """Return a detached JSON-compatible schema fragment."""
        type_value: str | list[str] = self._kind
        if self._nullable:
            type_value = [self._kind, "null"]
        result: dict[str, Any] = {"type": type_value}
        if self._description is not None:
            result["description"] = self._description
        if self.has_default:
            result["default"] = deepcopy(self._default)
        if self._enum is not None:
            result["enum"] = [deepcopy(value) for value in self._enum]
        for attribute, keyword in (
            (self._min_length, "minLength"),
            (self._max_length, "maxLength"),
            (self._pattern, "pattern"),
            (self._format, "format"),
            (self._minimum, "minimum"),
            (self._maximum, "maximum"),
            (self._exclusive_minimum, "exclusiveMinimum"),
            (self._exclusive_maximum, "exclusiveMaximum"),
            (self._multiple_of, "multipleOf"),
        ):
            if attribute is not None:
                result[keyword] = attribute
        if self._properties:
            result["properties"] = {
                name: property_schema.to_json_schema() for name, property_schema in self._properties
            }
        elif self._kind == "object":
            result["properties"] = {}
        if self._required:
            result["required"] = list(self._required)
        elif self._kind == "object":
            result["required"] = []
        if self._additional_properties is not None:
            additional = self._additional_properties
            result["additionalProperties"] = (
                additional.to_json_schema() if isinstance(additional, Schema) else additional
            )
        return result

    def cast(self, value: builtins.object) -> builtins.object:
        """Apply only the safe conversions and recursively copy JSON values."""
        if value is None:
            return None
        if self._kind == "integer":
            if isinstance(value, bool) or isinstance(value, int):
                return deepcopy(value)
            if isinstance(value, str) and _INTEGER_TEXT.fullmatch(value):
                return int(value)
            if isinstance(value, float) and math.isfinite(value) and value.is_integer():
                return int(value)
            return deepcopy(value)
        if self._kind == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            return deepcopy(value)
        if self._kind == "object" and isinstance(value, dict):
            properties = self.properties
            additional = self._additional_properties
            return {
                name: (
                    properties[name].cast(member)
                    if name in properties
                    else additional.cast(member)
                    if isinstance(additional, Schema)
                    else deepcopy(member)
                )
                for name, member in value.items()
            }
        return deepcopy(value)

    def validate(self, value: builtins.object, *, path: str = "$") -> list[SchemaError]:
        """Return every discoverable validation error in declaration order."""
        errors: list[SchemaError] = []
        self._validate(value, path, errors)
        return errors

    def _validate(self, value: builtins.object, path: str, errors: list[SchemaError]) -> None:
        if value is None:
            if not self._nullable:
                errors.append(SchemaError(path, f"expected {self._kind}, got null", "type"))
                return
        elif not _matches_kind(self._kind, value):
            errors.append(
                SchemaError(path, f"expected {self._kind}, got {_value_type(value)}", "type")
            )
            return
        if self._enum is not None and not any(_json_equal(value, member) for member in self._enum):
            errors.append(SchemaError(path, "value is not one of the allowed values", "enum"))
        if value is None:
            return
        if self._kind == "string":
            self._validate_string(cast(str, value), path, errors)
        elif self._kind == "integer":
            self._validate_number(cast(int | float, value), path, errors)
        elif self._kind == "object":
            self._validate_object(cast(dict[str, builtins.object], value), path, errors)

    def _validate_string(self, value: str, path: str, errors: list[SchemaError]) -> None:
        if self._min_length is not None and len(value) < self._min_length:
            errors.append(
                SchemaError(
                    path, f"must contain at least {self._min_length} characters", "minLength"
                )
            )
        if self._max_length is not None and len(value) > self._max_length:
            errors.append(
                SchemaError(
                    path, f"must contain at most {self._max_length} characters", "maxLength"
                )
            )
        if self._pattern is not None and re.search(self._pattern, value) is None:
            errors.append(SchemaError(path, "does not match the required pattern", "pattern"))
        if self._format is not None and not _valid_format(self._format, value):
            errors.append(SchemaError(path, f"is not a valid {self._format}", "format"))

    def _validate_number(self, value: int | float, path: str, errors: list[SchemaError]) -> None:
        if self._minimum is not None and value < self._minimum:
            errors.append(
                SchemaError(path, f"must be greater than or equal to {self._minimum}", "minimum")
            )
        if self._maximum is not None and value > self._maximum:
            errors.append(
                SchemaError(path, f"must be less than or equal to {self._maximum}", "maximum")
            )
        if self._exclusive_minimum is not None and value <= self._exclusive_minimum:
            errors.append(
                SchemaError(
                    path, f"must be greater than {self._exclusive_minimum}", "exclusiveMinimum"
                )
            )
        if self._exclusive_maximum is not None and value >= self._exclusive_maximum:
            errors.append(
                SchemaError(
                    path, f"must be less than {self._exclusive_maximum}", "exclusiveMaximum"
                )
            )
        if self._multiple_of is not None and not _is_multiple(value, self._multiple_of):
            errors.append(
                SchemaError(path, f"must be a multiple of {self._multiple_of}", "multipleOf")
            )

    def _validate_object(
        self,
        value: dict[str, builtins.object],
        path: str,
        errors: list[SchemaError],
    ) -> None:
        for name, property_schema in self._properties:
            property_path = _property_path(path, name)
            if name not in value:
                if name in self._required:
                    errors.append(SchemaError(property_path, "is required", "required"))
                continue
            property_schema._validate(value[name], property_path, errors)
        additional = self._additional_properties
        if additional is None or additional is True:
            return
        property_names = {name for name, _ in self._properties}
        for name in sorted(set(value) - property_names):
            property_path = _property_path(path, name)
            if additional is False:
                errors.append(
                    SchemaError(
                        property_path, "additional property is not allowed", "additionalProperties"
                    )
                )
            else:
                additional._validate(value[name], property_path, errors)


def _matches_kind(kind: SchemaKind, value: object) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return isinstance(value, dict)


def _value_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _property_path(path: str, name: str) -> str:
    if _IDENTIFIER.fullmatch(name):
        return f"{path}.{name}"
    return f"{path}[{dumps(name, ensure_ascii=False)}]"


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_json_number(left) and _is_json_number(right):
        if not _is_finite_number(left) or not _is_finite_number(right):
            return left == right
        return _number_fraction(left) == _number_fraction(right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[name], right[name]) for name in left
        )
    return left == right


def _valid_format(format_name: str, value: str) -> bool:
    if format_name == "email":
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))
    if format_name == "hostname":
        return bool(
            re.fullmatch(
                r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
                value,
            )
        )
    return True


def _is_multiple(value: int | float, divisor: int | float) -> bool:
    quotient = _number_fraction(value) / _number_fraction(divisor)
    return quotient.denominator == 1


def _is_json_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: int | float) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _number_fraction(value: int | float) -> Fraction:
    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def _require_finite_number(value: int | float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"Schema {name} must be a finite number")


def _require_json_value(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError(f"Schema {name} must be JSON-compatible")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{name}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Schema {name} must be JSON-compatible")
            _require_json_value(item, f"{name}.{key}")
        return value
    raise TypeError(f"Schema {name} must be JSON-compatible")


__all__ = [
    "Schema",
    "SchemaError",
    "ToolParam",
]
