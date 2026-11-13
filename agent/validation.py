"""Validate model-supplied tool arguments against a tool's JSON schema (subset used here)."""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _check_value(name: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected = schema.get("type")
    if expected:
        types = _TYPES.get(expected, ())
        is_bool = isinstance(value, bool)
        if expected in ("integer", "number") and is_bool:
            return f"'{name}' must be {expected}"
        if types and not isinstance(value, types):
            return f"'{name}' must be {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return f"'{name}' must be one of {schema['enum']}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"'{name}' must be >= {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"'{name}' must be <= {schema['maximum']}"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"'{name}' needs at least {schema['minItems']} items"
        item_schema = schema.get("items")
        if item_schema:
            for item in value:
                problem = _check_value(name, item, item_schema)
                if problem:
                    return problem
    return None


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return a problem description, or None when the arguments are valid."""
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(props))
        if unknown:
            return f"unknown argument(s) {unknown}; allowed: {sorted(props)}"
    for key, value in args.items():
        if key in props:
            problem = _check_value(key, value, props[key])
            if problem:
                return problem
    return None
