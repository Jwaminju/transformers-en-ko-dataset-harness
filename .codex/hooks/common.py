from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_event() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {"raw": raw}
    return {}


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for child in value.values():
            values.extend(_flatten_strings(child))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for child in value:
            values.extend(_flatten_strings(child))
        return values
    return []


def guess_command(event: dict[str, Any]) -> str:
    preferred_paths = [
        ("tool_input", "command"),
        ("input", "command"),
        ("command",),
        ("cmd",),
    ]
    for path in preferred_paths:
        current: Any = event
        found = True
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if found and isinstance(current, str):
            return current
    strings = _flatten_strings(event)
    return max(strings, key=len) if strings else ""


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def warn(message: str) -> None:
    print(message, file=sys.stderr)
