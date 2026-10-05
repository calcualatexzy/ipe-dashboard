"""Shared helpers for experiment adapters: hashing, flattening, sanitizing, JSON output."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(r"(api_?key|token|secret|password)", re.IGNORECASE)
RUN_STAMP = re.compile(r"_(\d{8})_(\d{6})(?:_\d+)?$")


def short_hash(obj: Any, n: int = 6) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys; lists and scalars are kept as leaf values."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def run_timestamp(name: str) -> str | None:
    """'..._20260928_182505' or '..._20260928_182505_629' -> '2026-09-28T18:25:05'."""
    m = RUN_STAMP.search(name)
    if not m:
        return None
    d, t = m.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}"


class Sanitizer:
    """Rewrites absolute paths so nothing machine- or user-specific reaches the published site.

    Paths under a known root become '<label>/<relative>'; any other absolute path keeps only
    its last `keep` components. Keys that look like credentials are dropped entirely.
    """

    def __init__(self, roots: dict[str, str], keep: int = 3):
        # Longest prefix first so nested roots win.
        self.roots = sorted(((p.rstrip("/"), label) for p, label in roots.items()), key=lambda x: -len(x[0]))
        self.keep = keep

    def path(self, s: str) -> str:
        if not s.startswith("/"):
            return s
        for prefix, label in self.roots:
            if s == prefix or s.startswith(prefix + "/"):
                return label + s[len(prefix):]
        parts = [p for p in s.split("/") if p]
        return "…/" + "/".join(parts[-self.keep:]) if len(parts) > self.keep else s

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.path(v)
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items() if not SECRET_KEY.search(str(k))}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        return v


def write_json(path: Path, obj: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text)
