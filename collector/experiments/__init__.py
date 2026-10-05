"""Experiment adapters. Each exposes `collect(spec, base_dir, registry, log) -> dict` and `report(data) -> str`."""

from . import foody

ADAPTERS = {"foody": foody}
