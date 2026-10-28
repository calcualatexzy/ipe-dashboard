"""Experiment adapters. Each exposes `collect(spec, base_dir, registry, log) -> dict` and `report(data) -> str`."""

from . import distill, foody, msm

ADAPTERS = {"foody": foody, "msm": msm, "distill": distill}
