"""Experiment adapters. Each exposes `collect(spec, base_dir, registry, log) -> dict` and `report(data) -> str`."""

from . import align, distill, foody, msm, spp

ADAPTERS = {"foody": foody, "msm": msm, "distill": distill, "align": align, "spp": spp}
