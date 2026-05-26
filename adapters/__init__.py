"""Adapter registry. Add new libraries here to make them available to the CLI."""
from __future__ import annotations

from .fasttreeshap_adapter import FastTreeSHAPv1, FastTreeSHAPv2
from .shapiq_adapter import ShapiqAdapter, ShapiqAdapterSII
from .stubs import GPUTreeShapAdapter, WoodelfAdapter

# Maps the CLI --adapter value to the adapter class.
REGISTRY = {
    cls.name: cls
    for cls in (
        FastTreeSHAPv1,
        FastTreeSHAPv2,
        ShapiqAdapter,
        ShapiqAdapterSII,
        GPUTreeShapAdapter,
        WoodelfAdapter,
    )
}


def get_adapter(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
