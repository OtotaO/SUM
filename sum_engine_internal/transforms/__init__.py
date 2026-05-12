"""Transform registry — single dispatch surface for every signed transform.

The registry is a fixed-at-import-time dict; new transforms are
registered by adding an entry here. The buyer-or-dream filter applies:
each registered transform must serve a named buyer, named funder
commitment, or named dream element. See ``docs/TRANSFORM_REGISTRY.md``
§5 for the v1 transform set.

Public API:

    register(transform)            — add a Transform instance to the registry
    get_transform(name)            — look up by registry id
    list_transforms()              — sorted list of registered names
    Transform, TransformEnv, TransformResult — protocol + data shapes

Built-in transforms auto-register on import of this module. The
v1 registry:

    "slider" — bidirectional knowledge-distillation render
               (closes the dream's compress direction)
"""
from __future__ import annotations

from typing import Dict

from sum_engine_internal.transforms._base import (
    DigitalSourceType,
    Provider,
    Transform,
    TransformEnv,
    TransformResult,
)


_REGISTRY: Dict[str, Transform] = {}


def register(transform: Transform) -> Transform:
    """Register a Transform instance. Idempotent on the same instance
    (registering twice with the same name AND the same object is a
    no-op); raises ``ValueError`` if the name is already taken by a
    different transform instance.

    Returns the registered transform so this can be used as a
    decorator-shaped one-liner at module load time.
    """
    existing = _REGISTRY.get(transform.name)
    if existing is transform:
        return transform
    if existing is not None:
        raise ValueError(
            f"transform name {transform.name!r} already registered to "
            f"a different instance ({type(existing).__name__}); "
            f"registry is fixed at import time — refactor the second "
            f"transform to use a different name."
        )
    _REGISTRY[transform.name] = transform
    return transform


def get_transform(name: str) -> Transform:
    """Look up a transform by registry id. Raises ``KeyError`` with
    the available names so callers can produce helpful errors."""
    if name not in _REGISTRY:
        known = sorted(_REGISTRY.keys())
        raise KeyError(
            f"unknown transform {name!r}; known: {known}"
        )
    return _REGISTRY[name]


def list_transforms() -> list[str]:
    """Sorted list of registered transform names. Useful for the
    ``sum transform --help`` CLI surface and for the
    ``sum compliance regimes``-style discovery endpoint."""
    return sorted(_REGISTRY.keys())


# ─── Built-in transforms — auto-register on import ──────────────────
#
# Order matters only for tie-breaking in `list_transforms()` (which
# sorts alphabetically anyway). Add new built-ins below; third-party
# transforms are out of scope for v1 (see TRANSFORM_REGISTRY.md §7
# deferral #1).

from sum_engine_internal.transforms.slider import SLIDER_TRANSFORM as _SLIDER  # noqa: E402

register(_SLIDER)


__all__ = [
    "DigitalSourceType",
    "Provider",
    "Transform",
    "TransformEnv",
    "TransformResult",
    "get_transform",
    "list_transforms",
    "register",
]
