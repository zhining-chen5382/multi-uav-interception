"""Shared three-dimensional interception geometry."""

import numpy as np


def segment_distances(pursuer_positions, attacker_positions, base_position):
    """Return 3D distance to the closed base-to-attacker segment."""
    pursuer = np.asarray(pursuer_positions, dtype=np.float64)
    attacker = np.asarray(attacker_positions, dtype=np.float64)
    base = np.asarray(base_position, dtype=np.float64)
    for array in (pursuer, attacker, base):
        if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
            raise ValueError("Positions must be finite arrays with final dimension 3")
    pursuer, attacker, base = np.broadcast_arrays(pursuer, attacker, base)
    direction = attacker - base
    denominator = np.sum(direction * direction, axis=-1)
    numerator = np.sum((pursuer - base) * direction, axis=-1)
    projection = np.zeros_like(denominator)
    np.divide(numerator, denominator, out=projection, where=denominator > 0)
    nearest = base + np.clip(projection, 0.0, 1.0)[..., None] * direction
    distances = np.linalg.norm(pursuer - nearest, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("Segment geometry exceeds the supported numerical range")
    return distances
