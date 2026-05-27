"""Non-deterministic seeds for policy sampling across evaluation trials."""

from __future__ import annotations

import secrets


def fresh_rng():
    import numpy as np

    return np.random.default_rng(secrets.randbits(128))


def fresh_seed() -> int:
    return secrets.randbelow(2**31 - 1)


def resolve_policy_seed(policy_seed: int | None, *, trial: int | None = None) -> int:
    """Return an integer seed for policy HDF5 selection; None → fresh draw each call."""
    if policy_seed is not None:
        if trial is not None:
            return int(policy_seed) + int(trial) * 10007
        return int(policy_seed)
    return fresh_seed()
