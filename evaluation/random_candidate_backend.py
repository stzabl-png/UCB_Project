"""Candidate generation backend ids for eval_pool."""

from __future__ import annotations

CANDIDATE_BACKEND_PDM = "pdm"
CANDIDATE_BACKEND_RANDOM_RP = "random_rp"
CANDIDATE_BACKEND_RANDOM_HP = "random_hp"
CANDIDATE_BACKEND_RANDOM_PURE = "random_pure"

RANDOM_CANDIDATE_BACKENDS = frozenset(
    {
        CANDIDATE_BACKEND_RANDOM_RP,
        CANDIDATE_BACKEND_RANDOM_HP,
        CANDIDATE_BACKEND_RANDOM_PURE,
    }
)

CANDIDATE_BACKEND_CHOICES = (
    CANDIDATE_BACKEND_PDM,
    CANDIDATE_BACKEND_RANDOM_RP,
    CANDIDATE_BACKEND_RANDOM_HP,
    CANDIDATE_BACKEND_RANDOM_PURE,
)


def is_random_candidate_backend(backend: str) -> bool:
    return str(backend) in RANDOM_CANDIDATE_BACKENDS


def uses_v6_affordance_gate(backend: str) -> bool:
    return str(backend) in (CANDIDATE_BACKEND_RANDOM_RP, CANDIDATE_BACKEND_RANDOM_HP)


def resolve_hp_affordance_for_backend(backend: str, hp_affordance: bool) -> bool:
    """Map backend to affordance checkpoint selection in batch_random_candidates."""
    if backend == CANDIDATE_BACKEND_RANDOM_RP:
        return False
    if backend == CANDIDATE_BACKEND_RANDOM_HP:
        return True
    if backend == CANDIDATE_BACKEND_RANDOM_PURE:
        return False
    return bool(hp_affordance)


def add_candidate_backend_args(parser) -> None:
    parser.add_argument(
        "--candidate-backend",
        choices=CANDIDATE_BACKEND_CHOICES,
        default=CANDIDATE_BACKEND_PDM,
        help=(
            "Candidate pool generator: pdm (default), random_rp (robot v6 gate), "
            "random_hp (HP v6 gate), random_pure (geometry gates only)."
        ),
    )
