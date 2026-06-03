"""Shared session path resolution for Phase 2 demo scripts (T1, T2, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionDirs:
    """Resolved paths for one Razor/Titan session."""

    session_root: Path
    input_dir: Path
    output_dir: Path

    @property
    def session_id(self) -> str:
        return self.session_root.name

    def input_rel(self, *parts: str) -> Path:
        return self.input_dir.joinpath(*parts)

    def output_rel(self, *parts: str) -> Path:
        return self.output_dir.joinpath(*parts)


def repo_root() -> Path:
    """Affordance2Grasp project root (parent of demo/)."""
    return Path(__file__).resolve().parents[2]


def resolve_session_dirs(
    session_dir: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> SessionDirs:
    """
    Resolve session_root, input/, and output/.

    Provide either:
      - --session-dir <root>  (must contain input/ or be input/ itself)
      - --input-dir <path/to/input>  (output defaults to sibling output/)
    Optional --output-dir overrides output location.
    """
    if input_dir is not None:
        input_dir = Path(input_dir).resolve()
        if input_dir.name != "input":
            raise ValueError(f"--input-dir must be an 'input/' folder, got: {input_dir}")
        session_root = input_dir.parent
        out = (
            Path(output_dir).resolve()
            if output_dir is not None
            else session_root / "output"
        )
        return SessionDirs(session_root=session_root, input_dir=input_dir, output_dir=out)

    if session_dir is None:
        raise ValueError("Provide --session-dir or --input-dir")

    session_root = Path(session_dir).resolve()
    if session_root.name == "input" and session_root.is_dir():
        session_root = session_root.parent
        inp = session_root / "input"
    elif (session_root / "input").is_dir():
        inp = session_root / "input"
    else:
        raise FileNotFoundError(
            f"No input/ under {session_root}. Expected <session_id>/input/ "
            "or pass --input-dir."
        )

    out = (
        Path(output_dir).resolve()
        if output_dir is not None
        else session_root / "output"
    )
    return SessionDirs(session_root=session_root, input_dir=inp, output_dir=out)
