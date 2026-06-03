"""Subprocess client for tools/sam2_server.py (JSON lines on stdin/stdout)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class SAM2Client:
    def __init__(self, python_exe: str, server_script: Path):
        self.server_script = Path(server_script).resolve()
        if not self.server_script.is_file():
            raise FileNotFoundError(f"SAM2 server not found: {self.server_script}")

        print(f"Starting SAM2 server: {python_exe} {self.server_script.name}")
        self.proc = subprocess.Popen(
            [python_exe, str(self.server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        resp = self._read()
        if resp.get("status") != "ready":
            raise RuntimeError(f"SAM2 server failed to start: {resp}")
        print("SAM2 server ready")

    def _read(self) -> dict[str, Any]:
        line = self.proc.stdout.readline()
        return json.loads(line) if line else {"status": "error", "msg": "no response"}

    def _send(self, obj: dict[str, Any]) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def set_image(self, image_path: str | Path) -> dict[str, Any]:
        self._send({"cmd": "set_image", "path": str(Path(image_path).resolve())})
        return self._read()

    def predict(self, fg: list[list[int]], bg: list[list[int]]) -> dict[str, Any]:
        self._send({"cmd": "predict", "fg": fg, "bg": bg})
        return self._read()

    def close(self) -> None:
        try:
            self._send({"cmd": "quit"})
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
