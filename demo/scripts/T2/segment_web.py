#!/usr/bin/env python3
"""
T2 — SAM2 segmentation via browser (Flask, no X11, no Gradio).

On Titan:
  conda activate bundlesdf
  python demo/scripts/T2/segment_web.py \\
    --session-dir demo/sessions/20260602_192346_chips \\
    --port 7860

On your laptop:
  ssh -L 7860:127.0.0.1:7860 vision@<titan-host>
  open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from _session_io import resolve_session_dirs  # noqa: E402

_T2_DIR = Path(__file__).resolve().parent
if str(_T2_DIR) not in sys.path:
    sys.path.insert(0, str(_T2_DIR))

from segment_common import (  # noqa: E402
    Sam2Predictor,
    check_sam2_installed,
    load_rgb_pil,
    mask_coverage_pct,
    render_overlay,
    session_id_from_dirs,
    write_outputs,
)

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>T2 SAM2 — {{ session_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; background: #1a1d24; color: #e8e8e8; }
    h1 { font-size: 1.2rem; }
    #wrap { position: relative; display: inline-block; cursor: crosshair; }
    #img { max-width: 100%; height: auto; display: block; }
    .bar { margin: 12px 0; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button { padding: 8px 14px; cursor: pointer; border-radius: 6px; border: none; }
    button.primary { background: #3d7eff; color: #fff; }
    button.secondary { background: #444; color: #fff; }
    button.done { background: #2d9a5b; color: #fff; }
    button.done:disabled { background: #355; color: #888; cursor: not-allowed; }
    #status, #saveOut { background: #2a2f3a; padding: 10px; border-radius: 6px; min-height: 1.2em; }
    label { margin-right: 8px; }
    .hint { color: #9ab; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>T2 SAM2 — {{ session_id }}</h1>
  <p class="hint">Click image: <b>Foreground</b> or <b>Background</b> (radio). SAM2 on Titan GPU.
    Tunnel: <code>ssh -L {{ port }}:127.0.0.1:{{ port }} user@titan</code></p>
  <div class="bar">
    <label><input type="radio" name="mode" value="fg" checked/> Foreground</label>
    <label><input type="radio" name="mode" value="bg"/> Background</label>
    <button class="secondary" onclick="clearPts()">Clear points</button>
    <button class="primary" onclick="saveMask()">Save mask to session</button>
    <button class="done" id="doneBtn" onclick="finishDone()">Done (exit)</button>
  </div>
  <div id="wrap"><img id="img" src="/overlay.png" alt="session frame"/></div>
  <p id="status">Loading…</p>
  <p id="saveOut"></p>
  <script>
    const wrap = document.getElementById('wrap');
    const img = document.getElementById('img');
    function mode() {
      return document.querySelector('input[name=mode]:checked').value;
    }
    wrap.addEventListener('click', async (e) => {
      const r = img.getBoundingClientRect();
      const sx = img.naturalWidth / r.width;
      const sy = img.naturalHeight / r.height;
      const x = Math.round((e.clientX - r.left) * sx);
      const y = Math.round((e.clientY - r.top) * sy);
      const res = await fetch('/api/click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({x, y, mode: mode()}),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.error || 'click failed'); return; }
      document.getElementById('status').textContent = data.status;
      img.src = '/overlay.png?t=' + Date.now();
    });
    async function clearPts() {
      const res = await fetch('/api/clear', {method: 'POST'});
      const data = await res.json();
      document.getElementById('status').textContent = data.status;
      img.src = '/overlay.png?t=' + Date.now();
    }
    function setDoneEnabled(saved) {
      document.getElementById('doneBtn').disabled = !saved;
    }
    async function saveMask() {
      const res = await fetch('/api/save', {method: 'POST'});
      const data = await res.json();
      if (!data.ok) {
        document.getElementById('saveOut').textContent = data.error || 'save failed';
        return;
      }
      document.getElementById('saveOut').textContent = data.message || '';
      if (data.status) document.getElementById('status').textContent = data.status;
      setDoneEnabled(!!data.saved);
    }
    async function finishDone() {
      const res = await fetch('/api/done', {method: 'POST'});
      const data = await res.json();
      if (!data.ok) {
        alert(data.error || 'Save mask first, then click Done.');
        return;
      }
      document.body.innerHTML = '<p style="padding:24px;font-size:1.1rem;">'
        + (data.message || 'T2 complete. Server stopped.') + '</p>';
      try { window.close(); } catch (e) {}
    }
    fetch('/api/status').then(r => r.json()).then(d => {
      document.getElementById('status').textContent = d.status;
      setDoneEnabled(!!d.saved);
    });
  </script>
</body>
</html>"""


@dataclass
class AnnotatorSession:
    rgb: np.ndarray
    engine: Sam2Predictor
    out_segment: Path
    mask_path: Path
    session_id: str
    fg: list[list[int]] = field(default_factory=list)
    bg: list[list[int]] = field(default_factory=list)
    mask: np.ndarray | None = None
    saved: bool = False

    def __post_init__(self) -> None:
        if self.mask_path.is_file() and (self.out_segment / "prompt_used.json").is_file():
            self.saved = True

    def is_saved(self) -> bool:
        return self.saved and self.mask_path.is_file()

    def status_text(self) -> str:
        cov = mask_coverage_pct(self.mask) if self.mask is not None else 0.0
        return (
            f"FG: {len(self.fg)}  BG: {len(self.bg)}  "
            f"coverage: {cov:.1f}%  — click then Save"
        )

    def overlay_png_bytes(self) -> bytes:
        vis = render_overlay(self.rgb, self.mask, self.fg, self.bg)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", vis_bgr)
        if not ok:
            raise RuntimeError("Failed to encode overlay PNG")
        return buf.tobytes()

    def clear_points(self) -> None:
        self.fg = []
        self.bg = []
        self.mask = None
        self.engine.set_image(self.rgb)

    def add_point(self, x: int, y: int, foreground: bool) -> None:
        h, w = self.rgb.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if foreground:
            self.fg.append([x, y])
        else:
            self.bg.append([x, y])
        self.mask = self.engine.predict_mask(self.fg, self.bg)

    def save(self) -> str:
        if self.mask is None or not self.fg:
            raise ValueError("Need at least one foreground point and a SAM2 mask before saving.")

        write_outputs(
            self.out_segment,
            self.mask,
            self.fg,
            self.bg,
            self.session_id,
            self.rgb.shape[:2],
            source="flask_web",
        )
        cov = mask_coverage_pct(self.mask)
        self.saved = True
        return f"Saved {self.mask_path}  (coverage {cov:.1f}%)"


def _stop_werkzeug_server(shutdown_func) -> None:
    """Stop Flask dev server after /api/done response is sent."""
    import os
    import signal

    time.sleep(0.25)
    if shutdown_func is not None:
        shutdown_func()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def create_app(sess: AnnotatorSession, port: int) -> Flask:
    app = Flask(__name__)
    app.config["SESSION"] = sess
    app.config["PORT"] = port

    @app.route("/")
    def index():
        return render_template_string(
            HTML,
            session_id=sess.session_id,
            port=port,
        )

    @app.route("/overlay.png")
    def overlay_png():
        return app.response_class(
            app.config["SESSION"].overlay_png_bytes(),
            mimetype="image/png",
        )

    @app.route("/api/status")
    def api_status():
        s = app.config["SESSION"]
        return jsonify({"ok": True, "status": s.status_text(), "saved": s.is_saved()})

    @app.route("/api/click", methods=["POST"])
    def api_click():
        s = app.config["SESSION"]
        data = request.get_json(force=True) or {}
        try:
            x, y = int(data["x"]), int(data["y"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "Need integer x, y"}), 400
        fg = (data.get("mode") or "fg") == "fg"
        s.add_point(x, y, foreground=fg)
        return jsonify({"ok": True, "status": s.status_text()})

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        s = app.config["SESSION"]
        s.clear_points()
        return jsonify({"ok": True, "status": s.status_text()})

    @app.route("/api/save", methods=["POST"])
    def api_save():
        s = app.config["SESSION"]
        try:
            msg = s.save()
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify(
            {
                "ok": True,
                "message": msg,
                "status": s.status_text(),
                "saved": s.is_saved(),
            }
        )

    @app.route("/api/done", methods=["POST"])
    def api_done():
        s = app.config["SESSION"]
        if not s.is_saved():
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Save mask to session first (mask.png + prompt_used.json).",
                    }
                ),
                400,
            )
        shutdown_func = request.environ.get("werkzeug.server.shutdown")
        threading.Thread(
            target=_stop_werkzeug_server,
            args=(shutdown_func,),
            daemon=True,
        ).start()
        return jsonify(
            {
                "ok": True,
                "message": "T2 complete. You can close this tab; server is shutting down.",
            }
        )

    return app


def run_segment_web(
    dirs,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    redo: bool = False,
) -> int:
    """
    Block until operator saves mask and clicks Done (Flask server exits).

    Returns 0 if mask.png exists after shutdown, else 1.
    """
    err = check_sam2_installed()
    if err:
        print(err, file=sys.stderr)
        return 2

    rgb_path = dirs.input_rel("rgb", "left_rgb.png")
    mask_out = dirs.output_rel("segment", "mask.png")

    if not rgb_path.is_file():
        print(f"Missing {rgb_path}", file=sys.stderr)
        return 1
    if mask_out.is_file() and not redo:
        print(f"{mask_out} exists — use --redo to re-annotate")
        return 0

    print("Loading SAM2...")
    engine = Sam2Predictor()
    rgb = load_rgb_pil(rgb_path)
    engine.set_image(rgb)

    session_id = session_id_from_dirs(dirs.input_dir, dirs.session_id)
    out_segment = dirs.output_rel("segment")
    out_segment.mkdir(parents=True, exist_ok=True)

    sess = AnnotatorSession(
        rgb=rgb,
        engine=engine,
        out_segment=out_segment,
        mask_path=out_segment / "mask.png",
        session_id=session_id,
    )

    print(f"RGB {rgb.shape[1]}x{rgb.shape[0]}  session {session_id}")
    print(f"T2 web UI: http://127.0.0.1:{port}  (server bind {host}:{port})")
    print(f"  ssh -L {port}:127.0.0.1:{port} <user>@<titan-host>")
    print("Save mask, then click Done to continue the Titan daemon.")

    app = create_app(sess, port)
    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    print("T2 web server stopped.")

    if mask_out.is_file() and (out_segment / "prompt_used.json").is_file():
        return 0
    print("T2 incomplete: save mask and click Done before closing.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 T2: SAM2 Flask web UI")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", type=Path)
    g.add_argument("--input-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)

    dirs = resolve_session_dirs(
        session_dir=args.session_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    return run_segment_web(
        dirs,
        host=args.host,
        port=args.port,
        redo=args.redo,
    )


if __name__ == "__main__":
    sys.exit(main())
