"""DepthForge · server.py — API lokal opsional (Flask).

Endpoint:
  GET  /api/ping              → status server (neural=true bila MiDaS siap)
  POST /api/depth?model=midas → multipart 'file' (+ 'params' JSON) → PNG depth 16-bit

Jalankan:
  pip install -r requirements.txt
  python py/server.py            # mode klasik
  python py/server.py --midas    # unduh & pakai MiDaS_small (perlu torch)

Lalu klik chip "server python" di aplikasi web untuk menyambungkan.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, send_file, jsonify
import numpy as np
from PIL import Image

import depth_core

app = Flask(__name__)
NEURAL = {"ready": False, "model": None, "tf": None}


def load_midas():
    try:
        import torch
        print("[midas] memuat MiDaS_small (unduhan pertama ±200 MB)…")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()
        tf = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        NEURAL.update(ready=True, model=model, tf=tf)
        print("[midas] siap.")
    except Exception as e:
        print(f"[midas] tidak tersedia ({e}) — server memakai pipeline klasik.")


def run_midas(img: Image.Image) -> np.ndarray:
    import torch
    arr = np.asarray(img)
    inp = NEURAL["tf"](arr)
    with torch.no_grad():
        pred = NEURAL["model"](inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=arr.shape[:2],
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
    lo, hi = float(pred.min()), float(pred.max())
    return ((pred - lo) / max(1e-6, hi - lo)).astype(np.float32)


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/api/ping")
def ping():
    return jsonify(ok=True, name="DepthForge Server", version="1.0", neural=NEURAL["ready"])


@app.post("/api/depth")
def depth():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="berkas 'file' tidak ditemukan"), 400
    img = Image.open(f.stream).convert("RGB")
    try:
        params = json.loads(request.form.get("params", "{}"))
    except json.JSONDecodeError:
        params = {}

    if request.args.get("model") == "midas" and NEURAL["ready"]:
        d = run_midas(img)
    else:
        d = depth_core.to_depth(img, **params)

    return send_file(io.BytesIO(depth_core.to_png16(d)),
                     mimetype="image/png", download_name="depth16.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--midas", action="store_true")
    a = ap.parse_args()
    if a.midas:
        load_midas()
    print(f"DepthForge server → http://127.0.0.1:{a.port}")
    app.run(host="127.0.0.1", port=a.port, debug=False)