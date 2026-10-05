"""DepthForge · depth_core.py — inti konversi gambar → depth map (mode klasik).

Dua cara pakai:
  1. CLI   : python py/depth_core.py masukan.jpg keluaran.png --detail 40 --invert
  2. Modul : dipakai server.py untuk endpoint /api/depth

Rantai proses identik dengan js/engine.js sehingga hasil lokal dan
hasil di aplikasi web konsisten.
"""
from __future__ import annotations

import argparse
import io
import math

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- util blur
def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    n = 2 * r + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(ap, axis=axis, dtype=np.float32)
    zshape = list(cs.shape)
    zshape[axis] = 1
    cs = np.concatenate([np.zeros(zshape, np.float32), cs], axis=axis)
    hi = [slice(None)] * a.ndim
    hi[axis] = slice(n, None)
    lo = [slice(None)] * a.ndim
    lo[axis] = slice(None, -n)
    return (cs[tuple(hi)] - cs[tuple(lo)]) / np.float32(n)


def box_blur(d: np.ndarray, r: int, passes: int = 3) -> np.ndarray:
    if r < 1:
        return d
    out = d
    for _ in range(passes):
        out = _box1d(_box1d(out, r, axis=1), r, axis=0)
    return out


def bilateral(d: np.ndarray, sigma_r: float, radius: int = 2) -> np.ndarray:
    """Smoothing yang menjaga tepi (sama seperti versi JS)."""
    H, W = d.shape
    pad = np.pad(d, radius, mode="edge")
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    ws = np.exp(-(yy * yy + xx * xx) / (2 * 1.4 * 1.4)).astype(np.float32)
    inv2sr = 1.0 / (2 * sigma_r * sigma_r)
    vsum = np.zeros_like(d)
    wsum = np.zeros_like(d)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            v = pad[radius + dy: radius + dy + H, radius + dx: radius + dx + W]
            diff = v - d
            w = np.exp(-(diff * diff) * inv2sr).astype(np.float32) * ws[dy + radius, dx + radius]
            wsum += w
            vsum += w * v
    return vsum / np.maximum(wsum, 1e-9)


def equalize_lut(x: np.ndarray) -> np.ndarray:
    hist = np.bincount((np.clip(x, 0, 1) * 255).astype(np.int32).ravel(), minlength=256)
    cdf = np.cumsum(hist).astype(np.float32)
    nonzero = np.nonzero(hist)[0]
    cdf_min = cdf[nonzero[0]] if len(nonzero) else 0
    denom = max(1.0, cdf[-1] - cdf_min)
    return np.clip(((cdf - cdf_min) / denom) * 255.0, 0, 255).astype(np.int32)


# ---------------------------------------------------------------- pipeline
def to_depth(img: Image.Image, channel="luma", blur=0, detail=0, detail_radius=3,
             smooth=0, equalize=False, contrast=0.0, brightness=0.0, gamma=1.0,
             in_black=0, in_white=255, out_black=0, out_white=255, invert=False,
             neural: np.ndarray | None = None) -> np.ndarray:
    """Kembalikan depth map float32 (0..1), putih = tinggi."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if neural is not None:
        d = neural.astype(np.float32)
    elif channel == "red":
        d = arr[..., 0]
    elif channel == "green":
        d = arr[..., 1]
    elif channel == "blue":
        d = arr[..., 2]
    elif channel == "max":
        d = arr.max(axis=2)
    else:
        d = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    blur_r = int(round(blur))
    det_r = max(1, min(20, int(round(detail_radius))))
    low = box_blur(d, blur_r)
    shape = low
    if detail and detail > 0:  # unsharp dua skala (makro + mikro)
        k = detail / 50.0
        small = box_blur(d, det_r)
        shape = low + k * ((small - low) + 0.55 * (d - small))
    if smooth and smooth > 0:
        shape = bilateral(shape, 0.03 + (smooth / 100.0) * 0.30)
    shape = np.clip(shape, 0, 1)

    # equalize → levels → gamma → brightness → kontras → invert (via LUT 256)
    lut_x = np.arange(256)
    if equalize:
        lut_x = equalize_lut(shape)[lut_x]
    v = lut_x.astype(np.float32) / 255.0

    rng = max(1.0, in_white - in_black)
    v = np.clip((v * 255.0 - in_black) / rng, 0, 1)
    orng = max(1.0, out_white - out_black)
    v = np.clip((v * (out_white - out_black) + out_black) / 255.0, 0, 1)
    v = np.power(v, 1.0 / max(0.05, gamma))
    v = np.clip(v + brightness / 200.0, 0, 1)
    cf = math.tan(((max(-98, min(98, contrast)) + 100) / 200.0) * math.pi / 2)
    v = np.clip(cf * (v - 0.5) + 0.5, 0, 1)
    if invert:
        v = 1.0 - v
    return v.astype(np.float32)


# ---------------------------------------------------------------- keluaran
def depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    return (np.clip(depth, 0, 1) * 65535.0 + 0.5).astype(np.uint16)


def to_png16(depth: np.ndarray) -> bytes:
    im = Image.fromarray(depth_to_uint16(depth), mode="I;16")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def relief_shading(depth: np.ndarray, az=315.0, el=45.0, material="gypsum",
                   src: Image.Image | None = None) -> np.ndarray:
    """Arsir relief 3D dari heightfield (Lambert + specular)."""
    s = 2.0
    gy, gx = np.gradient(depth)
    gx *= s
    gy *= s
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx * inv, -gy * inv, inv
    azr, elr = math.radians(az), math.radians(el)
    L = np.array([math.cos(azr) * math.cos(elr),
                  math.sin(azr) * math.cos(elr),
                  math.sin(elr)], np.float32)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, None)
    a = 0.14 + 0.92 * diff
    spec = np.power(diff, 22.0) * 0.45

    if material == "source" and src is not None:
        alb = np.asarray(src.convert("RGB"), dtype=np.float32)
        if alb.shape[:2] != depth.shape:
            alb = np.asarray(src.convert("RGB").resize(
                (depth.shape[1], depth.shape[0])), dtype=np.float32)
    elif material == "graphite":
        alb = np.full((*depth.shape, 3), 78.0, np.float32)
    elif material == "copper":
        alb = np.zeros((*depth.shape, 3), np.float32)
        alb[..., 0], alb[..., 1], alb[..., 2] = 205, 128, 82
    else:
        alb = np.full((*depth.shape, 3), 214.0, np.float32)

    rgb = alb * a[..., None] + spec[..., None] * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Konversi JPG/PNG → depth map 16-bit")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--channel", default="luma", choices=["luma", "red", "green", "blue", "max"])
    ap.add_argument("--blur", type=int, default=0)
    ap.add_argument("--detail", type=float, default=0)
    ap.add_argument("--detail-radius", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=0)
    ap.add_argument("--equalize", action="store_true")
    ap.add_argument("--contrast", type=float, default=0)
    ap.add_argument("--brightness", type=float, default=0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--in-black", type=int, default=0)
    ap.add_argument("--in-white", type=int, default=255)
    ap.add_argument("--out-black", type=int, default=0)
    ap.add_argument("--out-white", type=int, default=255)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--relief", metavar="PNG", help="juga simpan relief terarsir")
    ap.add_argument("--az", type=float, default=315.0, help="azimut cahaya (derajat)")
    ap.add_argument("--el", type=float, default=45.0, help="elevasi cahaya (derajat)")
    ap.add_argument("--material", default="gypsum",
                    choices=["gypsum", "source", "graphite", "copper"])
    a = ap.parse_args(argv)

    img = Image.open(a.input)
    depth = to_depth(img, channel=a.channel, blur=a.blur, detail=a.detail,
                     detail_radius=a.detail_radius, smooth=a.smooth, equalize=a.equalize,
                     contrast=a.contrast, brightness=a.brightness, gamma=a.gamma,
                     in_black=a.in_black, in_white=a.in_white, out_black=a.out_black,
                     out_white=a.out_white, invert=a.invert)
    Image.fromarray(depth_to_uint16(depth), "I;16").save(a.output)
    print(f"[ok] {a.output}  {depth.shape[1]}x{depth.shape[0]} · 16-bit")

    if a.relief:
        rel = relief_shading(depth, a.az, a.el, a.material, src=img)
        Image.fromarray(rel).save(a.relief)
        print(f"[ok] {a.relief}  (relief, az={a.az} el={a.el})")


if __name__ == "__main__":
    main()