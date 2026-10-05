/* DepthForge · engine.js
   Pipeline piksel gambar → depth map (mode klasik, tanpa ML).
   Rantai: sumber → blur → detail (unsharp 2 skala) → smoothing edge-aware
           → equalize → levels → gamma → brightness/kontras → invert.
   Hasil: depth 8-bit (tampilan) + 16-bit (ekspor CNC/lithophane). */
const Engine = (() => {
  'use strict';

  const DEFAULTS = {
    channel: 'luma', blur: 0, detail: 0, detailRadius: 3, smooth: 0,
    equalize: false, contrast: 0, brightness: 0, gamma: 1,
    inBlack: 0, inWhite: 255, outBlack: 0, outWhite: 255, invert: false
  };

  const clampI = (v, a, b) => v < a ? a : (v > b ? b : v);
  const cl01 = v => v < 0 ? 0 : (v > 1 ? 1 : v);

  /* ---------- ekstraksi kanal sumber ---------- */
  function toBase(img, mode, out) {
    const d = img.data;
    for (let i = 0, p = 0; i < out.length; i++, p += 4) {
      let v;
      switch (mode) {
        case 'red':   v = d[p]; break;
        case 'green': v = d[p + 1]; break;
        case 'blue':  v = d[p + 2]; break;
        case 'max':   v = Math.max(d[p], d[p + 1], d[p + 2]); break;
        default:      v = 0.2126 * d[p] + 0.7152 * d[p + 1] + 0.0722 * d[p + 2];
      }
      out[i] = v / 255;
    }
    return out;
  }

  /* ---------- box blur separable (3 lintasan ≈ gaussian) ---------- */
  function passH(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let s = 0;
      for (let k = -r; k <= r; k++) s += src[row + clampI(k, 0, w - 1)];
      for (let x = 0; x < w; x++) {
        dst[row + x] = s * inv;
        s += src[row + clampI(x + r + 1, 0, w - 1)] - src[row + clampI(x - r, 0, w - 1)];
      }
    }
  }
  function passV(src, dst, w, h, r) {
    const inv = 1 / (2 * r + 1);
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += src[clampI(k, 0, h - 1) * w + x];
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = s * inv;
        s += src[clampI(y + r + 1, 0, h - 1) * w + x] - src[clampI(y - r, 0, h - 1) * w + x];
      }
    }
  }
  function boxBlur(src, w, h, r) {
    if (r < 1) return src.slice();
    let a = Float32Array.from(src), b = new Float32Array(a.length);
    for (let p = 0; p < 3; p++) { passH(a, b, w, h, r); passV(b, a, w, h, r); }
    return a;
  }

  /* ---------- bilateral filter (menjaga tepi), LUT rentang ---------- */
  function bilateral(src, w, h, sigmaR) {
    const r = 2;
    const lut = new Float32Array(1025);
    for (let i = 0; i <= 1024; i++) {
      const d = (i - 512) / 512;
      lut[i] = Math.exp(-(d * d) / (2 * sigmaR * sigmaR));
    }
    const sw = new Float32Array(25);
    let k = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++)
      sw[k++] = Math.exp(-(dx * dx + dy * dy) / (2 * 1.4 * 1.4));

    const out = new Float32Array(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = src[y * w + x];
        let ws = 0, vs = 0; k = 0;
        for (let dy = -r; dy <= r; dy++) {
          const yy = clampI(y + dy, 0, h - 1) * w;
          for (let dx = -r; dx <= r; dx++, k++) {
            const v = src[yy + clampI(x + dx, 0, w - 1)];
            let di = ((v - c) * 512 + 512) | 0;
            di = di < 0 ? 0 : (di > 1024 ? 1024 : di);
            const wt = sw[k] * lut[di];
            ws += wt; vs += wt * v;
          }
        }
        out[y * w + x] = vs / ws;
      }
    }
    return out;
  }

  /* ---------- equalize histogram ---------- */
  function eqLUTFrom(src, n) {
    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i++) hist[clampI((src[i] * 255) | 0, 0, 255)]++;
    let cdf = 0; const cdfv = new Uint32Array(256);
    for (let i = 0; i < 256; i++) { cdf += hist[i]; cdfv[i] = cdf; }
    let first = 0; while (first < 256 && hist[first] === 0) first++;
    const cdfMin = cdfv[first], denom = Math.max(1, n - cdfMin);
    const lut = new Uint8Array(256);
    for (let i = 0; i < 256; i++) lut[i] = Math.round(((cdfv[i] - cdfMin) / denom) * 255);
    return lut;
  }

  /* ---------- LUT akhir: equalize → levels → gamma → brightness → kontras → invert ---------- */
  function buildLUT(p, eq) {
    const lut = new Float32Array(256);
    const range = Math.max(1, p.inWhite - p.inBlack);
    const cf = Math.tan(((clampI(p.contrast, -98, 98) + 100) / 200) * Math.PI / 2);
    const br = p.brightness / 200;
    const g = 1 / Math.max(0.05, p.gamma);
    for (let i = 0; i < 256; i++) {
      let v = (eq ? eq[i] : i) / 255;
      v = clampI((v * 255 - p.inBlack) / range, 0, 1);
      v = (v * (p.outWhite - p.outBlack) + p.outBlack) / 255;
      v = cl01(v);
      v = Math.pow(v, g);
      v = v + br;
      v = cf * (v - 0.5) + 0.5;
      v = cl01(v);
      if (p.invert) v = 1 - v;
      lut[i] = v;
    }
    return lut;
  }

  /* ---------- proses utama ---------- */
  function processData(imgData, p, opts = {}) {
    const t0 = performance.now();
    const w = imgData.width, h = imgData.height, n = w * h;

    let cur;
    if (opts.base) {
      cur = Float32Array.from(opts.base);            // depth neural (0..1)
    } else {
      cur = new Float32Array(n);
      toBase(imgData, p.channel, cur);
    }

    const blurR = Math.round(p.blur);
    const detR = clampI(Math.round(p.detailRadius), 1, 20);
    const low = blurR >= 1 ? boxBlur(cur, w, h, blurR) : cur;
    let shape = low;

    if (p.detail > 0) {                               // unsharp dua skala
      const k = p.detail / 50;
      const small = boxBlur(cur, w, h, detR);
      shape = new Float32Array(n);
      for (let i = 0; i < n; i++)
        shape[i] = low[i] + k * ((small[i] - low[i]) + 0.55 * (cur[i] - small[i]));
    }

    if (p.smooth > 0)
      shape = bilateral(shape, w, h, 0.03 + (p.smooth / 100) * 0.30);

    const eq  = p.equalize ? eqLUTFrom(shape, n) : null;
    const lut = buildLUT(p, eq);

    const depth8 = new Uint8ClampedArray(n);
    const depth16 = new Uint16Array(n);
    const hist = new Uint32Array(48);
    let mn = 1, mx = 0, sum = 0;

    for (let i = 0; i < n; i++) {
      const v = lut[cl01(shape[i]) * 255 | 0];
      const v16 = (v * 65535 + 0.5) | 0;
      depth16[i] = v16;
      depth8[i] = v16 >> 8;
      hist[Math.min(47, (v * 48) | 0)]++;
      if (v < mn) mn = v; if (v > mx) mx = v;
      sum += v;
    }

    return {
      depth8, depth16, w, h, hist,
      stats: { min: mn, max: mx, mean: sum / n },
      ms: performance.now() - t0
    };
  }

  /* ---------- relief: normal + pencahayaan Lambert+specular ---------- */
  function reliefNormals(depth8, w, h, strength) {
    const f = new Float32Array(w * h);
    for (let i = 0; i < f.length; i++) f[i] = depth8[i] / 255;
    const nx = new Float32Array(w * h), ny = new Float32Array(w * h), nz = new Float32Array(w * h);
    const s = strength * 1.6;
    for (let y = 0; y < h; y++) {
      const ym = (y > 0 ? y - 1 : 0) * w, yp = (y < h - 1 ? y + 1 : h - 1) * w, y0 = y * w;
      for (let x = 0; x < w; x++) {
        const xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
        const gx = (f[y0 + xp] - f[y0 + xm]) * s;
        const gy = (f[yp + x] - f[ym + x]) * s;
        const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
        const i = y0 + x;
        nx[i] = -gx * inv; ny[i] = -gy * inv; nz[i] = inv;
      }
    }
    return { nx, ny, nz };
  }

  const MATERIALS = {
    gypsum:   { kind: 'flat', v: 214 },
    graphite: { kind: 'flat', v: 78 },
    copper:   { kind: 'rgb', r: 205, g: 128, b: 82 },
    source:   { kind: 'src' }
  };

  function relight(norm, w, h, light, material, srcData) {
    const len = Math.hypot(light.x, light.y, light.z) || 1;
    const Lx = light.x / len, Ly = light.y / len, Lz = light.z / len;
    const n = w * h, out = new ImageData(w, h), d = out.data;
    const { nx, ny, nz } = norm;
    const mat = MATERIALS[material] || MATERIALS.gypsum;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      let diff = nx[i] * Lx + ny[i] * Ly + nz[i] * Lz;
      if (diff < 0) diff = 0;
      const a = 0.14 + 0.92 * diff;
      const sp = diff > 0 ? Math.pow(diff, 22) * 0.45 * 255 : 0;
      let r, g, b;
      if (mat.kind === 'src' && srcData) { r = srcData[p]; g = srcData[p + 1]; b = srcData[p + 2]; }
      else if (mat.kind === 'flat') { r = g = b = mat.v; }
      else { r = mat.r; g = mat.g; b = mat.b; }
      d[p]     = Math.min(255, r * a + sp);
      d[p + 1] = Math.min(255, g * a + sp);
      d[p + 2] = Math.min(255, b * a + sp);
      d[p + 3] = 255;
    }
    return out;
  }

  return { DEFAULTS, processData, reliefNormals, relight };
})();