/* DepthForge · viewer.js
   Viewport kanvas: mode Depth / Relief 3D / Kontur / Banding.
   Interaksi: seret = arah cahaya (relief) atau geser, roda = zum,
   dobel-klik = pas layar, animasi "scan reveal" saat gambar baru dimuat. */
class Viewer {
  constructor(canvas) {
    this.cv = canvas;
    this.cx = canvas.getContext('2d');
    this.mode = 'depth';
    this.colormap = 'grayscale';
    this.material = 'gypsum';
    this.reliefStrength = 2.2;
    this.contourCount = 14;
    this.light = { x: -0.6, y: -0.55, z: 0.62 };
    this.t = { s: 1, ox: 0, oy: 0, fit: true };
    this.splitX = 0.5;
    this.reveal = 1;
    this.onZoom = null;
    this.srcCanvas = null; this.srcPixels = null;
    this.depth = null; this.dw = 0; this.dh = 0;
    this.norm = null;
    this.depthCv = null; this.reliefCv = null;
    this.contourPaths = null;
    this.dirtyDepth = this.dirtyNorm = this.dirtyRelight = this.dirtyContour = true;
    this._lut = null; this._lutName = '';
    this._drag = null; this._raf = 0;
    this._bind();
  }

  /* ---------- input data ---------- */
  setSource(canvas, pixels) { this.srcCanvas = canvas; this.srcPixels = pixels; }
  newScan() { this.reveal = 0; this.t.fit = true; this._startReveal(); }
  setDepth(depth8, w, h) {
    this.depth = depth8; this.dw = w; this.dh = h;
    this.dirtyDepth = this.dirtyNorm = this.dirtyRelight = this.dirtyContour = true;
    this.invalidate();
  }
  setColormap(c) { this.colormap = c; this.dirtyDepth = true; this.invalidate(); }
  setMaterial(m) { this.material = m; this.dirtyRelight = true; this.invalidate(); }
  setStrength(s) { this.reliefStrength = s; this.dirtyNorm = true; this.invalidate(); }
  setContours(n) { this.contourCount = n; this.dirtyContour = true; this.invalidate(); }
  fit() { this.t.fit = true; this.invalidate(); }
  resetLight() { this.light = { x: -0.6, y: -0.55, z: 0.62 }; this.dirtyRelight = true; this.invalidate(); }
  invalidate() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); }); }

  /* ---------- peta warna ---------- */
  _colorLUT(name) {
    const STOPS = {
      grayscale: [[0,0,0],[255,255,255]],
      viridis: [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
      magma: [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,253,191]],
      iron: [[0,0,0],[120,0,0],[230,60,0],[255,180,0],[255,255,220]],
      topo: [[38,70,110],[56,130,114],[122,158,88],[188,148,92],[245,245,240]]
    };
    const s = STOPS[name] || STOPS.grayscale, lut = new Uint8Array(768);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (s.length - 1), k = Math.min(s.length - 2, Math.floor(t)), f = t - k;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = s[k][c] + (s[k + 1][c] - s[k][c]) * f;
    }
    return lut;
  }

  /* ---------- pembuatan lapisan (malas) ---------- */
  _ensureDepthCv() {
    if (!this.dirtyDepth && this.depthCv) return;
    if (!this._lut || this._lutName !== this.colormap) { this._lut = this._colorLUT(this.colormap); this._lutName = this.colormap; }
    if (!this.depthCv) this.depthCv = document.createElement('canvas');
    if (this.depthCv.width !== this.dw) this.depthCv.width = this.dw;
    if (this.depthCv.height !== this.dh) this.depthCv.height = this.dh;
    const cx = this.depthCv.getContext('2d');
    const img = cx.createImageData(this.dw, this.dh), d = img.data, dep = this.depth, lut = this._lut;
    for (let i = 0, p = 0; i < dep.length; i++, p += 4) {
      const k = dep[i] * 3;
      d[p] = lut[k]; d[p + 1] = lut[k + 1]; d[p + 2] = lut[k + 2]; d[p + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    this.dirtyDepth = false;
  }
  _ensureRelief() {
    if (!this.dirtyNorm && !this.dirtyRelight && this.reliefCv) return;
    if (this.dirtyNorm || !this.norm)
      this.norm = Engine.reliefNormals(this.depth, this.dw, this.dh, this.reliefStrength);
    if (!this.reliefCv) this.reliefCv = document.createElement('canvas');
    if (this.reliefCv.width !== this.dw) this.reliefCv.width = this.dw;
    if (this.reliefCv.height !== this.dh) this.reliefCv.height = this.dh;
    const img = Engine.relight(this.norm, this.dw, this.dh, this.light, this.material, this.srcPixels && this.srcPixels.data);
    this.reliefCv.getContext('2d').putImageData(img, 0, 0);
    this.dirtyNorm = this.dirtyRelight = false;
  }
  _ensureContour() {
    if (!this.dirtyContour && this.contourPaths) return;
    const f = new Float32Array(this.dw * this.dh);
    for (let i = 0; i < f.length; i++) f[i] = this.depth[i] / 255;
    const L = this.contourCount;
    this.contourPaths = [];
    for (let li = 0; li < L; li++)
      this.contourPaths.push(this._marchingSquares(f, this.dw, this.dh, (li + 0.5) / L));
    this.dirtyContour = false;
  }
  _t(a, b, lv) { const d = b - a; return Math.abs(d) < 1e-9 ? 0.5 : Math.max(0, Math.min(1, (lv - a) / d)); }
  _marchingSquares(f, w, h, lv) {
    const path = new Path2D();
    for (let y = 0; y < h - 1; y++) {
      const r0 = y * w, r1 = (y + 1) * w;
      for (let x = 0; x < w - 1; x++) {
        const a = f[r0 + x], b = f[r0 + x + 1], c = f[r1 + x + 1], d = f[r1 + x];
        const idx = (a > lv ? 1 : 0) | (b > lv ? 2 : 0) | (c > lv ? 4 : 0) | (d > lv ? 8 : 0);
        if (idx === 0 || idx === 15) continue;
        const top = () => [x + this._t(a, b, lv), y];
        const right = () => [x + 1, y + this._t(b, c, lv)];
        const bottom = () => [x + this._t(d, c, lv), y + 1];
        const left = () => [x, y + this._t(a, d, lv)];
        const P = {
          1:[left,top], 14:[left,top], 2:[top,right], 13:[top,right],
          3:[left,right], 12:[left,right], 4:[right,bottom], 11:[right,bottom],
          6:[top,bottom], 9:[top,bottom], 7:[left,bottom], 8:[left,bottom],
          5:[left,top,right,bottom], 10:[top,right,left,bottom]
        }[idx];
        for (let s = 0; s < P.length; s += 2) {
          const p1 = P[s](), p2 = P[s + 1]();
          path.moveTo(p1[0], p1[1]); path.lineTo(p2[0], p2[1]);
        }
      }
    }
    return path;
  }

  /* ---------- render ---------- */
  render() {
    const cv = this.cv, cx = this.cx;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    const pw = (W * dpr) | 0, ph = (H * dpr) | 0;
    if (cv.width !== pw || cv.height !== ph) { cv.width = pw; cv.height = ph; }
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, W, H);
    if (!this.depth) return;
    if (this.t.fit) this._applyFit(W, H);

    const world = () => { cx.translate(this.t.ox, this.t.oy); cx.scale(this.t.s, this.t.s); };

    if (this.reveal < 1) {
      cx.save(); cx.beginPath(); cx.rect(0, 0, W * this.reveal, H); cx.clip();
      cx.save(); world(); this._drawMode(cx); cx.restore(); cx.restore();
      cx.save(); cx.beginPath(); cx.rect(W * this.reveal, 0, W - W * this.reveal, H); cx.clip();
      cx.save(); world(); cx.drawImage(this.srcCanvas, 0, 0); cx.restore(); cx.restore();
      const lx = W * this.reveal;
      cx.strokeStyle = '#E09A3E'; cx.lineWidth = 1.5;
      cx.beginPath(); cx.moveTo(lx, 0); cx.lineTo(lx, H); cx.stroke();
      cx.font = '600 11px "JetBrains Mono", monospace';
      const label = 'PEMINDAIAN ' + (this.reveal * 100).toFixed(0) + '%';
      const tw = cx.measureText(label).width;
      cx.fillStyle = '#161412'; cx.fillRect(lx + 10, 14, tw + 16, 22);
      cx.strokeStyle = '#3A3327'; cx.strokeRect(lx + 10.5, 14.5, tw + 16, 22);
      cx.fillStyle = '#E09A3E'; cx.fillText(label, lx + 18, 29);
    } else {
      cx.save(); world(); this._drawMode(cx); cx.restore();
    }
  }

  _drawMode(cx) {
    if (this.mode === 'relief') {
      this._ensureRelief();
      cx.drawImage(this.reliefCv, 0, 0);
    } else if (this.mode === 'contour') {
      this._ensureContour();
      cx.fillStyle = '#191613'; cx.fillRect(0, 0, this.dw, this.dh);
      cx.globalAlpha = 0.16; cx.drawImage(this.depthCv, 0, 0); cx.globalAlpha = 1;
      const lw = 1 / this.t.s;
      for (let i = 0; i < this.contourPaths.length; i++) {
        const major = i % 5 === 0;
        cx.strokeStyle = major ? '#E09A3E' : '#8F8574';
        cx.lineWidth = major ? lw * 1.6 : lw;
        cx.stroke(this.contourPaths[i]);
      }
    } else if (this.mode === 'split') {
      this._ensureDepthCv();
      cx.drawImage(this.srcCanvas, 0, 0);
      const hx = this.splitX * this.dw;
      cx.save(); cx.beginPath(); cx.rect(hx, 0, this.dw - hx, this.dh); cx.clip();
      cx.drawImage(this.depthCv, 0, 0); cx.restore();
      cx.strokeStyle = '#E09A3E'; cx.lineWidth = 1.5 / this.t.s;
      cx.beginPath(); cx.moveTo(hx, 0); cx.lineTo(hx, this.dh); cx.stroke();
      const r = 9 / this.t.s;
      cx.beginPath(); cx.arc(hx, this.dh / 2, r, 0, Math.PI * 2);
      cx.fillStyle = '#161412'; cx.fill(); cx.stroke();
      cx.beginPath(); cx.moveTo(hx - r * 0.35, this.dh / 2); cx.lineTo(hx + r * 0.35, this.dh / 2); cx.stroke();
    } else {
      this._ensureDepthCv();
      cx.drawImage(this.depthCv, 0, 0);
    }
  }

  _applyFit(W, H) {
    const pad = 56;
    const s = Math.min((W - pad) / this.dw, (H - pad) / this.dh);
    this.t.s = s;
    this.t.ox = (W - this.dw * s) / 2;
    this.t.oy = (H - this.dh * s) / 2;
    this.t.fit = false;
    if (this.onZoom) this.onZoom(s);
  }

  /* ---------- animasi scan ---------- */
  _startReveal() {
    const t0 = performance.now();
    const step = (t) => {
      let r = Math.min(1, (t - t0) / 1100);
      this.reveal = r * r * (3 - 2 * r);
      this.invalidate();
      if (this.reveal < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------- interaksi ---------- */
  _pos(e) { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  _nearSplit(p) {
    const sx = this.t.ox + this.splitX * this.dw * this.t.s;
    return Math.abs(p.x - sx) < 14;
  }
  _setLight(p) {
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    const dx = (p.x - W / 2) / (W / 2), dy = (p.y - H / 2) / (H / 2), z = 0.55;
    const len = Math.hypot(dx, dy, z) || 1;
    this.light = { x: dx / len, y: -dy / len, z: z / len };
    if (this.norm) {
      this._ensureRelief(); // normal siap → cukup relight
      const img = Engine.relight(this.norm, this.dw, this.dh, this.light, this.material, this.srcPixels && this.srcPixels.data);
      this.reliefCv.getContext('2d').putImageData(img, 0, 0);
    }
    this.invalidate();
  }
  _bind() {
    const cv = this.cv;
    cv.addEventListener('pointerdown', e => {
      cv.setPointerCapture(e.pointerId);
      const p = this._pos(e);
      if (!this.depth) return;
      if (this.mode === 'relief') { this._drag = 'light'; this._setLight(p); }
      else if (this.mode === 'split' && this._nearSplit(p)) { this._drag = 'split'; }
      else { this._drag = 'pan'; this._pan = { x: e.clientX, y: e.clientY, ox: this.t.ox, oy: this.t.oy }; cv.style.cursor = 'grabbing'; }
    });
    cv.addEventListener('pointermove', e => {
      const p = this._pos(e);
      if (this._drag === 'light') this._setLight(p);
      else if (this._drag === 'split') {
        const ix = (p.x - this.t.ox) / this.t.s / this.dw;
        this.splitX = Math.max(0.05, Math.min(0.95, ix));
        this.invalidate();
      } else if (this._drag === 'pan') {
        this.t.ox = this._pan.ox + (e.clientX - this._pan.x);
        this.t.oy = this._pan.oy + (e.clientY - this._pan.y);
        this.t.fit = false;
        this.invalidate();
      } else {
        cv.style.cursor =
          this.mode === 'relief' ? 'crosshair' :
          (this.mode === 'split' && this._nearSplit(p)) ? 'ew-resize' : 'grab';
      }
    });
    const end = () => { this._drag = null; this.cv.style.cursor = 'grab'; };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointerleave', end);
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      if (!this.depth) return;
      const p = this._pos(e);
      const k = Math.exp(-e.deltaY * 0.0012);
      const ns = Math.min(10, Math.max(0.05, this.t.s * k));
      const f = ns / this.t.s;
      this.t.ox = p.x - (p.x - this.t.ox) * f;
      this.t.oy = p.y - (p.y - this.t.oy) * f;
      this.t.s = ns; this.t.fit = false;
      if (this.onZoom) this.onZoom(ns);
      this.invalidate();
    }, { passive: false });
    cv.addEventListener('dblclick', () => this.fit());
  }
}