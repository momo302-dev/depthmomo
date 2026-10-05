/* DepthForge · app.js — state, kontrol UI, ekspor, integrasi server Python. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const SERVER = 'http://127.0.0.1:8787';

  const params = { ...Engine.DEFAULTS };
  const view = { colormap: 'grayscale', material: 'gypsum', relief: 2.2, contours: 14 };
  const state = {
    file: null, fileName: '', fullCanvas: null, fullCtx: null,
    previewCanvas: null, previewData: null,
    neuralActive: false, neuralCanvas: null, server: null
  };

  const viewer = new Viewer($('#canvas'));
  const dropzone = $('#dropzone'), srcBadge = $('#srcBadge');
  const segChannel = $('#segChannel');
  const sliders = {};
  let _t = 0;

  /* ============================ toast ============================ */
  function toast(msg, kind = 'ok', ms = 3200) {
    const el = document.createElement('div');
    el.className = 'toast ' + kind;
    el.textContent = msg;
    $('#toasts').appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    let done = false;
    const dismiss = () => {
      if (done) return; done = true;
      el.classList.remove('show');
      setTimeout(() => el.remove(), 250);
    };
    if (ms > 0) setTimeout(dismiss, ms);
    return { dismiss };
  }

  /* ============================ slider ============================ */
  function buildSlider(cfg, mount, obj, onChange) {
    const scaled = cfg.scale || 1;
    const wrap = document.createElement('div');
    wrap.className = 'ctl';
    wrap.innerHTML = '<div class="ctl-h"><label>' + cfg.label + '</label><output></output></div>';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = cfg.min; inp.max = cfg.max; inp.step = cfg.step || 1;
    const out = wrap.querySelector('output');
    const fmt = cfg.fmt || (v => String(v));
    const show = () => { out.textContent = fmt(inp.value); };
    inp.value = Math.round(obj[cfg.key] * scaled);
    show();
    inp.addEventListener('input', () => {
      obj[cfg.key] = (+inp.value) / scaled;
      show();
      onChange && onChange(cfg.key);
    });
    mount.appendChild(wrap);
    return { key: cfg.key, refresh() { inp.value = Math.round(obj[cfg.key] * scaled); show(); } };
  }

  const PIPE_SLIDERS = [
    { key: 'blur',          label: 'Blur dasar',          min: 0, max: 24 },
    { key: 'detail',        label: 'Detail / ketajaman',  min: 0, max: 100, fmt: v => v + '%' },
    { key: 'detailRadius',  label: 'Radius detail',       min: 1, max: 12, fmt: v => v + ' px' },
    { key: 'smooth',        label: 'Halus (edge-aware)',  min: 0, max: 100, fmt: v => v + '%' },
    { key: 'contrast',      label: 'Kontras',             min: -100, max: 100 },
    { key: 'brightness',    label: 'Kecerahan',           min: -100, max: 100 },
    { key: 'gamma',         label: 'Gamma',               min: 20, max: 300, scale: 100, fmt: v => (v / 100).toFixed(2) },
    { key: 'inBlack',       label: 'Titik hitam (in)',    min: 0, max: 254 },
    { key: 'inWhite',       label: 'Titik putih (in)',    min: 1, max: 255 },
    { key: 'outBlack',      label: 'Keluaran hitam',      min: 0, max: 255 },
    { key: 'outWhite',      label: 'Keluaran putih',      min: 0, max: 255 }
  ];
  const VIEW_SLIDERS = [
    { key: 'relief',   label: 'Kekuatan relief', min: 2, max: 80, scale: 10, fmt: v => (v / 10).toFixed(1) },
    { key: 'contours', label: 'Jumlah kontur',   min: 3, max: 40 }
  ];

  PIPE_SLIDERS.forEach(cfg => sliders[cfg.key] = buildSlider(cfg, $('#slidersPipeline'), params, () => schedule()));
  VIEW_SLIDERS.forEach(cfg => buildSlider(cfg, $('#slidersView'), view, k => {
    if (k === 'relief') viewer.setStrength(view.relief);
    if (k === 'contours') viewer.setContours(view.contours);
  }));

  function refreshSliders() { Object.values(sliders).forEach(s => s.refresh()); }

  /* ============================ proses ============================ */
  function schedule() { clearTimeout(_t); _t = setTimeout(processNow, 70); }

  function processNow() {
    if (!state.previewData) return;
    const t0 = performance.now();
    const res = Engine.processData(state.previewData, params, { base: state.neuralBase });
    viewer.setDepth(res.depth8, res.w, res.h);
    $('#stFile').textContent = state.fileName || '—';
    $('#stDims').textContent = res.w + '×' + res.h;
    $('#stRange').textContent =
      'min ' + (res.stats.min * 100).toFixed(0) +
      ' · maks ' + (res.stats.max * 100).toFixed(0) +
      ' · rerata ' + (res.stats.mean * 100).toFixed(0);
    $('#stMs').textContent = (performance.now() - t0).toFixed(0) + ' ms';
    drawHist(res.hist);
  }

  function drawHist(hist) {
    const c = $('#hist'), cx = c.getContext('2d');
    cx.clearRect(0, 0, c.width, c.height);
    let mx = 1; for (const v of hist) if (v > mx) mx = v;
    cx.fillStyle = 'rgba(252,238,10,.75)';
    const bw = c.width / hist.length;
    for (let i = 0; i < hist.length; i++) {
      const h = Math.pow(hist[i] / mx, 0.5) * (c.height - 2);
      cx.fillRect(i * bw, c.height - h, Math.max(1, bw - 1.2), h);
    }
  }

  /* ============================ muat gambar ============================ */
  function loadImage(blob, name) {
    if (!blob || !blob.type.startsWith('image/'))
      return toast('Format tidak didukung — gunakan JPG/PNG.', 'err');
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      const fs = Math.min(1, Math.sqrt(16e6 / (W * H)));       // batas 16 MP untuk ekspor
      const fc = document.createElement('canvas');
      fc.width = Math.max(1, Math.round(W * fs));
      fc.height = Math.max(1, Math.round(H * fs));
      const fctx = fc.getContext('2d', { willReadFrequently: true });
      fctx.imageSmoothingQuality = 'high';
      fctx.drawImage(img, 0, 0, fc.width, fc.height);

      const ps = Math.min(1, 1200 / Math.max(fc.width, fc.height));
      const pc = document.createElement('canvas');
      pc.width = Math.max(1, Math.round(fc.width * ps));
      pc.height = Math.max(1, Math.round(fc.height * ps));
      const pctx = pc.getContext('2d', { willReadFrequently: true });
      pctx.imageSmoothingQuality = 'high';
      pctx.drawImage(fc, 0, 0, pc.width, pc.height);

      state.file = blob; state.fileName = name || 'gambar';
      state.fullCanvas = fc; state.fullCtx = fctx;
      state.previewCanvas = pc;
      state.previewData = pctx.getImageData(0, 0, pc.width, pc.height);
      state.neuralActive = false; state.neuralBase = null;
      setNeuralBtn(false);
      segChannel.classList.remove('off');

      viewer.setSource(pc, state.previewData);
      viewer.newScan();
      dropzone.classList.add('hide');
      updateBadge(W, H, fs);
      processNow();
      URL.revokeObjectURL(url);
    };
    img.onerror = () => toast('Gagal membaca gambar.', 'err');
    img.src = url;
  }

  function updateBadge(W, H, fs) {
    srcBadge.textContent = state.fileName + ' · ' + W + '×' + H +
      (fs < 1 ? ' → ' + state.fullCanvas.width + '×' + state.fullCanvas.height : '') +
      (state.neuralActive ? ' · neural MiDaS' : '');
    srcBadge.classList.add('show');
  }

  /* contoh: picsum */
  async function loadSample() {
    const seeds = ['basalt', 'kanion', 'kabut', 'dune', 'rimba', 'fjord'];
    const seed = seeds[(Math.random() * seeds.length) | 0];
    toast('Mengambil gambar contoh…', 'info', 1600);
    try {
      const r = await fetch('https://picsum.photos/seed/' + seed + '/1400/950.jpg');
      const b = await r.blob();
      loadImage(b, 'contoh-' + seed + '.jpg');
    } catch {
      toast('Gagal mengambil contoh (butuh koneksi internet).', 'err');
    }
  }

  /* ============================ input berkas ============================ */
  const fileInput = $('#fileInput');
  const pick = () => fileInput.click();
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) loadImage(fileInput.files[0], fileInput.files[0].name);
    fileInput.value = '';
  });
  $('#btnPick').addEventListener('click', pick);
  $('#dzPick').addEventListener('click', e => { e.stopPropagation(); pick(); });
  $('#btnSample').addEventListener('click', loadSample);
  $('#dzSample').addEventListener('click', e => { e.stopPropagation(); loadSample(); });
  dropzone.addEventListener('click', pick);

  const vp = $('#viewport');
  ['dragenter', 'dragover'].forEach(ev => vp.addEventListener(ev, e => { e.preventDefault(); vp.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev => vp.addEventListener(ev, e => { e.preventDefault(); vp.classList.remove('drag'); }));
  vp.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) loadImage(f, f.name);
  });
  window.addEventListener('paste', e => {
    const f = e.clipboardData.files[0];
    if (f && f.type.startsWith('image/')) loadImage(f, f.name || 'tempelan.png');
  });

  /* ============================ kontrol pipeline ============================ */
  function setParam(k, v) {
    params[k] = v;
    if (['inBlack', 'inWhite', 'outBlack', 'outWhite'].includes(k)) {
      params.inBlack = Math.min(params.inBlack, params.inWhite - 1);
      params.inWhite = Math.max(params.inWhite, params.inBlack + 1);
      params.outBlack = Math.min(params.outBlack, params.outWhite - 1);
      params.outWhite = Math.max(params.outWhite, params.outBlack + 1);
      ['inBlack', 'inWhite', 'outBlack', 'outWhite'].forEach(kk => sliders[kk] && sliders[kk].refresh());
    }
    schedule();
  }

  segChannel.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    segChannel.querySelectorAll('button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    setParam('channel', b.dataset.v);
  }));

  $('#tglEq').addEventListener('change', e => { params.equalize = e.target.checked; schedule(); });
  $('#tglInv').addEventListener('change', e => { params.invert = e.target.checked; schedule(); });

  const PRESETS = {
    foto:   { blur: 1, detail: 40, detailRadius: 2, smooth: 8,  equalize: false, contrast: 14, brightness: 0, gamma: 1.05, inBlack: 0,  inWhite: 255, outBlack: 0, outWhite: 255, invert: false },
    gravir: { blur: 2, detail: 70, detailRadius: 1, smooth: 0,  equalize: true,  contrast: 28, brightness: 0, gamma: 1.20, inBlack: 10, inWhite: 245, outBlack: 0, outWhite: 255, invert: false },
    halus:  { blur: 7, detail: 0,  detailRadius: 3, smooth: 60, equalize: false, contrast: 6,  brightness: 0, gamma: 1.00, inBlack: 0,  inWhite: 255, outBlack: 0, outWhite: 255, invert: false }
  };
  $('#presets').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    Object.assign(params, PRESETS[b.dataset.p]);
    $('#tglEq').checked = params.equalize;
    $('#tglInv').checked = params.invert;
    refreshSliders();
    schedule();
  }));

  $('#btnReset').addEventListener('click', () => {
    Object.assign(params, Engine.DEFAULTS);
    $('#tglEq').checked = false;
    $('#tglInv').checked = false;
    refreshSliders();
    schedule();
  });

  /* ============================ tampilan ============================ */
  $('#selCmap').addEventListener('change', e => viewer.setColormap(e.target.value));
  $('#selMat').addEventListener('change', e => viewer.setMaterial(e.target.value));
  $('#btnFit').addEventListener('click', () => viewer.fit());
  $('#btnZoomFit').addEventListener('click', () => viewer.fit());
  $('#btnLight').addEventListener('click', () => viewer.resetLight());
  viewer.onZoom = s => { $('#zoomVal').textContent = Math.round(s * 100) + '%'; };

  const HINTS = {
    depth: 'roda = zum · seret = geser · dobel-klik = pas layar · tombol 1-4 ganti mode',
    relief: 'seret di kanvas = arah cahaya · roda = zum · tombol 1-4 ganti mode',
    contour: 'garis halus = interval kecil · garis kuning = garis mayor · tombol 1-4 ganti mode',
    split: 'seret lingkaran tengah untuk membandingkan asli vs depth · tombol 1-4 ganti mode'
  };
  function setMode(m) {
    document.querySelectorAll('#modes button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
    viewer.mode = m;
    $('#hint').textContent = HINTS[m];
    viewer.invalidate();
  }
  document.querySelectorAll('#modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  setMode('depth');

  window.addEventListener('keydown', e => {
    if (e.target.matches('input,select,textarea')) return;
    const m = { 1: 'depth', 2: 'relief', 3: 'contour', 4: 'split' }[e.key];
    if (m) setMode(m);
    if (e.key === 'f' || e.key === 'F') viewer.fit();
    if (e.key === 'l' || e.key === 'L') viewer.resetLight();
  });

  new ResizeObserver(() => viewer.invalidate()).observe(vp);

  /* ============================ server Python & neural ============================ */
  function setSrvUI(on, neural) {
    $('#srvDot').className = 'dot' + (on ? (neural ? ' mid' : ' on') : '');
    $('#srvLabel').textContent = 'server python: ' + (on ? (neural ? 'midas siap' : 'klasik') : 'mati');
    $('#srvState').textContent = on ? (neural ? 'MiDaS siap' : 'klasik saja') : 'server mati';
  }

  async function pingServer(silent) {
    try {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 2500);
      const r = await fetch(SERVER + '/api/ping', { signal: ctrl.signal });
      clearTimeout(to);
      const j = await r.json();
      state.server = j;
      setSrvUI(true, j.neural);
      if (!silent) toast('Server tersambung' + (j.neural ? ' — neural MiDaS siap.' : ' — mode klasik.'), 'ok');
      return j;
    } catch {
      state.server = null;
      setSrvUI(false, false);
      if (!silent) toast('Tidak menemukan server di ' + SERVER + '. Jalankan: python py/server.py', 'warn', 6000);
      return null;
    }
  }
  $('#btnServer').addEventListener('click', () => pingServer(false));

  function setNeuralBtn(active) {
    $('#btnNeural').innerHTML = active
      ? '<i data-lucide="undo-2"></i><span>Kembali ke klasik</span>'
      : '<i data-lucide="cpu"></i><span>Konversi Neural (MiDaS)</span>';
    window.lucide && lucide.createIcons();
  }

  $('#btnNeural').addEventListener('click', async () => {
    if (state.neuralActive) {
      state.neuralActive = false; state.neuralBase = null;
      setNeuralBtn(false);
      segChannel.classList.remove('off');
      updateBadge(state.fullCanvas ? 0 : 0, 0, 1); // refresh label
      srcBadge.textContent = srcBadge.textContent.replace(' · neural MiDaS', '');
      processNow();
      return;
    }
    if (!state.server && !(await pingServer(false))) return;
    if (!state.file) return toast('Muat gambar terlebih dahulu.', 'warn');

    const busy = toast('Mengirim ke server — model MiDaS memproses…', 'info', 0);
    try {
      const fd = new FormData();
      fd.append('file', state.file);
      fd.append('params', JSON.stringify({
        channel: params.channel, blur: params.blur, detail: params.detail,
        detail_radius: params.detailRadius, smooth: params.smooth, equalize: params.equalize,
        contrast: params.contrast, brightness: params.brightness, gamma: params.gamma,
        in_black: params.inBlack, in_white: params.inWhite,
        out_black: params.outBlack, out_white: params.outWhite, invert: params.invert
      }));
      const r = await fetch(SERVER + '/api/depth?model=midas', { method: 'POST', body: fd });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const blob = await r.blob();
      const bmp = await createImageBitmap(blob);

      const pc = document.createElement('canvas');
      pc.width = state.previewData.width; pc.height = state.previewData.height;
      const px = pc.getContext('2d', { willReadFrequently: true });
      px.drawImage(bmp, 0, 0, pc.width, pc.height);
      state.neuralCanvas = pc;

      const d = px.getImageData(0, 0, pc.width, pc.height).data;
      const base = new Float32Array(pc.width * pc.height);
      for (let i = 0, p = 0; i < base.length; i++, p += 4) base[i] = d[p] / 255;
      state.neuralBase = base;
      state.neuralActive = true;

      setNeuralBtn(true);
      segChannel.classList.add('off');
      srcBadge.textContent += ' · neural MiDaS';
      busy.dismiss();
      viewer.newScan();
      processNow();
      toast('Depth neural dimuat — parameterlevel & invert masih bisa diatur.', 'ok');
    } catch (err) {
      busy.dismiss();
      toast('Gagal memanggil server: ' + err.message, 'err', 6000);
    }
  });

  /* ============================ ekspor ============================ */
  const saveBlob = (blob, name) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  };
  const baseName = () => (state.fileName || 'gambar').replace(/\.[^.]+$/, '');

  function fullDepth() {
    const c = state.fullCanvas;
    const srcData = state.fullCtx.getImageData(0, 0, c.width, c.height);
    let base = null;
    if (state.neuralActive) base = neuralFullBase(c.width, c.height);
    return { ...Engine.processData(srcData, params, { base }), srcData };
  }

  function neuralFullBase(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d', { willReadFrequently: true });
    cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = 'high';
    cx.drawImage(state.neuralCanvas, 0, 0, w, h);
    const d = cx.getImageData(0, 0, w, h).data;
    const base = new Float32Array(w * h);
    for (let i = 0, p = 0; i < base.length; i++, p += 4) base[i] = d[p] / 255;
    return base;
  }

  function encodePGM16(depth16, w, h) {
    const head = new TextEncoder().encode('P5\n' + w + ' ' + h + '\n65535\n');
    const data = new Uint8Array(w * h * 2);
    for (let i = 0, j = 0; i < depth16.length; i++, j += 2) {
      data[j] = depth16[i] >>> 8;
      data[j + 1] = depth16[i] & 0xFF;
    }
    return new Blob([head, data], { type: 'application/octet-stream' });
  }

  /* --- encoder PNG 16-bit grayscale (streaming, CRC bertahap) --- */
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      t[n] = c >>> 0;
    }
    return t;
  })();
  const crcUpdate = (c, buf) => {
    for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
    return c >>> 0;
  };
  function pngChunk(type, data) {
    const out = new Uint8Array(12 + data.length);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    let c = 0xFFFFFFFF;
    c = crcUpdate(c, out.subarray(4, 8 + data.length));
    dv.setUint32(8 + data.length, (c ^ 0xFFFFFFFF) >>> 0);
    return out;
  }
  async function encodePNG16(depth16, w, h) {
    const stride = w * 2 + 1;
    const cs = new CompressionStream('deflate');
    const writer = cs.writable.getWriter();
    const chunks = [];
    let crcC = crcUpdate(0xFFFFFFFF, new TextEncoder().encode('IDAT'));
    let readDone = false, readErr = null;
    (async () => {
      const rd = cs.readable.getReader();
      try {
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          chunks.push(value);
          crcC = crcUpdate(crcC, value);
        }
      } catch (e) { readErr = e; }
      readDone = true;
    })();

    const row = new Uint8Array(stride);
    for (let y = 0; y < h; y++) {
      row[0] = 0;
      const b = y * w;
      for (let x = 0; x < w; x++) {
        const v = depth16[b + x];
        row[1 + x * 2] = v >>> 8;
        row[2 + x * 2] = v & 0xFF;
      }
      await writer.write(row);
      if ((y & 255) === 255) await new Promise(r => setTimeout(r));
    }
    await writer.close();
    while (!readDone) await new Promise(r => setTimeout(r, 5));
    if (readErr) throw readErr;

    const total = chunks.reduce((s, c) => s + c.length, 0);
    const head = new Uint8Array(8);
    new DataView(head.buffer).setUint32(0, total);
    head.set([0x49, 0x44, 0x41, 0x54], 4);
    const tail = new Uint8Array(4);
    new DataView(tail.buffer).setUint32(0, crcC);

    const ihdr = new Uint8Array(13);
    const dv = new DataView(ihdr.buffer);
    dv.setUint32(0, w); dv.setUint32(4, h);
    ihdr[8] = 16; // bit depth, color type 0 (grayscale)
    const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    return new Blob([sig, pngChunk('IHDR', ihdr), head, new Blob(chunks), tail, pngChunk('IEND', new Uint8Array(0))],
      { type: 'image/png' });
  }

  async function exportDepth(kind) {
    if (!state.fullCanvas) return toast('Muat gambar terlebih dahulu.', 'warn');
    const busy = toast('Memproses resolusi penuh… (gambar besar butuh beberapa detik)', 'info', 0);
    await new Promise(r => setTimeout(r, 40));
    try {
      const res = fullDepth();
      const name = baseName();
      if (kind === 'png8') {
        const c = document.createElement('canvas');
        c.width = res.w; c.height = res.h;
        c.getContext('2d').putImageData(new ImageData(res.depth8, res.w, res.h), 0, 0);
        saveBlob(await new Promise(r => c.toBlob(r, 'image/png')), name + '-depth8.png');
      } else if (kind === 'png16') {
        if (!('CompressionStream' in window)) throw new Error('tanpa CompressionStream');
        saveBlob(await encodePNG16(res.depth16, res.w, res.h), name + '-depth16.png');
      } else if (kind === 'pgm') {
        saveBlob(encodePGM16(res.depth16, res.w, res.h), name + '-depth16.pgm');
      } else if (kind === 'relief') {
        const norm = Engine.reliefNormals(res.depth8, res.w, res.h, view.relief);
        const img = Engine.relight(norm, res.w, res.h, viewer.light, view.material,
          view.material === 'source' ? res.srcData.data : null);
        const c = document.createElement('canvas');
        c.width = res.w; c.height = res.h;
        c.getContext('2d').putImageData(img, 0, 0);
        saveBlob(await new Promise(r => c.toBlob(r, 'image/png')), name + '-relief.png');
      }
      busy.dismiss();
      toast('Ekspor selesai — ' + res.w + '×' + res.h + ' · ' + kind.toUpperCase(), 'ok');
    } catch (err) {
      busy.dismiss();
      if (kind === 'png16') {
        toast('PNG 16-bit gagal di peramban ini — memakai PNG 8-bit.', 'warn', 5000);
        exportDepth('png8');
      } else {
        toast('Ekspor gagal: ' + err.message, 'err', 5000);
      }
    }
  }
  $('#expPng16').addEventListener('click', () => exportDepth('png16'));
  $('#expPng8').addEventListener('click', () => exportDepth('png8'));
  $('#expPgm').addEventListener('click', () => exportDepth('pgm'));
  $('#expRelief').addEventListener('click', () => exportDepth('relief'));

  /* ============================ init ============================ */
  window.lucide && lucide.createIcons();
  $('#hint').textContent = HINTS.depth;
  pingServer(true); // deteksi diam-diam
})();
