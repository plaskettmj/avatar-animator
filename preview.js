// Live motion preview for the avatar animator editor (step 3 of the live-preview plan).
// Animates float, breathe, tilt, blink and parallax on a Canvas 2D from the layers POST /preview writes, so the
// sliders update instantly with no server call. The motion math mirrors Animator.motion(), subject_matrix(),
// parallax_matrix() and blink_amount() in avatar_animator/core.py line for line (tests/test_live.py runs this file
// under node and checks the numbers match), so keep the two in step when changing either.
// Motions that need per-pixel flow or noise (swirl, hue, twinkle, particles, sparkles, sweep) are not animated
// here: the preview shows them as they look at t=0 and lists them as "frozen". Export renders everything.
(function (root) {
  "use strict";
  const LIVE = ["float", "breathe", "tilt", "blink", "parallax"];

  // cv2.getRotationMatrix2D((px, py), ang, sc) plus a translation: rows [[a, b, e], [c, d, f]] as [a, b, e, c, d, f]
  function affineAbout(px, py, angDeg, sc, tx, ty) {
    const r = angDeg * Math.PI / 180, a = sc * Math.cos(r), b = sc * Math.sin(r);
    return [a, b, (1 - a) * px - b * py + tx, -b, a, b * px + (1 - a) * py + ty];
  }
  // Animator.motion(t): subject pose (angle_deg, scale, tx, ty) about the pivot
  function motion(t, P, motions, style, out) {
    const tau = 2 * Math.PI * t;
    let ang = 0, sc = 1, tx = 0, ty = 0;
    if (motions.includes("float")) ty = -P.float_amp * out * (1 - Math.cos(tau)) / 2;
    if (motions.includes("breathe"))
      sc = style === "logo" ? 1 + P.breathe_amp * (0.5 - 0.5 * Math.cos(2 * tau)) : 1 + P.breathe_amp * Math.sin(tau) * 0.6;
    if (motions.includes("tilt")) ang = P.tilt_deg * Math.sin(tau + 0.6);
    return [ang, sc, tx, ty];
  }
  // Animator.subject_matrix(t), or null at rest
  function subjectMatrix(t, P, motions, cfg) {
    const [ang, sc, tx, ty] = motion(t, P, motions, cfg.style, cfg.out);
    if (ang === 0 && sc === 1 && tx === 0 && ty === 0) return null;
    return affineAbout(cfg.pivot[0], cfg.pivot[1], ang, sc, tx, ty);
  }
  // Animator.parallax_matrix(t)
  function parallaxMatrix(t, P, cfg) {
    const E = cfg.canvas, A = P.parallax_amp * cfg.out, tau = 2 * Math.PI * t;
    const M = affineAbout(E / 2, E / 2, 0, 1 + 2 * P.parallax_amp, 0, 0);
    M[2] += A * Math.cos(tau); M[5] += 0.6 * A * Math.sin(tau);
    return M;
  }
  // Animator.blink_amount(t)
  function blinkAmount(t, P) {
    const u = (t - P.blink_at) / P.blink_len;
    return 0 < u && u < 1 ? Math.pow(Math.sin(Math.PI * u), 0.6) : 0;
  }
  // cv2 warps treat pixel centres as integers, canvas as integers + 0.5: shift so the two line up exactly
  function setCv2Transform(ctx, M) {
    const [a, b, e, c, d, f] = M;
    ctx.transform(a, c, b, d, e + 0.5 - 0.5 * (a + b), f + 0.5 - 0.5 * (c + d));
  }
  function loadImage(url) {
    return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error("could not load " + url)); im.src = url; });
  }
  function canvasOf(w, h) {
    const c = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), {width: w, height: h});
    return c;
  }

  class LivePreview {
    constructor(canvas, mirrors) {
      this.cv = canvas; this.ctx = canvas.getContext("2d"); this.mirrors = mirrors || [];
      this.cfg = null; this.P = {}; this.motions = []; this.seconds = 3; this.fps = 20;
      this.playing = false; this.t = 0; this.start0 = 0; this.raf = 0;
      this.stats = {renders: 0, last_ms: 0, avg_ms: 0}; this.onframe = null;
    }
    // cfg and urls from POST /preview. Keeps the current slider values unless keepParams is false.
    async load(cfg, urls, keepParams) {
      const [bg, subj, blink] = await Promise.all([loadImage(urls.background), loadImage(urls.subject),
                                                   urls.blink ? loadImage(urls.blink) : Promise.resolve(null)]);
      const E = cfg.canvas, levels = [];
      const base = canvasOf(E, E); base.getContext("2d").drawImage(subj, 0, 0); levels.push(base);
      if (blink && cfg.blink) {
        const [x0, y0, x1, y1] = cfg.blink.box, w = x1 - x0, h = y1 - y0;
        for (let k = 0; k < cfg.blink.levels; k++) {
          const c = canvasOf(E, E), g = c.getContext("2d");
          g.drawImage(subj, 0, 0); g.clearRect(x0, y0, w, h); g.drawImage(blink, k * w, 0, w, h, x0, y0, w, h);
          levels.push(c);
        }
      }
      this.cfg = cfg; this.bg = bg; this.levels = levels;
      if (!keepParams || !Object.keys(this.P).length) {
        this.P = Object.assign({}, cfg.params); this.seconds = cfg.seconds; this.fps = cfg.fps;
        this.motions = cfg.live.slice();
      }
      this.render();
    }
    // slider / toggle changes: redraw now when paused, on the next animation frame when playing
    set(key, value) { this.P[key] = value; this.dirty = true; if (!this.playing) this.render(); }
    setMotion(m, on) {
      this.motions = this.motions.filter(x => x !== m); if (on) this.motions.push(m);
      this.dirty = true; if (!this.playing) this.render();
    }
    setSeconds(s) { this.seconds = s; this.dirty = true; if (this.playing) this.play(true); else this.render(this.quantize(this.t)); }
    frames() { return Math.max(1, Math.round(this.fps * this.seconds)); }
    // time as the export samples it: frame k of N is t = k / N
    quantize(t) { const N = this.frames(); return (Math.floor(((t % 1) + 1) % 1 * N + 1e-9) % N) / N; }
    // draw loop time t in [0, 1) (already quantized when it comes from the clock)
    render(t) {
      if (t !== undefined) this.t = t;
      const cfg = this.cfg, ctx = this.ctx;
      if (!cfg) return;
      const t0 = performance.now(), s = this.cv.width / cfg.out, P = this.P, t1 = this.t;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, this.cv.width, this.cv.height);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.setTransform(s, 0, 0, s, -cfg.off * s, -cfg.off * s);
      ctx.save();
      if (this.motions.includes("parallax")) setCv2Transform(ctx, parallaxMatrix(t1, P, cfg));
      ctx.drawImage(this.bg, 0, 0);
      ctx.restore();
      if (cfg.has_subject) {
        let layer = this.levels[0];
        if (this.motions.includes("blink") && this.levels.length > 1) {
          const k = Math.round(blinkAmount(t1, P) * cfg.blink.levels);
          layer = this.levels[Math.min(k, this.levels.length - 1)];
        }
        ctx.save();
        const M = subjectMatrix(t1, P, this.motions, cfg);
        if (M) setCv2Transform(ctx, M);
        ctx.drawImage(layer, 0, 0);
        ctx.restore();
      }
      for (const m of this.mirrors) { const g = m.getContext("2d"); g.imageSmoothingQuality = "high"; g.clearRect(0, 0, m.width, m.height); g.drawImage(this.cv, 0, 0, m.width, m.height); }
      const ms = performance.now() - t0, st = this.stats;
      st.renders++; st.last_ms = ms; st.avg_ms += (ms - st.avg_ms) / Math.min(st.renders, 60);
      if (this.onframe) this.onframe(t1, ms);
    }
    play(restart) {
      if (this.playing && !restart) return;
      cancelAnimationFrame(this.raf);
      this.playing = true; this.start0 = performance.now() - this.t * this.seconds * 1000;
      let lastK = -1;
      const tick = now => {
        if (!this.playing) return;
        const t = this.quantize((now - this.start0) / 1000 / this.seconds), k = Math.round(t * this.frames());
        if (k !== lastK || this.dirty) { lastK = k; this.dirty = false; this.render(t); }
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }
    pause() { this.playing = false; cancelAnimationFrame(this.raf); }
  }
  const api = {LIVE, affineAbout, motion, subjectMatrix, parallaxMatrix, blinkAmount, LivePreview};
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AvatarPreview = api;
})(typeof self !== "undefined" ? self : this);
