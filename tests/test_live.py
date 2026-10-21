"""Tests for the live motion preview (live-preview step 3): preview.js mirrors the renderer's motion math, the
layers POST /preview writes recompose into the exported frames, and the editor's slider settings reach the export.

Uses the synthetic cartoon face from test_eyes (injected subject mask), so no model runs. The JS checks run
preview.js under node and are skipped when node is not installed."""
import io, json, os, shutil, subprocess, sys, tempfile, threading, unittest, urllib.request
import numpy as np, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import animate
from avatar_animator.core import Animator, DEFAULTS
from test_eyes import cartoon_analysis, EYES, QUIET

NODE = shutil.which("node")
EYE_SPECS = [dict(x=x, y=y, w=None, h=None, kind="cartoon", space="src") for x, y in EYES]
TS = [0.0, 0.07, 0.25, 0.4, 0.5, 0.61, 0.66, 0.8, 0.93]

def animator(style="portrait", subject=None, bg=None, params=None, eyes=EYE_SPECS, out=256):
    img, ana = cartoon_analysis()
    return Animator(img, out=out, style=style, subject=subject, bg=bg, params=params, eyes=eyes, seconds=1.0, fps=10,
                    analysis=ana, log=QUIET)

def u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

class LivePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aa-live-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @unittest.skipUnless(NODE, "node not installed")
    def test_preview_js_motion_math_matches_the_renderer(self):
        cases = []
        for style, subject, params in (("portrait", None, None), ("logo", None, dict(breathe_amp=0.06)),
                                       ("space", ["float", "blink"], dict(float_amp=0.07, blink_at=0.3, blink_len=0.2)),
                                       ("pet", ["tilt", "breathe", "float"], dict(tilt_deg=9.5, parallax_amp=0.05))):
            an = animator(style, subject, params=params)
            cfg = dict(canvas=an.E, out=an.out, pivot=[float(v) for v in an.pivot], style=an.style)
            P = {k: float(an.P[k]) for k in animate.LIVE_PARAMS}
            want = [dict(subject=None if an.subject_matrix(t) is None else an.subject_matrix(t).ravel().tolist(),
                         parallax=an.parallax_matrix(t).ravel().tolist(), blink=an.blink_amount(t)) for t in TS]
            cases.append(dict(cfg=cfg, P=P, motions=an.subj, want=want, style=style))
        js = """const p = require(process.argv[1]); const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
        const ts = %s;
        console.log(JSON.stringify(cases.map(c => ts.map(t => ({subject: p.subjectMatrix(t, c.P, c.motions, c.cfg),
          parallax: p.parallaxMatrix(t, c.P, c.cfg), blink: p.blinkAmount(t, c.P)})))));""" % json.dumps(TS)
        r = subprocess.run([NODE, "-e", js, os.path.join(ROOT, "preview.js")], input=json.dumps(cases),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout)
        for c, g in zip(cases, got):
            for t, w, j in zip(TS, c["want"], g):
                with self.subTest(style=c["style"], t=t):
                    self.assertAlmostEqual(w["blink"], j["blink"], places=9)
                    np.testing.assert_allclose(j["parallax"], w["parallax"], atol=1e-6)
                    if w["subject"] is None:
                        self.assertIsNone(j["subject"])
                    else:
                        np.testing.assert_allclose(j["subject"], w["subject"], atol=1e-6)

    def test_layers_recompose_into_the_exported_frames(self):
        """What the canvas does (parallax-warp the frozen background, paste the blink patch, warp the subject,
        alpha-composite), done here with cv2 on the saved PNGs, must give the renderer's frames."""
        params = dict(tilt_deg=7.0, float_amp=0.05, blink_at=0.55, blink_len=0.2)
        an = animator("portrait", ["float", "breathe", "tilt", "blink"], ["parallax"], params)
        d = os.path.join(self.tmp, "live-recompose")
        cfg = animate.live_layers(an, d)
        self.assertEqual(sorted(cfg["live"]), ["blink", "breathe", "float", "parallax", "tilt"])
        self.assertEqual(cfg["frozen"], []); self.assertEqual(cfg["eyes"], 2)
        bg = np.asarray(Image.open(os.path.join(d, "background.png"))).astype(np.float32) / 255
        sub = np.asarray(Image.open(os.path.join(d, "subject.png"))).astype(np.float32) / 255
        strip = np.asarray(Image.open(os.path.join(d, "blink.png"))).astype(np.float32) / 255
        x0, y0, x1, y1 = cfg["blink"]["box"]; L = cfg["blink"]["levels"]; w = x1 - x0
        self.assertEqual(strip.shape[:2], (y1 - y0, w * L))
        E, o, out = an.E, an.off, an.out
        t_closed = params["blink_at"] + params["blink_len"] / 2       # blink amount exactly 1 -> the last level
        for t in TS + [t_closed]:
            with self.subTest(t=t):
                b = an.blink_amount(t)
                k = int(round(b * L))
                if t != t_closed and 0 < k < L:
                    continue   # in-between blink amounts are quantised to L levels in the preview
                layer = sub.copy()
                if k:
                    layer[y0:y1, x0:x1] = strip[:, (k - 1) * w:k * w]
                f_bg = cv2.warpAffine(bg, an.parallax_matrix(t), (E, E), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                M = an.subject_matrix(t)
                rgb, a = layer[..., :3], layer[..., 3]
                if M is not None:
                    rgb = cv2.warpAffine(rgb, M, (E, E), flags=cv2.INTER_LINEAR)
                    a = cv2.warpAffine(a, M, (E, E), flags=cv2.INTER_LINEAR)
                f = f_bg * (1 - a[..., None]) + rgb * a[..., None]
                got = u8(f[o:o + out, o:o + out]).astype(int)
                want = an.frame_t(t).astype(int)
                diff = np.abs(got - want)
                self.assertLess(diff.mean(), 0.6)
                self.assertLessEqual(np.percentile(diff, 99.9), 3)

    def test_blink_patches_cover_every_changed_pixel(self):
        an = animator("portrait")
        d = os.path.join(self.tmp, "live-blink")
        cfg = animate.live_layers(an, d)
        x0, y0, x1, y1 = cfg["blink"]["box"]
        closed = u8(an.apply_blink(an.fg, 1.0)); base = u8(an.fg)
        outside = np.ones(base.shape[:2], bool); outside[y0:y1, x0:x1] = False
        self.assertTrue((closed[outside] == base[outside]).all())
        strip = np.asarray(Image.open(os.path.join(d, "blink.png")))
        np.testing.assert_array_equal(strip[:, -(x1 - x0):, :3], closed[y0:y1, x0:x1])
        # both eyes are inside the box
        for x, y in EYES:
            cx, cy = an.src_to_canvas(x, y)
            self.assertTrue(x0 <= cx < x1 and y0 <= cy < y1)
        # no eyes / no blink: no blink layer, and an old one is removed
        nb = animate.live_layers(animator("portrait", ["tilt"], eyes=None), d)
        self.assertIsNone(nb["blink"]); self.assertFalse(os.path.exists(os.path.join(d, "blink.png")))

    def test_frozen_motions_are_listed_per_style(self):
        logo = animate.live_layers(animator("logo", eyes=None), os.path.join(self.tmp, "live-logo"))
        self.assertEqual(logo["live"], ["float", "breathe"])
        self.assertEqual(sorted(logo["frozen"]), ["hue", "sparkles", "sweep"])   # backdrop is static: neither
        space = animate.live_layers(animator("space"), os.path.join(self.tmp, "live-space"))
        self.assertEqual(space["live"], ["float", "blink"]); self.assertEqual(sorted(space["frozen"]), ["swirl", "twinkle"])
        # the frozen background is the t=0 background without parallax
        an = animator("portrait")
        d = os.path.join(self.tmp, "live-portrait"); animate.live_layers(an, d)
        bg = np.asarray(Image.open(os.path.join(d, "background.png")))
        np.testing.assert_array_equal(bg, u8(an.background(0.0, skip=("parallax",))))
        self.assertEqual(set(logo["params"]), set(animate.LIVE_PARAMS))

    def test_webui_preview_endpoint_and_export_with_slider_settings(self):
        jobs = os.path.join(self.tmp, "jobs"); os.makedirs(jobs)
        os.environ["AVATAR_ANIMATOR_JOBS"] = jobs
        import importlib, webui
        webui = importlib.reload(webui)
        from http.server import ThreadingHTTPServer
        job = "20260101-000000-abc123"; d = os.path.join(jobs, job); os.makedirs(os.path.join(d, "prep"))
        img, _ = cartoon_analysis(os.path.join(d, "prep"))
        Image.fromarray(img).save(os.path.join(d, "input"), format="PNG")
        json.dump(dict(style="portrait"), open(os.path.join(d, "settings.json"), "w"))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        post = lambda path, obj: json.load(urllib.request.urlopen(urllib.request.Request(
            base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=300))
        try:
            page = urllib.request.urlopen(base + "/editor").read()
            self.assertIn(b"/preview.js", page); self.assertIn(b"Live preview", page)
            js = urllib.request.urlopen(base + "/preview.js")
            self.assertIn("javascript", js.headers["Content-Type"]); self.assertIn(b"LivePreview", js.read())
            eyes = [dict(x=x, y=y, kind="cartoon") for x, y in EYES]
            r = post("/preview", dict(job=job, eyes=eyes))
            live = r["live"]
            self.assertEqual([e["kind"] for e in r["eyes"]], ["cartoon", "cartoon"])
            self.assertEqual(sorted(live["live"]), ["blink", "breathe", "parallax", "tilt"])
            self.assertEqual(live["frozen"], ["particles"])
            self.assertEqual(set(r["urls"]), {"background", "subject", "blink"})
            for u in r["urls"].values():
                self.assertEqual(Image.open(io.BytesIO(urllib.request.urlopen(base + u).read())).size[0] > 0, True)
            sub = Image.open(io.BytesIO(urllib.request.urlopen(base + r["urls"]["subject"]).read()))
            self.assertEqual((sub.mode, sub.size), ("RGBA", (live["canvas"], live["canvas"])))
            # motion toggles change the frozen layer and the lists; no_blink drops the blink layer
            r2 = post("/preview", dict(job=job, eyes=eyes, subject=["float", "blink"], bg=["sparkles"], no_blink=True))
            self.assertIsNone(r2["live"]["blink"]); self.assertEqual(r2["live"]["live"], ["float"])
            self.assertEqual(r2["live"]["frozen"], ["sparkles"]); self.assertNotIn("blink", r2["urls"])
            # Export with the editor's slider values and toggles
            params = dict(live["params"], tilt_deg=8.0, blink_at=0.3)
            ex = post("/export", dict(job=job, seconds=1.5, fps=6, eyes=r["eyes"], params=params,
                                      subject=["tilt", "blink"], bg=["parallax"]))
            rep = ex["report"]
            self.assertEqual(rep["params"], dict(tilt_deg=8.0, blink_at=0.3))
            self.assertEqual(rep["subject_motions"], ["tilt", "blink"]); self.assertEqual(rep["bg_motions"], ["parallax"])
            self.assertEqual((rep["seconds"], rep["frames"]), (1.5, 9))
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post("/preview", dict(job="nope"))
            self.assertEqual(cm.exception.code, 404)
        finally:
            srv.shutdown()

if __name__ == "__main__":
    unittest.main()
