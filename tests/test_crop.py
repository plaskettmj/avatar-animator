"""Tests for crop drag and zoom (live-preview step 4): the crop square always stays inside the image (no smeared
border, including the logo style's auto crop), re-cropping keeps the eyes and skips the slow steps, and the web
editor's crop reaches /eyes, /preview and /export.

Uses the synthetic cartoon face from test_eyes (injected subject mask), so no model runs."""
import json, os, shutil, subprocess, sys, tempfile, threading, unittest, urllib.request
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import animate
from avatar_animator import core
from avatar_animator.core import Animator, clamp_crop
from test_eyes import cartoon_analysis, cartoon_face, EYES, QUIET
from test_prep import NoSlowCalls

NODE = shutil.which("node")

def animator(style="portrait", **kw):
    img, ana = cartoon_analysis()
    return Animator(img, out=256, style=style, seconds=1.0, fps=10, analysis=ana, log=QUIET, **kw)

def inside(an):
    cx, cy, side = an.crop; W, H = an.src_size
    return cx - side / 2 >= -1e-6 and cy - side / 2 >= -1e-6 and cx + side / 2 <= W + 1e-6 and cy + side / 2 <= H + 1e-6

class CropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aa-crop-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_clamp_crop(self):
        self.assertEqual(clamp_crop(500, 400, 300, 1000, 800), (500.0, 400.0, 300.0))    # already inside
        self.assertEqual(clamp_crop(50, 790, 300, 1000, 800), (150.0, 650.0, 300.0))     # pushed in from the edges
        self.assertEqual(clamp_crop(500, 400, 1500, 1000, 800), (500.0, 400.0, 800.0))   # never larger than the image
        self.assertEqual(clamp_crop(0, 0, 2, 100, 100)[2], 8.0)

    @unittest.skipUnless(NODE, "node not installed")
    def test_preview_js_clamp_matches(self):
        cases = [(500, 400, 300, 1000, 800), (50, 790, 300, 1000, 800), (500, 400, 1500, 1000, 800),
                 (-90, 1200, 640, 640, 1280), (10, 10, 1, 64, 32)]
        js = ("const p = require(process.argv[1]); const c = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(c.map(a => p.clampCrop(...a))));")
        r = subprocess.run([NODE, "-e", js, os.path.join(ROOT, "preview.js")], input=json.dumps(cases),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        np.testing.assert_allclose(json.loads(r.stdout), [clamp_crop(*c) for c in cases])

    def test_logo_auto_crop_stays_inside_the_image(self):
        """The logo style frames 1.55x the subject: for this 600 px disc in an 800 px image that used to be a 930 px
        square, 65 px past each border (replicated edge pixels). Now it is the whole image."""
        an = animator("logo")
        self.assertTrue(an.crop_clamped)
        self.assertAlmostEqual(an.crop_wanted[2], 930, delta=5)
        self.assertEqual(an.crop[2], 800.0); self.assertTrue(inside(an))
        # the visible square is the source image itself, resampled: no replicated border
        img, _ = cartoon_face()
        o, out = an.off, an.out
        vis = (np.clip(an.img[o:o + out, o:o + out], 0, 1) * 255 + 0.5).astype(np.uint8)
        want = np.asarray(Image.fromarray(img).resize((out, out), Image.BOX))
        self.assertLess(np.abs(vis.astype(int) - want.astype(int)).mean(), 2.0)
        for style in ("portrait", "pet", "space"):
            self.assertTrue(inside(animator(style)), style)

    def test_center_and_zoom_are_clamped(self):
        an = animator(center=(0.02, 0.98), zoom=2.0)        # 400 px square asked at the bottom-left corner
        self.assertEqual(an.crop, (200.0, 600.0, 400.0)); self.assertTrue(an.crop_clamped)
        an = animator(center=(0.5, 0.5), zoom=0.6)          # zoom < 1 would pass the border
        self.assertEqual(an.crop[2], 800.0)
        an = animator(center=(0.4, 0.45), zoom=2.5)
        self.assertEqual(an.crop, (320.0, 360.0, 320.0)); self.assertFalse(an.crop_clamped)

    def test_recrop_keeps_eyes_and_skips_slow_steps(self):
        d = os.path.join(self.tmp, "recrop"); os.makedirs(os.path.join(d, "prep"))
        img, _ = cartoon_analysis(os.path.join(d, "prep"))
        path = os.path.join(d, "input.png"); Image.fromarray(img).save(path)
        eyes = [dict(x=x, y=y, w=None, h=None, kind="cartoon", space="src") for x, y in EYES]
        p0 = animate.prepare(path, d, eyes=eyes, size=256, log=QUIET)
        self.assertEqual(p0["source"], dict(width=800, height=800))
        with NoSlowCalls():
            p1 = animate.prepare(path, d, eyes=eyes, size=256, center=(0.5, 0.42), zoom=1.8, log=QUIET)
        self.assertNotEqual(p0["crop"]["side"], p1["crop"]["side"])
        self.assertEqual(p1["crop"]["args"], "--center 0.5000,0.4200 --zoom 1.8000")
        for e0, e1 in zip(p0["eyes"], p1["eyes"]):        # same eyes in source px, different output boxes
            self.assertAlmostEqual(e0["src"]["x"], e1["src"]["x"], delta=1.5)
            self.assertAlmostEqual(e0["src"]["y"], e1["src"]["y"], delta=1.5)
            self.assertGreater(e1["box"][2], e0["box"][2])
        # the printed args reproduce the crop (the CLI equivalent of a drag)
        c = p1["crop"]
        an = animator(center=tuple(c["center"]), zoom=c["zoom"])
        self.assertAlmostEqual(an.crop[2], c["side"], delta=0.5)

    def test_eyes_outside_the_crop_are_kept_aside(self):
        """Zooming in on one corner must not fail on eyes the crop leaves out; they come back with the crop."""
        eyes = [dict(x=x, y=y, w=None, h=None, kind="cartoon", space="src") for x, y in EYES]
        an = animator(center=(0.9, 0.9), zoom=5.0, eyes=eyes)        # 160 px square in the bottom-right corner
        self.assertEqual(an.eyes, [])
        self.assertEqual([(round(e["x"]), round(e["y"])) for e in an.eyes_off], [(x, y) for x, y in EYES])
        d = os.path.join(self.tmp, "off-crop"); cfg = animate.live_layers(an, d)
        self.assertIsNone(cfg["blink"])
        back = animator(eyes=[dict(e, space="src") for e in an.eyes_off])
        self.assertEqual(len(back.eyes), 2); self.assertEqual(back.eyes_off, [])

    def test_segmentation_session_is_reused_without_a_memory_arena(self):
        """The web UI's second full-model Prepare used to be OOM-killed: ONNX Runtime's CPU arena grew on reuse."""
        import rembg
        made = []
        class FakeSession: pass
        def fake_new_session(model, sess_opts=None):
            made.append((model, sess_opts)); return FakeSession()
        def fake_remove(im, session=None, only_mask=True):
            return Image.new("L", im.size, 255)
        saved = rembg.new_session, rembg.remove, dict(core._sessions)
        rembg.new_session, rembg.remove = fake_new_session, fake_remove
        core._sessions.clear()
        try:
            img = np.zeros((32, 32, 3), np.uint8)
            for _ in range(3):
                m = core.segment(img, "birefnet-general")
            self.assertEqual(m.shape, (32, 32)); self.assertEqual(len(made), 1)       # one session, reused
            so = made[0][1]
            self.assertFalse(so.enable_cpu_mem_arena); self.assertFalse(so.enable_mem_pattern)
            core.segment(img, "birefnet-general-lite")
            self.assertEqual(list(core._sessions), ["birefnet-general-lite"])         # still one model in memory
        finally:
            rembg.new_session, rembg.remove = saved[0], saved[1]
            core._sessions.clear(); core._sessions.update(saved[2])

    def test_webui_crop_reaches_eyes_preview_and_export(self):
        jobs = os.path.join(self.tmp, "jobs"); os.makedirs(jobs)
        os.environ["AVATAR_ANIMATOR_JOBS"] = jobs
        import importlib, webui
        webui = importlib.reload(webui)
        from http.server import ThreadingHTTPServer
        job = "20260101-000000-cc0001"; d = os.path.join(jobs, job); os.makedirs(os.path.join(d, "prep"))
        img, _ = cartoon_analysis(os.path.join(d, "prep"))
        Image.fromarray(img).save(os.path.join(d, "input"), format="PNG")
        json.dump(dict(style="logo"), open(os.path.join(d, "settings.json"), "w"))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        post = lambda path, obj: json.load(urllib.request.urlopen(urllib.request.Request(
            base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=300))
        try:
            page = urllib.request.urlopen(base + "/editor").read()
            self.assertIn(b'id=autocrop', page); self.assertIn(b"clampCrop", page)
            r0 = post("/prepare", dict(job=job))
            self.assertTrue(r0["prep"]["crop"]["clamped"]); self.assertEqual(r0["prep"]["crop"]["side"], 800.0)
            src = urllib.request.urlopen(base + r0["urls"]["prep/source.jpg"])
            self.assertEqual(src.headers["Content-Type"], "image/jpeg")
            with NoSlowCalls():   # re-cropping runs only the crop and fill steps
                r1 = post("/prepare", dict(job=job, center=[0.5, 0.45], zoom=2))
            self.assertEqual(r1["prep"]["crop"]["side"], 400.0); self.assertEqual(r1["prep"]["crop"]["cy"], 360.0)
            self.assertEqual(json.load(open(os.path.join(d, "settings.json")))["zoom"], 2.0)
            # out-of-range values are clamped, not rejected
            r2 = post("/prepare", dict(job=job, center=[1.7, -3], zoom=0.2))
            self.assertEqual((r2["prep"]["crop"]["cx"], r2["prep"]["crop"]["cy"], r2["prep"]["crop"]["side"]), (400.0, 400.0, 800.0))
            post("/prepare", dict(job=job, center=[0.5, 0.45], zoom=2))
            eyes = [dict(x=x, y=y, kind="cartoon") for x, y in EYES]
            e = post("/eyes", dict(job=job, eyes=eyes))
            self.assertAlmostEqual(e["eyes"][0]["x"], EYES[0][0], delta=15)
            pv = post("/preview", dict(job=job, eyes=eyes))
            # the preview layers use the saved crop: the subject (the disc) now overflows the visible square
            a = np.asarray(Image.open(os.path.join(d, "live", "subject.png")))[..., 3]
            o, out = pv["live"]["off"], pv["live"]["out"]
            self.assertGreater((a[o:o + out, o:o + out] > 128).mean(), 0.9)
            ex = post("/export", dict(job=job, seconds=1, fps=6, eyes=e["eyes"]))
            self.assertEqual(ex["report"]["crop"]["side"], 400.0)
            self.assertIn("--zoom 2.0000", ex["report"]["crop"]["args"])
            post("/prepare", dict(job=job, center=[0.95, 0.95], zoom=6))      # eyes outside the crop
            off = post("/preview", dict(job=job, eyes=eyes))
            self.assertEqual([e.get("off_crop") for e in off["eyes"]], [True, True]); self.assertIsNone(off["live"]["blink"])
            r3 = post("/prepare", dict(job=job, auto_crop=True))
            self.assertEqual(r3["prep"]["crop"]["side"], 800.0); self.assertTrue(r3["prep"]["crop"]["auto"])
            self.assertNotIn("zoom", json.load(open(os.path.join(d, "settings.json"))))
        finally:
            srv.shutdown()

if __name__ == "__main__":
    unittest.main()
