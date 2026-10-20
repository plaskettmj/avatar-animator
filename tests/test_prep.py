"""Tests for the prepare/export split (live-preview step 1).

Run with: .venv/bin/python -m unittest discover -s tests -v   (run_tests.sh does this too)
Segmentation runs once per test image (in this process only); everything else reuses the cache."""
import hashlib, io, json, os, shutil, subprocess, sys, tempfile, threading, unittest, urllib.request, uuid
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import animate
from avatar_animator import core
from avatar_animator.core import Analysis, Animator

IMG = lambda name: os.path.join(ROOT, "test-images", name)
QUIET = lambda *a, **k: None
SMALL = dict(out=256, seconds=1.0, fps=10, log=QUIET)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def frames_hash(an):
    return hashlib.sha256(b"".join(an.frame(k).tobytes() for k in range(an.N))).hexdigest()

class NoSlowCalls:
    """Make segmentation and the eye detectors fail loudly, to prove a cached path never runs them."""
    def __enter__(self):
        self.saved = core.segment, core.detect_face_eyes, core.detect_pet_eyes
        def boom(*a, **k): raise AssertionError("slow step ran despite cache")
        core.segment = core.detect_face_eyes = core.detect_pet_eyes = boom
    def __exit__(self, *exc):
        core.segment, core.detect_face_eyes, core.detect_pet_eyes = self.saved

class PrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aa-test-")
        cls.src = {n: np.asarray(Image.open(IMG(n)).convert("RGB")) for n in ("pet-cat.jpg", "portrait-woman.jpg", "logo-badge.png")}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cache(self, name):
        return os.path.join(self.tmp, "cache-" + name)

    def test_cached_analysis_renders_identical_frames(self):
        for name, style in (("portrait-woman.jpg", "portrait"), ("pet-cat.jpg", "pet")):
            with self.subTest(name):
                src = self.src[name]
                fresh = Animator(src, style=style, analysis=Analysis(src, cache_dir=self.cache(name), log=QUIET), **SMALL)
                h0 = frames_hash(fresh)
                self.assertEqual(len(fresh.eyes), 2)
                with NoSlowCalls():
                    ana = Analysis(src, cache_dir=self.cache(name), log=QUIET)
                    again = Animator(src, style=style, analysis=ana, **SMALL)
                    self.assertEqual(frames_hash(again), h0)
                self.assertEqual(ana.misses, [])
                self.assertIn("mask:birefnet-general-lite", ana.hits)

    def test_cache_is_dropped_for_a_different_image(self):
        d = self.cache("pet-cat.jpg")
        Analysis(self.src["pet-cat.jpg"], cache_dir=d, log=QUIET).mask()
        other = self.src["pet-cat.jpg"].copy(); other[0, 0] ^= 1
        ana = Analysis(other, cache_dir=os.path.join(self.tmp, "cache-other"), log=QUIET)
        shutil.copytree(d, ana.cache_dir, dirs_exist_ok=True)
        ana2 = Analysis(other, cache_dir=ana.cache_dir, log=QUIET)   # key mismatch -> stale files removed
        self.assertFalse(any(f.startswith("mask-") for f in os.listdir(ana2.cache_dir)))

    def test_prepare_writes_layers_and_eyes_round_trip(self):
        out = os.path.join(self.tmp, "pet")
        shutil.copytree(self.cache("pet-cat.jpg"), os.path.join(out, "prep"), dirs_exist_ok=True) if os.path.isdir(self.cache("pet-cat.jpg")) else None
        prep = animate.prepare(IMG("pet-cat.jpg"), out, style="pet", size=256, log=QUIET)
        for f in ("prep.json", "subject.png", "background.png", "preview.png", "analysis.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, "prep", f)), f)
        E = prep["canvas"]["size"]
        with Image.open(os.path.join(out, "prep", "subject.png")) as im:
            self.assertEqual((im.mode, im.size), ("RGBA", (E, E)))
        self.assertEqual(prep["eye_source"], "pet")
        self.assertEqual(len(prep["eyes"]), 2)
        boxes = animate.parse_eyes(prep["eyes_arg"])
        for (cx, cy, w, h) in boxes:
            self.assertTrue(0 < cx < 1 and 0 < cy < 1 and 0 < w < 0.5 and 0 < h < 0.5)
        # feeding the detected eyes back in as manual eyes lands them in the same place
        src = self.src["pet-cat.jpg"]
        with NoSlowCalls():
            an = Animator(src, style="pet", eyes=boxes, analysis=Analysis(src, cache_dir=os.path.join(out, "prep"), log=QUIET), **SMALL)
        self.assertEqual(an.eye_source(), "manual")
        np.testing.assert_allclose(np.array(an.eye_boxes()), np.array(boxes), atol=0.01)
        # the crop args reproduce the crop
        with NoSlowCalls():
            an2 = Animator(src, style="pet", center=tuple(prep["crop"]["center"]), zoom=prep["crop"]["zoom"],
                           analysis=Analysis(src, cache_dir=os.path.join(out, "prep"), log=QUIET), **SMALL)
        for k, v in zip(("cx", "cy", "side"), an2.crop):
            self.assertAlmostEqual(float(v), prep["crop"][k], delta=1.0)

    def test_manual_eyes_blink_where_none_were_found(self):
        src = self.src["logo-badge.png"]
        ana = Analysis(src, cache_dir=self.cache("logo-badge.png"), log=QUIET)
        an = Animator(src, style="logo", subject=["blink"], bg=[], blink="off",
                      eyes=[(0.4, 0.4, 0.1, 0.06), (0.6, 0.4, 0.1, 0.06)], analysis=ana, **SMALL)
        self.assertEqual(len(an.eyes), 2)
        t_blink = an.P["blink_at"] + an.P["blink_len"] / 2
        diff = np.abs(an.frame_t(t_blink).astype(int) - an.frame_t(0.0).astype(int)).sum(2)
        ys, xs = np.nonzero(diff > 8)
        self.assertGreater(len(ys), 20, "blink frame should differ from rest frame")
        self.assertTrue(0.25 * 256 < ys.mean() < 0.55 * 256, "changes should be around the eye row")

    def test_cli_prepare_only_then_fast_reexport_with_eyes(self):
        out = os.path.join(self.tmp, "cli")
        # Seed the mask cache from this process (which already holds the model), so the child processes never
        # load BiRefNet: two copies at once can need more RAM than a small box has.
        Analysis(self.src["pet-cat.jpg"], cache_dir=os.path.join(out, "prep"), log=QUIET).mask()
        py = sys.executable
        base = [py, os.path.join(ROOT, "animate.py"), IMG("pet-cat.jpg"), "--style", "pet", "-o", out,
                "--size", "256", "--seconds", "1", "--fps", "8"]
        r = subprocess.run(base + ["--prepare-only"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        prep = load_json(os.path.join(out, "prep", "prep.json"))
        self.assertNotIn("mask:birefnet-general-lite", prep["cache"]["misses"])
        self.assertEqual(len(prep["eyes"]), 2)
        self.assertFalse(os.path.exists(os.path.join(out, "avatar.gif")), "prepare-only must not encode")
        eyes = "0.40,0.45,0.10,0.06;0.60,0.45,0.10,0.06"
        r = subprocess.run(base + ["--eyes", eyes], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        rep = load_json(os.path.join(out, "report.json"))
        self.assertEqual(rep["cache"]["misses"], [])
        self.assertEqual(rep["eye_source"], "manual")
        self.assertEqual(rep["eyes_arg"], "0.4000,0.4500,0.1000,0.0600;0.6000,0.4500,0.1000,0.0600")
        self.assertTrue(rep["gif"]["seam_ok"] and rep["webp"]["seam_ok"] and rep["periodic_t1_equals_t0"])
        self.assertTrue(os.path.isfile(os.path.join(out, "avatar.gif")))
        self.assertEqual(load_json(os.path.join(out, "prep", "prep.json"))["eye_source"], "manual")

    def test_parse_eyes(self):
        self.assertEqual(animate.parse_eyes("0.1,0.2,0.3,0.4;0.5,0.6,0.7,0.8"), [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)])
        self.assertEqual(animate.parse_eyes("none"), [])
        self.assertIsNone(animate.parse_eyes(None))
        with self.assertRaises(SystemExit):
            animate.parse_eyes("0.1,0.2,0.3")

    def test_webui_prepare_and_export(self):
        jobs = os.path.join(self.tmp, "jobs"); os.makedirs(jobs)
        os.environ["AVATAR_ANIMATOR_JOBS"] = jobs
        import importlib, webui
        webui = importlib.reload(webui)
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            b = uuid.uuid4().hex
            body = (f"--{b}\r\nContent-Disposition: form-data; name=\"style\"\r\n\r\nlogo\r\n"
                    f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\n"
                    f"Content-Type: image/png\r\n\r\n").encode() + read_bytes(IMG("logo-badge.png")) + f"\r\n--{b}--\r\n".encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/prepare", body,
                                         {"Content-Type": f"multipart/form-data; boundary={b}"})
            res = json.load(urllib.request.urlopen(req, timeout=300))
            self.assertIn("prep/preview.png", res["urls"])
            self.assertEqual(res["prep"]["style"], "logo")
            png = urllib.request.urlopen(f"http://127.0.0.1:{port}{res['urls']['prep/subject.png']}").read()
            self.assertEqual(Image.open(io.BytesIO(png)).mode, "RGBA")
            ex = dict(job=res["job"], seconds=1, fps=8, subject=["float", "blink"], eyes=[[0.4, 0.4, 0.1, 0.06], [0.6, 0.4, 0.1, 0.06]])
            req = urllib.request.Request(f"http://127.0.0.1:{port}/export", json.dumps(ex).encode(), {"Content-Type": "application/json"})
            res2 = json.load(urllib.request.urlopen(req, timeout=300))
            self.assertEqual(res2["report"]["cache"]["misses"], [])
            self.assertEqual(res2["report"]["eye_source"], "manual")
            self.assertIn("avatar.gif", res2["urls"])
            bad = urllib.request.Request(f"http://127.0.0.1:{port}/export", b'{"job": "../etc"}', {"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(bad)
            self.assertEqual(cm.exception.code, 404)
        finally:
            srv.shutdown()

if __name__ == "__main__":
    unittest.main()
