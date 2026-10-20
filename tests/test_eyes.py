"""Tests for the eye editor (live-preview step 2): grabbing an eye's shape from a click, per-eye blink styles,
eye positions in source-image px, the chat grid, prep/ staying as prepared, and the /eyes web endpoint.

Most tests use a synthetic cartoon face with an injected subject mask, so no model runs and they take seconds."""
import io, json, os, shutil, sys, tempfile, threading, unittest, urllib.request, uuid
import numpy as np, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import animate
from avatar_animator import core
from avatar_animator.core import Analysis, Animator, grab_eye, illustration_score, parse_eye_spec

QUIET = lambda *a, **k: None
SMALL = dict(out=256, seconds=1.0, fps=10, log=QUIET)
EYES = ((300, 330), (500, 330))       # synthetic eye centres, source px (800 x 800 image)
EYE_W, EYE_H = 110, 130

def cartoon_face():
    """800x800 flat mascot: brown disc on green, two outlined white eyes with brown irises and black pupils."""
    img = np.zeros((800, 800, 3), np.uint8); img[:] = (30, 90, 40)
    cv2.circle(img, (400, 420), 300, (20, 20, 20), -1)
    cv2.circle(img, (400, 420), 292, (140, 80, 40), -1)
    for (x, y) in EYES:
        cv2.ellipse(img, (x, y), (EYE_W // 2, EYE_H // 2), 0, 0, 360, (15, 15, 15), -1)
        cv2.ellipse(img, (x, y), (EYE_W // 2 - 6, EYE_H // 2 - 6), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, (x + 5, y + 8), 28, (90, 50, 25), -1)
        cv2.circle(img, (x + 5, y + 8), 13, (5, 5, 5), -1)
    cv2.ellipse(img, (400, 560), (120, 50), 0, 0, 180, (20, 20, 20), 8)
    mask = np.zeros((800, 800), np.float32); cv2.circle(mask, (400, 420), 300, 1.0, -1)
    return img, mask

def cartoon_analysis(cache_dir=None):
    """Analysis for the synthetic face with its mask and detector results filled in (no model runs)."""
    img, mask = cartoon_face()
    ana = Analysis(img, cache_dir=cache_dir, log=QUIET)
    ana._masks["birefnet-general-lite"] = mask
    ana._meta["face"] = dict(eyes=[], box=None)
    ana._meta["pet"] = []
    if cache_dir:
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(cache_dir, "mask-birefnet-general-lite.png"))
        ana._save()
    return img, ana

class EyeEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aa-eyes-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_illustration_score_separates_artwork_from_photos(self):
        img, _ = cartoon_face()
        self.assertGreater(illustration_score(img), core.ILLUSTRATION_THRESHOLD)
        photo = np.asarray(Image.open(os.path.join(ROOT, "test-images", "portrait-woman.jpg")).convert("RGB"))
        self.assertLess(illustration_score(photo), core.ILLUSTRATION_THRESHOLD)

    def test_click_grabs_the_whole_eye_from_anywhere_on_it(self):
        img, ana = cartoon_analysis()
        for dx, dy in ((0, 0), (5, 8), (-35, -40), (30, 45), (-48, 0)):   # centre, pupil, sclera, near the outline
            with self.subTest(dx=dx, dy=dy):
                an = Animator(img, eyes=[dict(x=EYES[0][0] + dx, y=EYES[0][1] + dy, space="src")], analysis=ana, **SMALL)
                (e,) = an.eye_specs()
                self.assertEqual(e["kind"], "cartoon")
                self.assertAlmostEqual(e["x"], EYES[0][0], delta=4); self.assertAlmostEqual(e["y"], EYES[0][1], delta=4)
                self.assertAlmostEqual(e["w"], EYE_W, delta=12); self.assertAlmostEqual(e["h"], EYE_H, delta=12)
                self.assertTrue(an.eyes[0]["grab"]["found"])

    def test_click_on_plain_skin_falls_back_to_a_default_ellipse(self):
        img, ana = cartoon_analysis()
        an = Animator(img, eyes=[dict(x=250, y=520, space="src")], analysis=ana, **SMALL)
        self.assertFalse(an.eyes[0]["grab"]["found"])
        self.assertEqual(an.eyes[0]["kind"], "photo")   # nothing to squash: eyelid slide over a default-size eye
        canvas = an.img.copy()
        m, info = grab_eye(canvas, *an.src_to_canvas(250, 520), an.out)
        self.assertIsNone(m)

    def test_kind_auto_and_overrides(self):
        img, ana = cartoon_analysis()
        pts = [dict(x=x, y=y, space="src") for x, y in EYES]
        self.assertEqual([e["kind"] for e in Animator(img, eyes=pts, analysis=ana, **SMALL).eyes], ["cartoon"] * 2)
        an = Animator(img, eyes=[dict(pts[0], kind="photo"), pts[1]], analysis=ana, **SMALL)
        self.assertEqual([e["kind"] for e in an.eyes], ["photo", "cartoon"])   # per-eye choice
        an = Animator(img, eyes=pts, eye_kind="photo", analysis=ana, **SMALL)
        self.assertEqual([e["kind"] for e in an.eyes], ["photo"] * 2)          # global default for auto eyes

    def test_detected_photo_eyes_on_artwork_are_refit_as_cartoon(self):
        """MediaPipe often finds cartoon faces but outlines thin slivers of the eyes; on flat artwork those are
        re-fitted as clicked points and squash-blinked instead of eyelid-slid."""
        img, ana = cartoon_analysis()
        sliver = lambda x, y: [[x - 40, y - 30], [x, y - 40], [x + 40, y - 30], [x, y - 22]]
        ana._meta["face"] = dict(eyes=[sliver(*EYES[1]), sliver(*EYES[0])], box=[150, 150, 500, 500])
        an = Animator(img, analysis=ana, **SMALL)
        self.assertEqual(an.eye_source(), "face")
        self.assertEqual([e["kind"] for e in an.eyes], ["cartoon"] * 2)
        specs = an.eye_specs()
        self.assertLess(specs[0]["x"], specs[1]["x"], "eye 1 is the leftmost")
        for s, (x, y) in zip(specs, EYES):
            self.assertAlmostEqual(s["y"], y, delta=5); self.assertAlmostEqual(s["h"], EYE_H, delta=12)

    def test_manual_eyes_turn_blink_on_for_styles_without_it(self):
        img, ana = cartoon_analysis()
        an = Animator(img, style="logo", eyes=[dict(x=x, y=y, space="src") for x, y in EYES], analysis=ana, **SMALL)
        self.assertIn("blink", an.subj)
        self.assertEqual(len(an.eyes), 2)
        self.assertNotIn("blink", Animator(img, style="logo", analysis=ana, **SMALL).subj)

    def test_cartoon_blink_closes_the_eye(self):
        img, ana = cartoon_analysis()
        an = Animator(img, eyes=[dict(x=x, y=y, space="src") for x, y in EYES], analysis=ana, **SMALL)
        x, y = [int(v) - an.off for v in an.src_to_canvas(*EYES[0])]
        r = int(0.3 * EYE_W * an.scale)
        white = lambda f: (f[y - r:y + r, x - r:x + r].min(2) > 200).mean()
        self.assertGreater(white(an.still(0.0)), 0.2)
        self.assertLess(white(an.still(1.0)), 0.1)
        # and the rendered loop's blink frame matches
        t = an.P["blink_at"] + an.P["blink_len"] / 2
        self.assertLess(white(an.frame_t(t)), 0.15)

    def test_source_positions_survive_crop_changes(self):
        img, ana = cartoon_analysis()
        pts = [dict(x=x, y=y, space="src") for x, y in EYES]
        a = Animator(img, eyes=pts, analysis=ana, **SMALL)
        b = Animator(img, eyes=pts, analysis=ana, zoom=1.6, center=(0.6, 0.5), **SMALL)
        for sa, sb in zip(a.eye_specs(), b.eye_specs()):
            for k in ("x", "y", "w", "h"):
                self.assertAlmostEqual(sa[k], sb[k], delta=3, msg=k)
        self.assertNotAlmostEqual(a.eye_boxes()[0][0], b.eye_boxes()[0][0], delta=0.02)   # the crop did change
        # the --eyes-src string round-trips
        specs = [parse_eye_spec(s) for s in core.format_eye_specs(a.eye_specs()).split(";")]
        c = Animator(img, eyes=specs, analysis=ana, zoom=1.3, **SMALL)
        np.testing.assert_allclose([[s[k] for k in "xywh"] for s in c.eye_specs()],
                                   [[s[k] for k in "xywh"] for s in a.eye_specs()], atol=1.0)

    def test_parsers(self):
        self.assertEqual(animate.parse_eyes("0.1,0.2,0.3,0.4,cartoon"), [(0.1, 0.2, 0.3, 0.4, "cartoon")])
        s = animate.parse_eyes_src("300,330;500,330,110,130,photo")
        self.assertEqual((s[0]["w"], s[0]["kind"], s[1]["w"], s[1]["kind"]), (None, "auto", 110.0, "photo"))
        c = animate.parse_cells("D5, g5:cartoon")
        self.assertEqual([(e["x"], e["y"], e["kind"]) for e in c], [(0.35, 0.45, "auto"), (0.65, 0.45, "cartoon")])
        for bad in ("K5", "D11", "D5:sideways", "55"):
            with self.assertRaises(SystemExit):
                animate.parse_cells(bad)
        with self.assertRaises(SystemExit):
            animate.parse_eyes_src("300")
        self.assertEqual(len(animate.collect_eyes("0.1,0.2,0.3,0.4", "300,330", "D5")), 3)
        self.assertIsNone(animate.collect_eyes(None, "", None))

    def test_chat_grid_path_and_prep_files(self):
        out = os.path.join(self.tmp, "grid"); os.makedirs(os.path.join(out, "prep"))
        img, _ = cartoon_analysis(os.path.join(out, "prep"))
        path = os.path.join(out, "face.png"); Image.fromarray(img).save(path)
        prep = animate.prepare(path, out, size=256, log=QUIET)
        for f in ("grid.png", "mask.png", "crop.png", "preview.png"):
            self.assertTrue(os.path.isfile(os.path.join(out, "prep", f)), f)
        self.assertEqual(prep["cache"]["misses"], [])
        self.assertFalse(set(prep["cache"]["hits"]) & set(prep["cache"]["misses"]))
        # full-image crop: the eyes at (300,330)/(500,330) of 800 px sit in cells D5 and G5 of the 10x10 grid
        before = open(os.path.join(out, "prep", "prep.json")).read()
        rep = animate.export(path, out, size=256, seconds=1, fps=6, eyes=animate.parse_cells("D5,G5"), log=QUIET)
        self.assertEqual(rep["eye_source"], "manual")
        self.assertEqual(rep["eyes"], ["cartoon", "cartoon"])
        for d, (x, y) in zip(rep["eye_details"], EYES):
            self.assertAlmostEqual(d["src"]["x"], x, delta=5); self.assertAlmostEqual(d["src"]["y"], y, delta=5)
        self.assertEqual(open(os.path.join(out, "prep", "prep.json")).read(), before, "export must not rewrite prep.json")
        self.assertTrue(os.path.isfile(os.path.join(out, "eyes.png")))

    def test_cache_listing_has_no_step_in_both_hits_and_misses(self):
        img, _ = cartoon_face()
        d = os.path.join(self.tmp, "hm")
        ana = Analysis(img, cache_dir=d, log=QUIET)
        ana._masks["birefnet-general-lite"] = cartoon_face()[1]
        saved = core.detect_face_eyes, core.detect_pet_eyes
        core.detect_face_eyes, core.detect_pet_eyes = (lambda *a: ([], None)), (lambda *a: [])
        try:
            ana.face(); ana.pet(); ana.face(); ana.pet()   # computed once, then reused in memory
        finally:
            core.detect_face_eyes, core.detect_pet_eyes = saved
        self.assertEqual(ana.misses, ["face", "pet"]); self.assertEqual(ana.hits, [])
        again = Analysis(img, cache_dir=d, log=QUIET); again.face(); again.pet()
        self.assertEqual(again.hits, ["face", "pet"]); self.assertEqual(again.misses, [])

    def test_webui_eyes_endpoint_and_export_with_source_eyes(self):
        jobs = os.path.join(self.tmp, "jobs"); os.makedirs(jobs)
        os.environ["AVATAR_ANIMATOR_JOBS"] = jobs
        import importlib, webui
        webui = importlib.reload(webui)
        from http.server import ThreadingHTTPServer
        # a job prepared from the synthetic face (cache seeded so no model runs)
        job = "20260101-000000-abcdef"; d = os.path.join(jobs, job); os.makedirs(os.path.join(d, "prep"))
        img, _ = cartoon_analysis(os.path.join(d, "prep"))
        Image.fromarray(img).save(os.path.join(d, "input"), format="PNG")
        json.dump(dict(style="portrait"), open(os.path.join(d, "settings.json"), "w"))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        post = lambda path, obj: json.load(urllib.request.urlopen(urllib.request.Request(
            base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"}), timeout=300))
        try:
            self.assertIn(b"Eye editor", urllib.request.urlopen(base + "/editor").read())
            res = post("/prepare", dict(job=job, style="pet"))           # re-prepare an existing job
            self.assertEqual(res["prep"]["style"], "pet"); self.assertIn("prep/mask.png", res["urls"])
            r = post("/eyes", dict(job=job, eyes=[dict(x=EYES[1][0] + 10, y=EYES[1][1]), dict(x=EYES[0][0], y=EYES[0][1], kind="photo")],
                                   test_blink=True))
            self.assertEqual([e["kind"] for e in r["eyes"]], ["photo", "cartoon"])   # sorted left to right
            self.assertTrue(r["eyes"][1]["grabbed"]); self.assertIn("outline", r["eyes"][1])
            strip = Image.open(io.BytesIO(urllib.request.urlopen(base + r["urls"]["blink"]).read()))
            self.assertEqual(strip.size, (r["size"] * r["frames"], r["size"]))
            ex = post("/export", dict(job=job, seconds=1, fps=6, eyes=r["eyes"]))
            self.assertEqual(ex["report"]["eye_source"], "manual")
            self.assertEqual(ex["report"]["eyes"], ["photo", "cartoon"])
            self.assertIn("eyes.png", ex["urls"])
            nb = post("/export", dict(job=job, seconds=1, fps=6, no_blink=True))
            self.assertNotIn("blink", nb["report"]["subject_motions"])
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post("/eyes", dict(job="../x"))
            self.assertEqual(cm.exception.code, 404)
        finally:
            srv.shutdown()

if __name__ == "__main__":
    unittest.main()
