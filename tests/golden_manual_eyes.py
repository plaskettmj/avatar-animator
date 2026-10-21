"""Golden test (live-preview step 5): hand-placed eyes make an image blink when auto-detection finds none.

Renders real sample images end to end (segmentation, crop, encode) and checks the encoded GIF:
  1. without eyes the detector finds none, so the loop has no blink (every frame is the same),
  2. with `--eyes` (CLI) / `eyes=` (Python) the blink frame differs from the open frame inside the eye boxes,
     and nowhere else.
Only the blink moves (`--subject blink --bg none`), so any change between frames comes from the eyelids.

The sample is pet-cat.jpg with `--blink face`: the human face detector finds nothing on a cat, which stands in
for the dogs and side profiles that no detector handles (the sample set has no dog or profile photo, and the
default `--blink auto` finds this cat with the cat-face detector). It is checked through the CLI (`--eyes`) and
through the Python API (`animate.run(..., eyes=...)`), sharing one cached cut-out.

Run: .venv/bin/python -m unittest tests/golden_manual_eyes.py -v   (run_tests.sh runs it after the unit tests).
Takes about a minute on CPU: each image is segmented once, then the cache in the output folder is reused."""
import json, os, shutil, subprocess, sys, tempfile, unittest
import numpy as np
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import animate

IMG = lambda name: os.path.join(ROOT, "test-images", name)
QUIET = lambda *a, **k: None
SIZE, SECONDS, FPS = 256, 1.0, 10
FAST = ["--size", str(SIZE), "--seconds", str(SECONDS), "--fps", str(FPS), "--subject", "blink", "--bg", "none"]
# Eyes as fractions of the output square (cx, cy, w, h), placed by hand on prep/grid.png.
CAT_EYES = "0.347,0.546,0.120,0.117;0.612,0.583,0.136,0.101"
CAT_EYES_PY = [(0.347, 0.546, 0.120, 0.117, "photo"), (0.612, 0.583, 0.136, 0.101, "photo")]

def gif_frames(path, fps=FPS):
    """Decoded GIF frames on the loop's frame grid. gifsicle merges identical neighbouring frames into one
    longer frame, so each decoded frame is repeated for its duration."""
    out = []
    with Image.open(path) as im:
        for f in ImageSequence.Iterator(im):
            n = max(1, int(round(f.info.get("duration", 1000 / fps) * fps / 1000)))
            out += [np.asarray(f.convert("RGB")).astype(np.float32)] * n
    return out

def blink_index(report):
    """Frame index of the middle of the blink (same rule as the contact sheet)."""
    p = dict(blink_at=0.62, blink_len=0.11); p.update(report.get("params", {}))
    return int(round((p["blink_at"] + p["blink_len"] / 2) * report["frames"])) % report["frames"]

def eye_mask(report, side, grow=1.0):
    """Boolean mask of the eye boxes from report.json (output fractions), each box scaled by `grow`."""
    m = np.zeros((side, side), bool)
    for e in report["eye_details"]:
        cx, cy, w, h = e["box"]
        x0, x1 = int((cx - w * grow / 2) * side), int(np.ceil((cx + w * grow / 2) * side))
        y0, y1 = int((cy - h * grow / 2) * side), int(np.ceil((cy + h * grow / 2) * side))
        m[max(0, y0):y1, max(0, x0):x1] = True
    return m

class GoldenManualEyes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aa-golden-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, image, out, *args):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "animate.py"), IMG(image), "-o", out, *FAST, *args],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        with open(os.path.join(out, "report.json")) as f:
            return json.load(f)

    def check_no_blink(self, out, report):
        self.assertEqual(report["eyes"], [], "auto-detection should find no eyes on this image")
        self.assertEqual(report["eye_source"], "none")
        fs = gif_frames(os.path.join(out, "avatar.gif"))
        worst = max(float(np.abs(f - fs[0]).max(2).mean()) for f in fs)
        self.assertLess(worst, 0.5, "without eyes nothing should move (blink is the only motion)")

    def check_blink(self, out, report):
        self.assertEqual(report["eye_source"], "manual")
        self.assertEqual(len(report["eyes"]), 2)
        self.assertIn("blink", report["subject_motions"])
        fs = gif_frames(os.path.join(out, "avatar.gif"))
        self.assertEqual(len(fs), report["frames"])
        side = fs[0].shape[0]
        diff = np.abs(fs[blink_index(report)] - fs[0]).max(2)
        # the photo blink slides the skin above the eye down over it, so the lid area around each box changes too
        inside, near = eye_mask(report, side), eye_mask(report, side, grow=2.2)
        changed_in = float((diff[inside] > 24).mean())
        changed_out = float((diff[~near] > 24).mean())
        self.assertGreater(changed_in, 0.15, f"blink frame should differ inside the eyes ({changed_in:.1%} changed)")
        self.assertLess(changed_out, 0.002, f"blink frame should match outside the eyes ({changed_out:.2%} changed)")
        # and the loop opens again: the last frame is back to open
        self.assertLess(float((np.abs(fs[-1] - fs[0]).max(2)[inside] > 24).mean()), 0.02)
        return changed_in

    def test_cli_eyes_make_a_cat_blink_when_the_face_detector_finds_none(self):
        out = os.path.join(self.tmp, "cat")
        self.check_no_blink(out, self.cli("pet-cat.jpg", out, "--style", "pet", "--blink", "face"))
        r = self.cli("pet-cat.jpg", out, "--style", "pet", "--blink", "face", "--eyes", CAT_EYES)
        self.assertEqual(r["cache"]["misses"], [], "re-export should reuse the cached cut-out")
        changed = self.check_blink(out, r)
        print(f"\n  cat (--eyes): {changed:.0%} of eye pixels change at the blink frame", file=sys.stderr)

    def test_python_eyes_make_a_cat_blink_when_the_face_detector_finds_none(self):
        out = os.path.join(self.tmp, "cat-py")
        kw = dict(style="pet", subject=["blink"], bg=[], seconds=SECONDS, fps=FPS, size=SIZE, blink="face", log=QUIET)
        self.check_no_blink(out, animate.run(IMG("pet-cat.jpg"), out, **kw))
        changed = self.check_blink(out, animate.run(IMG("pet-cat.jpg"), out, eyes=CAT_EYES_PY, **kw))
        print(f"\n  cat (eyes=): {changed:.0%} of eye pixels change at the blink frame", file=sys.stderr)

if __name__ == "__main__":
    unittest.main()
