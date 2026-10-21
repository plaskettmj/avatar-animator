#!/usr/bin/env bash
# Unit tests, then the golden test (hand-placed eyes make a blink when auto-detection finds none), then renders
# the four sample images and checks each result (size budget, seamless loop).
# Runs sequentially: each render peaks at about 2-3 GB RAM. First run also downloads the models (~400 MB).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Run ./install.sh first." >&2; exit 1; }
$PY -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED|FAIL:|ERROR:)|Error" ; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "unit tests failed (run: $PY -m unittest discover -s tests -v)" >&2; exit 1; }
$PY -W ignore -m unittest tests/golden_manual_eyes.py 2>&1 | grep -E "^(Ran|OK|FAILED|FAIL:|ERROR:)|eye pixels|Error" ; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "golden test failed (run: $PY -m unittest tests/golden_manual_eyes.py -v)" >&2; exit 1; }
$PY animate.py test-images/portrait-woman.jpg --style portrait -o out/portrait > /dev/null
$PY animate.py test-images/pet-cat.jpg        --style pet      -o out/pet      > /dev/null
$PY animate.py test-images/logo-badge.png     --style logo     -o out/logo     > /dev/null
$PY animate.py test-images/space-planet.png   --style space --set swirl_ratio=0.35 -o out/space > /dev/null
$PY - <<'PY'
import json, sys
fail = 0
for name in ("portrait", "pet", "logo", "space"):
    r = json.load(open(f"out/{name}/report.json"))
    checks = {
        "gif<5MB": r["gif"]["size_mb"] < 5.0, "webp<5MB": r["webp"]["size_mb"] < 5.0,
        "gif seam": r["gif"]["seam_ok"], "webp seam": r["webp"]["seam_ok"],
        "periodic": r["periodic_t1_equals_t0"],
    }
    if name != "logo":  # portrait (face), pet (cat cascade) and space (cartoon eyes) should all blink
        checks["eyes found"] = len(r["eyes"]) == 2
    bad = [k for k, v in checks.items() if not v]
    fail += bool(bad)
    print(f"{'PASS' if not bad else 'FAIL'} {name:8s} gif {r['gif']['size_mb']} MB {r['gif']['dims'][0]}px, "
          f"webp {r['webp']['size_mb']} MB, eyes={r['eyes']}, {r['render_s']} s" + (f"  failed: {bad}" if bad else ""))
sys.exit(1 if fail else 0)
PY
