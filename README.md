# avatar-animator

Turns a still image into a seamless looping animated avatar: a 512×512 GIF and an animated WebP, each under 5 MB,
so they fit the usual avatar upload limit for bots and chat apps. It runs locally on CPU. There's no server and no
API key, and model weights download automatically on first run.

It works on photos of people, pets, logos and illustrations: the subject breathes, tilts, floats and blinks,
and the background drifts, swirls or sparkles.

## Install (Linux, Python 3.10+)

```bash
git clone https://github.com/plaskettmj/avatar-animator.git && cd avatar-animator && ./install.sh
```

`install.sh` creates `.venv/` and installs the pinned packages from `requirements.txt` (about 1 GB). The first
render downloads about 400 MB of model weights into `./models/`. Recommended system packages (Debian/Ubuntu):

```bash
sudo apt-get install -y libegl1 libgles2 gifsicle
# libegl1/libgles2: needed by MediaPipe face detection (without them photos of people still animate, but don't blink)
# gifsicle: optional, makes GIFs smaller
```

Each render needs about 2–3 GB of free RAM and takes roughly 30–90 seconds on a typical CPU.

## Quick start

```bash
.venv/bin/python animate.py my-photo.jpg --style portrait -o out/me
# -> out/me/avatar.gif, out/me/avatar.webp, out/me/contact-sheet.png, out/me/report.json
```

Use `avatar.webp` when the destination accepts WebP: it stays at 512 px and is usually 2–3× smaller and cleaner.
`avatar.gif` works everywhere. If a photo's GIF would go over the budget, it is automatically shrunk to 448 or
384 px. `report.json` lists the final sizes, dimensions, whether eyes were found, and a seamless-loop check.

### Prepare once, re-export fast

The slow analysis (background removal, face and cat-eye detection) is cached in `<out>/prep/`. Run again with
the same `-o` and it is reused, so changing `--eyes`, motions or `--set` values skips segmentation.

```bash
# 1. slow step only: mask, crop, background fill, eye detection (about 10 s cold, under 1 s when cached)
.venv/bin/python animate.py my-photo.jpg --style pet -o out/me --prepare-only
# -> out/me/prep/preview.png (crop with the detected eyes outlined and numbered), prep.json,
#    subject.png + background.png (layers on the padded canvas)
# 2. fix the eyes if needed (prep.json "eyes_arg" is in the same format), then export
.venv/bin/python animate.py my-photo.jpg --style pet -o out/me --eyes "0.35,0.55,0.12,0.11;0.61,0.58,0.13,0.10"
```

`prep.json` also lists the crop as `--center`/`--zoom` values and which detector found the eyes
(`eye_source`: face, pet, cartoon, manual or none). `--no-cache` turns the cache off.

## Styles

| `--style` | for | subject motions | background motions |
|---|---|---|---|
| `portrait` (default) | photos of people | breathe, tilt, blink | parallax, particles |
| `pet` | animals | tilt, breathe, blink | parallax, sparkles |
| `logo` | logos, icons, flat graphics | float, breathe, sweep | backdrop, hue, sparkles |
| `space` | illustrations on a sky or space background | float, blink | swirl, twinkle |

```bash
.venv/bin/python animate.py cat.jpg    --style pet      -o out/cat
.venv/bin/python animate.py logo.png   --style logo     -o out/logo
.venv/bin/python animate.py planet.png --style space --set swirl_ratio=0.35 -o out/planet
```

## Options

| option | meaning |
|---|---|
| `--subject float,blink` | replace the style's subject motions (`float`, `breathe`, `tilt`, `blink`, `sweep`, or `none`) |
| `--bg swirl,twinkle` | replace the background motions (`parallax`, `swirl`, `twinkle`, `sparkles`, `particles`, `sweep`, `hue`, `backdrop`, or `none`) |
| `--blink auto\|face\|pet\|cartoon\|off` | eye detector to use for blinking (default `auto`) |
| `--eyes "cx,cy,w,h;cx,cy,w,h"` | place the eyes by hand, as fractions of the output square (e.g. `"0.40,0.42,0.08,0.05;0.58,0.42,0.08,0.05"`) |
| `--seconds 3` / `--fps 20` | loop length and frame rate |
| `--center 0.43,0.47` / `--zoom 1.4` | crop centre as fractions of the source image; zoom > 1 crops tighter |
| `--size 512` | output size in px |
| `--model birefnet-general-lite` | mask model: `birefnet-general-lite` (default), `birefnet-general`, `u2net`, `isnet-general-use` |
| `--set key=value` | tune a motion (repeatable), e.g. `--set tilt_deg=6 --set swirl_deg=50` |
| `--prepare-only` | run only the slow step and write `<out>/prep/` (no frames, no encoding) |
| `--no-cache` | don't read or write the `<out>/prep/` analysis cache |

More examples:

```bash
.venv/bin/python animate.py img.png --subject float,blink --bg swirl,twinkle   # pick motions yourself
.venv/bin/python animate.py img.png --bg none                                 # move only the subject
.venv/bin/python animate.py img.png --blink off --seconds 4 --fps 20
```

Tunable keys for `--set`, with their defaults: `float_amp` 0.035, `breathe_amp` 0.035, `tilt_deg` 4,
`blink_at` 0.62, `blink_len` 0.11, `parallax_amp` 0.03, `swirl_deg` 80, `swirl_ratio` 1.0 (lower values flatten
the swirl into a tilted disk), `swirl_tilt` 0, `hue_deg` 40, `sweep_strength` 0.6, `n_sparkles` 12,
`n_particles` 36.

## Using it from an AI agent / bot

A bot running on its own Linux computer can do the whole flow with these commands:

```bash
git clone https://github.com/plaskettmj/avatar-animator.git && cd avatar-animator && ./install.sh
.venv/bin/python animate.py /path/to/input.jpg --style portrait -o out/avatar
# then upload out/avatar/avatar.gif (or avatar.webp) as the new avatar; both are < 5 MB
```

Pick the style that matches the image (person: `portrait`, animal: `pet`, logo/icon: `logo`, illustration or
space scene: `space`). Check `out/avatar/report.json`: if `eyes` is empty and the subject has eyes, rerun with
`--eyes`, or use `--blink off`. Look at `contact-sheet.png` before uploading.

## Web UI (optional)

```bash
.venv/bin/python webui.py      # http://127.0.0.1:8765  (PORT env var to change)
```

This is a small local page, standard library only. Upload an image, pick a style or motions, then download the
GIF/WebP and see 96/48/32 px previews. Jobs are stored in `webui-jobs/`. It listens on localhost only.

It also has two JSON endpoints for editors and bots: `POST /prepare` (multipart upload, same fields as the form)
runs only the slow step and returns the crop, the detected eyes and links to the preview and layer PNGs;
`POST /export` with `{"job": ..., "eyes": "cx,cy,w,h;cx,cy,w,h"}` (plus optional `style`, `subject`, `bg`,
`seconds`, `fps`, `blink`, `params`) renders from the job's cache. See the docstring at the top of `webui.py`.

## How it works

```
image ─► subject mask (BiRefNet-lite via rembg) ─► square crop on the subject
      ─► layers: subject (RGBA) + background with the subject hole inpainted (OpenCV)
      ─► eyes: MediaPipe face landmarks │ cat-face cascade │ cartoon-eye heuristic │ manual --eyes
      ─► per-layer motions, all periodic in loop time t, so frame(t=1) == frame(t=0)
      ─► GIF (one global palette, optional gifsicle, auto-shrinks to fit) + WebP ─► contact sheet + report.json
```

## Tests

```bash
./run_tests.sh
```

This runs the unit tests in `tests/` (prepare/export split, cache reuse, eye round-trip, CLI and web endpoints),
then renders the four CC0 sample images in `test-images/` and checks the size budget and the seamless loop
for each one.

## Limitations

Frontal faces and cats blink. Dogs, profiles and sunglasses usually don't, so pass `--eyes` for those. Large
moves can show smeary background fill. See [docs/ROADMAP.md](docs/ROADMAP.md) for known limitations and planned
improvements.

## License

MIT (see [LICENSE](LICENSE)). Third-party models and libraries keep their own permissive licenses; see
[CREDITS.md](CREDITS.md). No model weights are bundled.
