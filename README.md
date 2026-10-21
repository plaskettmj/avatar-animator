# avatar-animator

Turns a still image into a seamless looping animated avatar: a 512×512 GIF and an animated WebP, each under 5 MB,
so they fit the usual avatar upload limit for bots and chat apps. It runs locally on CPU. There's no server and no
API key, and model weights download automatically on first run.

It works on photos of people, pets, logos and illustrations: the subject breathes, tilts, floats and blinks,
and the background drifts, swirls or sparkles.

When the automatic result is off, you can fix it without redoing the slow part:

* **Eyes:** place them by naming cells on a grid (works from a chat) or by clicking them in the browser editor,
  and choose how each one blinks (cartoon squash or photo eyelid). Dogs, side profiles and odd mascots need this.
* **Cut-out:** check `prep/mask.png` and switch to the full cut-out model if parts of the subject are missing.
* **Crop:** drag and zoom the square in the editor, or pass `--center`/`--zoom`.
* **Motion:** tune float, breathe, tilt, blink and background drift with sliders on a live preview that plays in
  the browser, so nothing is rendered until you export.

The slow analysis (background removal and eye detection) runs once per image and is cached, so every fix after
that re-exports in seconds.

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
the same `-o` and it is reused, so changing the eyes, motions or `--set` values skips segmentation.

```bash
# 1. slow step only: mask, crop, background fill, eye detection (about 10 s cold, under 1 s when cached)
.venv/bin/python animate.py my-photo.jpg --style pet -o out/me --prepare-only
# -> out/me/prep/preview.png  crop with the detected eyes outlined, numbered and labelled cartoon/photo
#    out/me/prep/grid.png     the same square with an A-J / 1-10 grid (for placing eyes from a chat)
#    out/me/prep/mask.png     the cut-out: what moves as the subject is bright, the rest is dimmed
#    out/me/prep/crop.png, subject.png, background.png, prep.json
# 2. fix the eyes if needed, then export (prep/ is left as prepared; out/me/eyes.png shows the eyes used)
.venv/bin/python animate.py my-photo.jpg --style pet -o out/me --eyes-at "D5,G5"
```

`prep.json` lists the eyes (`eyes[*].src` in source-image pixels, `eyes[*].box` in output fractions, `kind`),
`eyes_src_arg` / `eyes_arg` strings you can pass back, the crop as `--center`/`--zoom` values, and which detector
found the eyes (`eye_source`: face, pet, cartoon, manual or none). `--no-cache` turns the cache off.

### Placing eyes by hand

Three ways, all of which can be mixed:

| flag | example | meaning |
|---|---|---|
| `--eyes-at` | `"D5,G5"` | cells of `prep/grid.png`. The eye's shape is found around each cell centre |
| `--eyes-src` | `"365,470;605,470"` or `"365,470,120,130,cartoon"` | source-image pixels: a point (shape found automatically) or a box. Stays valid if the crop, zoom or style changes |
| `--eyes` | `"0.36,0.46,0.12,0.14"` | fractions of the output square (tied to the current crop), optional `,cartoon` / `,photo` |

Each eye blinks in one of two ways:

* **cartoon**: the eye shape is lifted off the face and squashed shut. Right for mascots, logos, app icons and
  drawings with outlined eyes.
* **photo**: the skin above the eye slides down like an eyelid. Right for real faces and pets. On a cartoon it
  drags the eyebrows into the eyes.

`auto` (the default) picks cartoon when the eye's shape was found and the image is flat artwork (or the style
is `logo`/`space`), otherwise photo. Force it per eye (`D5:cartoon`, `365,470,cartoon`) or for all eyes with
`--eye-kind cartoon|photo`. Placing eyes turns blinking on even for styles that don't blink by default. Eyes the
face detector finds on flat artwork are re-fitted as cartoon eyes automatically.

**Chat flow for bots:** run `--prepare-only`, send the user `prep/grid.png`, ask which cells the eyes are in,
then export with `--eyes-at "<cells>"`. Check `eyes.png` (or `report.json` → `eye_details[*].grabbed`) and
send the result.

**Cut-out model:** if `mask.png` shows parts missing from the subject (headsets, props, bodies), re-run with
`--model birefnet-general` (about 30 s and up to 8 GB RAM the first time, cached after). Missing parts stay
still and get smeared by the background fill when the subject moves.

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
| `--eyes "cx,cy,w,h[,kind];..."` | place the eyes by hand, as fractions of the output square (e.g. `"0.40,0.42,0.08,0.05;0.58,0.42,0.08,0.05"`) |
| `--eyes-src "x,y[,w,h][,kind];..."` | place the eyes in source-image pixels; a bare point finds the eye's shape |
| `--eyes-at "D5,G5"` | place the eyes by `prep/grid.png` cell (`D5:cartoon` to force the blink style) |
| `--eye-kind auto\|cartoon\|photo` | blink style for eyes left on auto (see "Placing eyes by hand") |
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

A bot running on its own Linux computer (a Grok Bot's computer, for example) can do the whole flow itself, so the
person it works for doesn't install anything:

```bash
git clone https://github.com/plaskettmj/avatar-animator.git && cd avatar-animator && ./install.sh
# 1. slow step only (cached in out/avatar/prep/)
.venv/bin/python animate.py /path/to/input.jpg --style pet -o out/avatar --prepare-only
# 2. render and encode, reusing the cache (add --eyes-at, --center/--zoom or --set from the user's fixes)
.venv/bin/python animate.py /path/to/input.jpg --style pet -o out/avatar
# then upload out/avatar/avatar.gif (or avatar.webp) as the new avatar; both are < 5 MB
```

Pick the style that matches the image (person: `portrait`, animal: `pet`, logo/icon: `logo`, illustration or
space scene: `space`). After the prepare step, look at `prep/preview.png` and `prep/prep.json` (or
`report.json` after an export):

* **No eyes found, or the user says they're wrong.** If `eyes` is empty and the subject has eyes (typical for
  dogs, side profiles and unusual mascots), offer to place them. Offer the chat grid first, because it needs
  nothing but the chat:
  1. *Chat grid (default):* send the user `prep/grid.png`, ask which cells the eyes are in ("D5 and G5"),
     export with `--eyes-at "D5,G5"`, and check `eyes.png` (or `report.json` → `eye_details[*].grabbed`)
     before sending the result.
  2. *Browser editor (option):* run `.venv/bin/python webui.py` and open `http://127.0.0.1:8765/editor` in
     the browser on the bot's own computer. The page listens on localhost only, so the user watches or drives it
     on the bot's desktop (a Grok Bot's owner can open its desktop to watch or help); never expose the port to
     the internet. There they click the eyes, press Test blink, drag or zoom the crop and tune the live-preview
     sliders, then Export. Results are saved in `webui-jobs/<job>/` (`avatar.gif`, `avatar.webp`,
     `report.json`).

  If they'd rather it didn't blink, export with `--blink off`.
* **Blink style.** For cartoons and logos, `eyes` should say `cartoon`: `photo` eyes on a drawing blink by
  sliding the eyebrows down (pass `--eye-kind cartoon`, or `D5:cartoon` per eye).
* **Parts missing from the cut-out** (`prep/mask.png`): re-run with `--model birefnet-general`.
* **Crop cuts the subject or sits off-centre:** adjust `--center`/`--zoom`, starting from `crop.args` in
  `prep.json`, or have the user drag the crop in the editor.
* **More or less motion:** `--set` (for example `--set tilt_deg=6`), or the editor's live-preview sliders.

Look at `contact-sheet.png` before uploading. It includes a mid-blink frame whenever the eyes blink.

## Web UI (optional)

```bash
.venv/bin/python webui.py      # http://127.0.0.1:8765  (PORT env var to change)
```

This is a small local page, standard library only. Upload an image, pick a style or motions, then download the
GIF/WebP and see 96/48/32 px previews. Jobs are stored in `webui-jobs/`. It listens on localhost only.

**Eye editor:** http://127.0.0.1:8765/editor. Choose an image, a style and a cut-out model (fast or full),
then Prepare. Detected eyes are drawn on the cropped square. Click an eye to add it (its shape is found
automatically), drag to move, drag the square handle to resize, Delete to remove, and pick cartoon/photo per
eye. "Show cut-out" shows the mask; switching the cut-out model or style keeps your eyes. "Test blink" plays
the real renderer closing the eyes in about a second; "Export" renders and encodes the loop.

**Live preview:** after Prepare, the "3. Live preview" panel under the editor plays the avatar in the browser,
with round 96/48/32 px copies at sidebar sizes. Sliders for float height, breathe, tilt, blink timing and
length, background drift (parallax) and loop length update it instantly; nothing is rendered on the server
while you drag. Ticking float, breathe, tilt or parallax on or off is also instant. The preview shows the frames
the export will sample (same fps and loop length), and Export uses the slider values and toggles. Particles,
sparkles, twinkle, swirl, hue and sweep need the full renderer, so the preview shows them as they look at t=0
and labels them; Export animates them. Eye edits refresh the preview layers in about half a second.

**Crop:** drag the live preview to move the crop (the picture follows the pointer) and scroll on it, or use the
Zoom slider, to zoom. While you drag, the preview shows the source image under the new crop; on release the
server re-crops in about a second (the cut-out is cached, so only the crop and background fill run) and the
preview carries on with your sliders. The crop always stops at the image edges, so the border is never smeared,
and the small overview shows where the crop sits in the whole image. "Auto crop" goes back to the automatic
framing. Eyes stay where you put them; an eye the crop leaves out is kept (listed as outside the crop) and blinks
again when the crop moves back. The same crop from the command line is `--center x,y --zoom z`; `prep.json`
and `report.json` print it as `crop.args`.

JSON endpoints for editors and bots (details in the docstring at the top of `webui.py`): `POST /prepare`
(multipart upload, or JSON `{"job", "model", "style"}` to re-prepare a job, plus `"center"`/`"zoom"` or
`"auto_crop": true` to re-crop it), `POST /eyes` (resolve clicked
points into eyes, optional blink test strip), `POST /preview` (the layers the live preview animates: frozen
background, subject, eye patches closing in steps) and `POST /export` (eyes in source pixels or any CLI format,
plus `params`, `subject`, `bg`, `seconds` from the sliders).

## How it works

```
image ─► subject mask (BiRefNet-lite via rembg) ─► square crop on the subject
      ─► layers: subject (RGBA) + background with the subject hole inpainted (OpenCV)
      ─► eyes: MediaPipe face landmarks │ cat-face cascade │ cartoon-eye heuristic │ manual (clicked point → shape)
      ─► blink per eye: cartoon (squash the eye shape) │ photo (eyelid slide)
      ─► per-layer motions, all periodic in loop time t, so frame(t=1) == frame(t=0)
      ─► GIF (one global palette, optional gifsicle, auto-shrinks to fit) + WebP ─► contact sheet + report.json
```

## Tests

```bash
./run_tests.sh
```

This runs the unit tests in `tests/` (prepare/export split, cache reuse, eye round-trip, eye grabbing and blink
styles, grid cells, crop clamping and re-cropping, live-preview layers and the browser motion math checked against the renderer under node,
CLI and web endpoints),
then a golden test (`tests/golden_manual_eyes.py`) that renders the cat sample with only the human face
detector, so no eyes are found and nothing blinks, then places the eyes by hand (`--eyes` on the command line and
`eyes=` in Python) and checks that the encoded blink frame differs from the open frame inside the eyes and
nowhere else. Finally it renders the four CC0 sample images in `test-images/` and checks the size budget and the
seamless loop for each one.

## Limitations

Frontal faces, cats and most outlined cartoon eyes blink. Dogs, profiles and sunglasses usually don't, so place
the eyes by hand (`--eyes-at`, `--eyes-src` or the editor) for those. Large
moves can show smeary background fill. See [docs/ROADMAP.md](docs/ROADMAP.md) for known limitations and planned
improvements.

## License

MIT (see [LICENSE](LICENSE)). Third-party models and libraries keep their own permissive licenses; see
[CREDITS.md](CREDITS.md). No model weights are bundled.
