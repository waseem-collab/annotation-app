# annotation-app

A single-file Flask web app for editing **YOLO** bounding-box annotations, with a
radial class picker, an image scrubber, model-assisted auto-annotation, and
one-click upload to CVAT.

![status](https://img.shields.io/badge/python-3.8%2B-blue)

## Features

- **Box editing** — drag empty canvas to add, drag inside to move, grab a handle to
  resize, right-click or `Delete` to remove. Each box is drawn in its class colour.
- **Classes from `labels.txt`** — cycle through only the classes defined for the
  dataset. Press number keys, use the dropdown, or **hold `C`** for a radial class
  wheel (move onto a class, release to pick).
- **Existing labels always shown & editable** by default; edits autosave to `labels/`.
- **Top scrubber** — drag the slider to fly through every image in real time.
- **Box-count filter**, zoom (scroll wheel), and keyboard nav (`A`/`D`, `S` to save).
- **Per-class visibility** — checkboxes to show/hide labels on the canvas (view only;
  hidden boxes are still saved).
- **Delete image** — remove the current image and its label from disk, then advance.
- **Automatic annotation** — pick a YOLO `.pt` model (popup), set a confidence, and
  batch-annotate the whole folder with a live progress bar. Detections are mapped to
  your class list by name.
- **CVAT upload** — list projects from a dropdown, pull a project's classes to annotate
  with, and upload the current folder (images + YOLO labels) as a new task — or, for an
  imported folder, **update the source task's annotations** in place.
- **CVAT import** — pick a project then a task and import its images + annotations into a
  local folder to edit here.

## Dataset layout

Point the **Dataset folder** box at a root laid out like:

```
my_dataset/
  images/        .jpg / .jpeg / .png
  labels/        YOLO .txt (one per image, same stem)
  labels.txt     class names, ONE per line; line number = class id (0-based)
```

`classes.txt` is accepted as a fallback. If neither exists, classes are auto-named.

## Install

```bash
pip install flask                 # required
pip install ultralytics           # optional: automatic annotation
pip install cvat-sdk python-dotenv # optional: CVAT upload
```

## Run

```bash
python3 annotation_app.py
# then open http://127.0.0.1:5000
```

The last-used folder is remembered between launches.

## Automatic annotation (optional)

Put YOLO `.pt` models under a `models/` folder beside the script (or set
`MODELS_DIR`). Each model shows up by its top-level folder name. Click
**Automatic annotation…**, choose a model + confidence, and **Annotate** to run over
every image in the folder.

## CVAT upload (optional)

Create a `.env` next to the script:

```ini
CVAT_URL=https://app.cvat.ai
CVAT_USERNAME=you@example.com
CVAT_PASSWORD=your-password
CVAT_ORG_SLUG=your-org        # optional
```

Then pick a project from the dropdown (optionally **lock** it), enter a task name,
and **Upload this folder**. Annotations are sent in CVAT "YOLO 1.1" format; the
project's labels must include the class names being used.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `A` / `D` (or ←/→) | previous / next image |
| `S` | save |
| `0`–`9` | set active class |
| hold `C` | radial class wheel |
| `Delete` / `Backspace` | delete selected box |
| scroll | zoom |

## Notes

- `.env`, `models/`, datasets, and `*.pt` files are gitignored — keep credentials and
  weights out of version control.
- First inference with a model is slow (it loads the weights), then it's cached. It
  uses the GPU automatically if available, otherwise CPU.
