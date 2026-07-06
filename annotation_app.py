#!/usr/bin/env python3
"""
Web-based YOLO annotation editor with class support + radial class picker.

Run:
    python3 annotation_app.py
then open the printed URL (default http://127.0.0.1:5000) in a browser.

Dataset layout it expects (point the "Dataset folder" box at the root):

    my_dataset/
        images/        <- .jpg/.jpeg/.png
        labels/        <- YOLO .txt (one per image, same stem)
        labels.txt     <- class names, ONE per line; line number = class id (0-based)

If labels.txt is missing it falls back to classes.txt, and if neither exists
classes are auto-named "class 0", "class 1", ... as they appear.

What you can do in the browser:
  * see every image with its boxes drawn on top;
  * MOVE a box   -> click inside it and drag;
  * RESIZE a box -> grab any of the 8 handles (4 corners + 4 edges) and drag;
  * ADD a box    -> drag on empty canvas to rubber-band a new box (uses the
                    currently active class);
  * DELETE a box -> select it (click) and press Delete/Backspace, right-click it,
                    or the Delete button;
  * PICK A CLASS -> press number keys 0-9, the dropdown, OR **hold C** to open a
                    radial wheel of the classes from labels.txt: move the cursor
                    onto a class and release C to select it (applies to the
                    selected box too);
  * SAVE         -> click Save (or press S). Writes the YOLO .txt back to labels/.

Labels are YOLO format: "<class> <cx> <cy> <w> <h>" normalised to [0,1].
"""

import os
import re
import glob
import json
import threading
from flask import Flask, jsonify, request, send_file, Response

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **k):
        return False

BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# CVAT credentials (read from .env next to this script)
# --------------------------------------------------------------------------- #
load_dotenv(os.path.join(BASE, ".env"))
CVAT_URL = (os.getenv("CVAT_URL") or os.getenv("CVAT_HOST") or "").rstrip("/")
CVAT_USER = os.getenv("CVAT_USERNAME")
CVAT_PASS = os.getenv("CVAT_PASSWORD")
CVAT_ORG = os.getenv("CVAT_ORG_SLUG") or "visionify"

# Folder of YOLO .pt models for automatic annotation (override with MODELS_DIR).
MODELS_DIR = os.getenv("MODELS_DIR") or os.path.join(BASE, "models")

# Where imported CVAT tasks are unpacked (override with IMPORTS_DIR).
IMPORTS_DIR = os.getenv("IMPORTS_DIR") or os.path.join(BASE, "imports")
# Remembers the last folder you loaded, so the UI reopens it next launch.
STATE_FILE = os.path.join(BASE, ".annotation_app_state")

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _load_state():
    """Persisted session state: {path, image, active_class}. Falls back to the
    legacy plain-text-path format if that's what's on disk."""
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            raw = fh.read().strip()
        if not raw:
            return {}
        if raw[0] == "{":
            return json.loads(raw)
        return {"path": raw}              # legacy: file held just the folder path
    except (OSError, ValueError):
        return {}


STATE = _load_state()


def _save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(STATE, fh)
    except OSError:
        pass


def _remembered_folder(default):
    p = STATE.get("path")
    if p and os.path.isdir(os.path.join(p, "images")):
        return p
    return default


def _remember_folder(path):
    # switching folders resets the per-folder context (image + active class)
    if STATE.get("path") != path:
        STATE["image"] = None
        STATE["active_class"] = 0
    STATE["path"] = path
    _save_state()


DATA = _remembered_folder(os.path.join(BASE, "dataset"))
IMG_DIR = os.path.join(DATA, "images")
LBL_DIR = os.path.join(DATA, "labels")

app = Flask(__name__)


@app.after_request
def _no_cache(resp):
    # The browser must never serve stale label data / filter results from cache,
    # otherwise edits look like they "didn't save" after reloading.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def list_images():
    """Sorted list of image basenames present in images/."""
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(IMG_DIR, "*" + ext))
        files += glob.glob(os.path.join(IMG_DIR, "*" + ext.upper()))
    return sorted({os.path.basename(f) for f in files})


def read_classes():
    """Class names from labels.txt (preferred) or classes.txt, one per line.

    The list index IS the YOLO class id. Blank lines are skipped but still
    consume an id slot only if they are trailing — to stay faithful to common
    tooling we keep non-empty lines in order and drop blanks."""
    for fname in ("labels.txt", "classes.txt"):
        p = os.path.join(DATA, fname)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    names = [ln.strip() for ln in fh.read().splitlines()]
                names = [n for n in names if n != ""]
                if names:
                    return names
            except OSError:
                pass
    return []


# Built once at startup; rebuilt on folder switch.
IMAGES = list_images()
CLASSES = read_classes()


def label_path_for(img_name):
    stem = os.path.splitext(img_name)[0]
    return os.path.join(LBL_DIR, stem + ".txt")


def _read_label_file(p):
    boxes = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = parts[:5]
                boxes.append({"cls": int(float(cls)), "cx": float(cx),
                              "cy": float(cy), "w": float(w), "h": float(h)})
    return boxes


def read_label(img_name):
    """Return list of dicts {cls, cx, cy, w, h} (normalised) for an image."""
    return _read_label_file(label_path_for(img_name))


def _write_label_file(path, boxes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for b in boxes:
        # clamp to [0,1] and drop degenerate boxes
        cx = min(max(b["cx"], 0.0), 1.0)
        cy = min(max(b["cy"], 0.0), 1.0)
        w = min(max(b["w"], 0.0), 1.0)
        h = min(max(b["h"], 0.0), 1.0)
        if w <= 0 or h <= 0:
            continue
        cls = int(b.get("cls", 0))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


def write_label(img_name, boxes):
    _write_label_file(label_path_for(img_name), boxes)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def _linked_task():
    """If the current folder was imported from a CVAT task, return its info
    (includes the full frame map for upload use)."""
    try:
        with open(os.path.join(DATA, ".cvat_task.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _linked_task_public():
    """Linked-task info without the (potentially large) frame map, for the UI."""
    lt = _linked_task()
    return {k: v for k, v in lt.items() if k != "frames"} if lt else None


def _saved_image_index():
    img = STATE.get("image")
    if img and img in IMAGES:
        return IMAGES.index(img)
    return 0


@app.route("/api/meta")
def api_meta():
    return jsonify({"count": len(IMAGES), "path": DATA, "classes": CLASSES,
                    "linked_task": _linked_task_public(),
                    "last_image": _saved_image_index(),
                    "active_class": int(STATE.get("active_class") or 0)})


@app.route("/api/session", methods=["POST"])
def api_session():
    """Persist the working context (current image + active class) so the next
    launch resumes exactly where the user left off."""
    d = request.get_json(force=True, silent=True) or {}
    if "image" in d:
        STATE["image"] = d["image"]
    if "active_class" in d:
        try:
            STATE["active_class"] = int(d["active_class"])
        except (TypeError, ValueError):
            pass
    STATE["path"] = DATA
    _save_state()
    return jsonify({"ok": True})


@app.route("/api/classes", methods=["POST"])
def api_classes():
    """Persist a class list to the folder's labels.txt (used when classes are
    imported from CVAT) so they survive a restart."""
    global CLASSES
    names = (request.get_json(force=True, silent=True) or {}).get("classes") or []
    names = [str(n).strip() for n in names if str(n).strip()]
    if not names:
        return jsonify({"error": "no classes"}), 400
    try:
        with open(os.path.join(DATA, "labels.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    CLASSES = names
    STATE["active_class"] = 0
    _save_state()
    return jsonify({"ok": True, "classes": CLASSES})


@app.route("/api/setfolder", methods=["POST"])
def api_setfolder():
    """Switch the dataset folder at runtime. `path` is a dataset root that
    contains images/ (and optionally labels/ and labels.txt); relative paths
    resolve against the script's directory."""
    global DATA, IMG_DIR, LBL_DIR, IMAGES, CLASSES
    path = (request.get_json(force=True).get("path") or "").strip()
    if not path:
        return jsonify({"error": "empty path"}), 400
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    img = os.path.join(path, "images")
    if not os.path.isdir(img):
        return jsonify({"error": f"no images/ subfolder in {path}"}), 400
    DATA, IMG_DIR, LBL_DIR = path, img, os.path.join(path, "labels")
    IMAGES = list_images()
    CLASSES = read_classes()
    _remember_folder(DATA)          # so the UI reopens this folder next launch
    return jsonify({"ok": True, "count": len(IMAGES),
                    "path": DATA, "classes": CLASSES,
                    "linked_task": _linked_task_public()})


@app.route("/api/item/<int:idx>")
def api_item(idx):
    if idx < 0 or idx >= len(IMAGES):
        return jsonify({"error": "out of range"}), 404
    name = IMAGES[idx]
    return jsonify({
        "idx": idx,
        "name": name,
        "count": len(IMAGES),
        "boxes": read_label(name),
    })


@app.route("/api/image/<int:idx>")
def api_image(idx):
    if idx < 0 or idx >= len(IMAGES):
        return Response("out of range", status=404)
    return send_file(os.path.join(IMG_DIR, IMAGES[idx]))


@app.route("/api/save/<int:idx>", methods=["POST"])
def api_save(idx):
    if idx < 0 or idx >= len(IMAGES):
        return jsonify({"error": "out of range"}), 404
    data = request.get_json(force=True)
    boxes = data.get("boxes", [])
    write_label(IMAGES[idx], boxes)
    return jsonify({"ok": True, "saved": IMAGES[idx], "n": len(boxes)})


@app.route("/api/delete/<int:idx>", methods=["POST"])
def api_delete(idx):
    """Delete an image and its label file from disk, then refresh the list."""
    global IMAGES
    if idx < 0 or idx >= len(IMAGES):
        return jsonify({"error": "out of range"}), 404
    name = IMAGES[idx]
    try:
        img_path = os.path.join(IMG_DIR, name)
        if os.path.exists(img_path):
            os.remove(img_path)
        lbl_path = label_path_for(name)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    IMAGES = list_images()
    return jsonify({"ok": True, "deleted": name, "count": len(IMAGES)})


# --------------------------------------------------------------------------- #
# CVAT upload
# --------------------------------------------------------------------------- #
def _cvat_client():
    """Create an authenticated CVAT SDK client (org-scoped). Caller closes it."""
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        raise RuntimeError("CVAT_URL / CVAT_USERNAME / CVAT_PASSWORD missing in .env")
    from cvat_sdk import make_client
    client = make_client(host=CVAT_URL, credentials=(CVAT_USER, CVAT_PASS))
    if CVAT_ORG:
        client.organization_slug = CVAT_ORG
    return client


def _build_yolo_zip(images, classes, lbl_dir, zip_path):
    """Write a CVAT 'YOLO 1.1' import archive (obj.names/obj.data/train.txt +
    obj_train_data/<stem>.txt). Class indices map to `classes` order, which is
    exactly the labels.txt order."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(classes) + "\n")
        z.writestr("obj.data",
                   f"classes = {len(classes)}\n"
                   "train = data/train.txt\n"
                   "names = data/obj.names\n"
                   "backup = backup/\n")
        train_lines = []
        for img in images:
            stem = os.path.splitext(img)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            content = ""
            if os.path.exists(lbl):
                with open(lbl, encoding="utf-8") as fh:
                    content = fh.read()
            z.writestr(f"obj_train_data/{stem}.txt", content)
            train_lines.append(f"data/obj_train_data/{img}")
        z.writestr("train.txt", "\n".join(train_lines) + "\n")


def _build_update_zip(lbl_dir, classes, frames, subset, zip_path):
    """Build a YOLO 1.1 zip whose frame paths/subset match the CVAT task exactly,
    so importing it updates the right frames. `frames` maps image basename ->
    the task's frame path (e.g. 'ppe-detection/V4.0/PM4 ...jpg')."""
    import zipfile
    sub = subset or "train"
    folder = f"obj_{sub}_data"
    listname = f"{sub}.txt"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(classes) + "\n")
        z.writestr("obj.data",
                   f"classes = {len(classes)}\n"
                   f"{sub} = data/{listname}\n"
                   "names = data/obj.names\n"
                   "backup = backup/\n")
        lines = []
        for base, frame in frames.items():
            stem_base = os.path.splitext(base)[0]
            lp = os.path.join(lbl_dir, stem_base + ".txt")
            content = ""
            if os.path.exists(lp):
                with open(lp, encoding="utf-8") as fh:
                    content = fh.read()
            frame_stem = os.path.splitext(frame)[0]
            z.writestr(f"{folder}/{frame_stem}.txt", content)
            lines.append(f"data/{folder}/{frame}")
        z.writestr(listname, "\n".join(lines) + "\n")


def _cvat_mark_deleted_frames(task_id, frame_paths):
    """Mark frames as deleted on a CVAT task (by matching their name/basename).
    Returns how many new frames were marked."""
    s = _cvat_session()
    meta = s.get(f"{CVAT_URL}/api/tasks/{task_id}/data/meta").json()
    frames = meta.get("frames", [])
    name_to_idx = {f.get("name"): i for i, f in enumerate(frames)}
    base_to_idx = {os.path.basename(f.get("name", "")): i for i, f in enumerate(frames)}
    existing = set(meta.get("deleted_frames") or [])
    new = set()
    for fp in frame_paths:
        idx = name_to_idx.get(fp)
        if idx is None:
            idx = base_to_idx.get(os.path.basename(fp))
        if idx is not None:
            new.add(idx)
    fresh = new - existing
    if fresh:
        s.patch(f"{CVAT_URL}/api/tasks/{task_id}/data/meta",
                json={"deleted_frames": sorted(existing | new)})
    return len(fresh)


def _cvat_import_annotations(client, task_id, zip_path):
    """Upload annotations to an existing task via TUS with location=local (needed
    when the task's storage is cloud-backed). Replicates the SDK uploader but
    adds the location param the high-level helper omits."""
    from pathlib import Path
    from cvat_sdk.core.uploading import AnnotationUploader
    task = client.tasks.retrieve(int(task_id))
    up = AnnotationUploader(client)
    fn = Path(zip_path)
    url = client.api_map.make_endpoint_url(
        task.api.create_annotations_endpoint.path, kwsub={"id": int(task_id)})
    params = {"format": "YOLO 1.1", "filename": fn.name, "location": "local"}
    resp = up.upload_file(url, fn, query_params=params,
                          meta={"filename": params["filename"]})
    rq_id = json.loads(resp.data).get("rq_id")
    if not rq_id:
        raise RuntimeError("no rq_id from annotation upload")
    client.wait_for_completion(rq_id, status_check_period=2)


_cvat_job = {"running": False, "state": "idle", "message": "",
             "task_id": None, "task_url": None, "error": None}
_cvat_job_lock = threading.Lock()


def _set_job(**kw):
    with _cvat_job_lock:
        _cvat_job.update(kw)


def _do_cvat_upload(project_id, task_name, task_id, frames, subset,
                    images, img_dir, lbl_dir, classes):
    import tempfile
    zip_path = None
    try:
        from cvat_sdk.core.proxies.tasks import ResourceType
        import cvat_sdk.models as models

        # ---- update annotations on an EXISTING task (from an import) ----
        if task_id:
            if not classes:
                raise RuntimeError("no labels.txt/classes to upload")
            fmap = frames if frames else {n: n for n in images}
            # images the user deleted locally -> delete those frames in CVAT too
            present = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()
            deleted_paths = [fp for base, fp in fmap.items() if base not in present]
            remaining = {b: fp for b, fp in fmap.items() if b in present}
            ndel = 0
            if deleted_paths:
                _set_job(state="uploading",
                         message=f"removing {len(deleted_paths)} deleted frame(s) from task {task_id}…")
                ndel = _cvat_mark_deleted_frames(task_id, deleted_paths)
            fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_upd_")
            os.close(fd)
            _build_update_zip(lbl_dir, classes, remaining, subset, zip_path)
            _set_job(state="uploading",
                     message=f"updating annotations on task {task_id}…")
            with _cvat_client() as client:
                task = client.tasks.retrieve(int(task_id))
                task.remove_annotations()                 # clean replace
                _cvat_import_annotations(client, int(task_id), zip_path)
            msg = f"updated task {task_id} ✓"
            if ndel:
                msg += f" ({ndel} frame(s) deleted)"
            _set_job(state="done", running=False, task_id=int(task_id),
                     task_url=f"{CVAT_URL}/tasks/{task_id}", message=msg)
            return

        # ---- create a NEW task ----
        if classes:
            fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_yolo_")
            os.close(fd)
            _build_yolo_zip(images, classes, lbl_dir, zip_path)
        image_paths = [os.path.join(img_dir, n) for n in images]
        kwargs = dict(
            spec=models.TaskWriteRequest(name=task_name, project_id=int(project_id)),
            resources=image_paths,
            resource_type=ResourceType.LOCAL,
        )
        if zip_path:
            kwargs["annotation_path"] = zip_path
            kwargs["annotation_format"] = "YOLO 1.1"
            _set_job(state="uploading",
                     message=f"uploading {len(images)} images + annotations to CVAT…")
        else:
            _set_job(state="uploading",
                     message=f"uploading {len(images)} images to CVAT (no labels.txt → no annotations)…")
        with _cvat_client() as client:
            task = client.tasks.create_from_data(**kwargs)
            tid = task.id
        _set_job(state="done", running=False, task_id=tid,
                 task_url=f"{CVAT_URL}/tasks/{tid}",
                 message=f"uploaded as task {tid} ✓")
    except Exception as e:
        _set_job(state="error", running=False, error=str(e),
                 message=f"upload failed: {e}")
    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


# Cached project/task lists (persisted to disk so they survive restarts; the UI
# has Refresh buttons to re-fetch when CVAT changes).
_CVAT_CACHE_FILE = os.path.join(BASE, ".cvat_cache.json")
_cvat_cache = {"projects": None, "tasks": {}}
_cvat_cache_lock = threading.Lock()


def _load_cvat_cache():
    global _cvat_cache
    try:
        with open(_CVAT_CACHE_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            _cvat_cache = {"projects": d.get("projects"), "tasks": d.get("tasks") or {}}
    except (OSError, ValueError):
        pass


def _save_cvat_cache():
    try:
        with open(_CVAT_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(_cvat_cache, fh)
    except OSError:
        pass


_load_cvat_cache()


@app.route("/api/cvat/projects")
def api_cvat_projects():
    """List CVAT projects (id + name). Served from cache unless ?refresh=1."""
    refresh = request.args.get("refresh") == "1"
    with _cvat_cache_lock:
        cached = _cvat_cache.get("projects")
    if not refresh and cached is not None:
        return jsonify({"projects": cached, "org": CVAT_ORG, "url": CVAT_URL, "cached": True})
    try:
        with _cvat_client() as client:
            projects = [{"id": p.id, "name": p.name} for p in client.projects.list()]
        projects.sort(key=lambda p: p["id"], reverse=True)   # newest first
        with _cvat_cache_lock:
            _cvat_cache["projects"] = projects
            _save_cvat_cache()
        return jsonify({"projects": projects, "org": CVAT_ORG, "url": CVAT_URL})
    except Exception as e:
        if cached is not None:                    # fall back to cache on error
            return jsonify({"projects": cached, "org": CVAT_ORG, "url": CVAT_URL,
                            "cached": True, "warn": str(e)})
        return jsonify({"error": str(e)}), 500


@app.route("/api/cvat/projectlabels")
def api_cvat_projectlabels():
    """Class names for a CVAT project (ordered by label id), so the annotator can
    use the project's own classes. The order IS the class id used in labels."""
    pid = request.args.get("project_id")
    if not pid:
        return jsonify({"error": "no project id"}), 400
    try:
        with _cvat_client() as client:
            proj = client.projects.retrieve(int(pid))
            labels = list(proj.get_labels())
        labels.sort(key=lambda l: l.id)
        names = [l.name for l in labels]
        colors = [getattr(l, "color", "") or "" for l in labels]
        return jsonify({"project_id": int(pid), "classes": names, "colors": colors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cvat/upload", methods=["POST"])
def api_cvat_upload():
    """Kick off a background upload of the CURRENT folder's images (+ YOLO labels)
    as a new task under the chosen project."""
    data = request.get_json(force=True)
    task_id = data.get("task_id")             # set => update this existing task
    project_id = data.get("project_id")
    task_name = (data.get("task_name") or "").strip()
    # classes the boxes were drawn against (from CVAT project or labels.txt);
    # falls back to the folder's labels.txt if the client didn't send any.
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        classes = list(CLASSES)
    if task_id:
        if not classes:
            return jsonify({"error": "no labels to upload"}), 400
    else:
        if not project_id:
            return jsonify({"error": "no project selected"}), 400
        if not task_name:
            return jsonify({"error": "task name is required"}), 400
    if not IMAGES:
        return jsonify({"error": "no images in the current folder"}), 400
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        return jsonify({"error": "CVAT credentials missing in .env"}), 400
    with _cvat_job_lock:
        if _cvat_job["running"]:
            return jsonify({"error": "an upload is already running"}), 409
        _cvat_job.update(running=True, state="starting", message="preparing upload…",
                         task_id=None, task_url=None, error=None)
    # for an update, pull the frame map / subset recorded at import time
    frames = subset = None
    if task_id:
        linked = _linked_task() or {}
        frames = linked.get("frames")
        subset = linked.get("subset")
    # snapshot the current dataset so a folder switch mid-upload can't corrupt it
    args = (project_id, task_name, task_id, frames, subset,
            list(IMAGES), IMG_DIR, LBL_DIR, list(classes))
    threading.Thread(target=_do_cvat_upload, args=args, daemon=True).start()
    return jsonify({"started": True, "count": len(IMAGES),
                    "annotations": bool(classes), "update": bool(task_id)})


@app.route("/api/cvat/uploadstatus")
def api_cvat_uploadstatus():
    with _cvat_job_lock:
        return jsonify(dict(_cvat_job))


# --------------------------------------------------------------------------- #
# CVAT import (download a task's images + annotations, open it locally)
# --------------------------------------------------------------------------- #
def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "task"


def _norm_dt(v):
    """Normalise a CVAT updated_date (datetime or string) to a comparable string."""
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        v = v.isoformat()
    return str(v).replace("Z", "+00:00")


def _local_task_status(task_id, updated=None):
    """Whether a task already has a local imported copy, and if it's current.
    Cheap (local filesystem only) — no CVAT calls."""
    dirs = _local_task_dirs(task_id)
    if not dirs:
        return {"imported": False, "up_to_date": False, "local_path": None}
    local_path = dirs[0]
    # Without a reference updated_date we can't judge freshness -> assume current
    # (don't show a false "update available"); checking on open makes it exact.
    if not updated:
        return {"imported": True, "up_to_date": True, "local_path": local_path}
    up_to_date = False
    for d in dirs:
        try:
            with open(os.path.join(d, ".cvat_task.json"), encoding="utf-8") as fh:
                link = json.load(fh)
        except (OSError, ValueError):
            continue
        if _norm_dt(link.get("updated_date")) == _norm_dt(updated):
            up_to_date = True
            break
    return {"imported": True, "up_to_date": up_to_date, "local_path": local_path}


def _annotate_imports(tasks):
    """Return task dicts tagged with current local-import status (fresh each call)."""
    return [{**t, **_local_task_status(t.get("id"), t.get("updated_date"))} for t in tasks]


def _cvat_list_tasks(project_id):
    with _cvat_client() as client:
        out, page = [], 1
        while True:
            data, _ = client.api_client.tasks_api.list(
                project_id=int(project_id), page=page, page_size=100)
            for t in data.results:
                out.append({"id": t.id, "name": t.name, "size": getattr(t, "size", None),
                            "updated_date": _norm_dt(getattr(t, "updated_date", None))})
            if not getattr(data, "next", None):
                break
            page += 1
        return out


@app.route("/api/cvat/tasks")
def api_cvat_tasks():
    """Tasks for a project. Served from cache unless ?refresh=1."""
    pid = request.args.get("project_id")
    if not pid:
        return jsonify({"error": "no project id"}), 400
    refresh = request.args.get("refresh") == "1"
    key = str(pid)
    with _cvat_cache_lock:
        cached = _cvat_cache["tasks"].get(key)
    if not refresh and cached is not None:
        return jsonify({"tasks": _annotate_imports(cached), "cached": True})
    try:
        tasks = _cvat_list_tasks(pid)
        tasks.sort(key=lambda t: t["id"], reverse=True)   # newest first
        with _cvat_cache_lock:
            _cvat_cache["tasks"][key] = tasks
            _save_cvat_cache()
        return jsonify({"tasks": _annotate_imports(tasks)})
    except Exception as e:
        if cached is not None:
            return jsonify({"tasks": _annotate_imports(cached), "cached": True, "warn": str(e)})
        return jsonify({"error": str(e)}), 500


_imp_job = {"running": False, "state": "idle", "message": "",
            "path": None, "count": 0, "error": None}
_imp_lock = threading.Lock()


def _set_imp(**kw):
    with _imp_lock:
        _imp_job.update(kw)


def _extract_yolo_export(zip_path, out_dir):
    """Unpack a CVAT 'YOLO 1.1' (with images) export into images/ + labels/ +
    labels.txt. Also returns the frame paths (relative to obj_<subset>_data, as
    CVAT names its frames) and the subset name, so an update can rebuild a zip
    that matches the task's frames exactly."""
    import zipfile
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cvat_imp_")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        img_out = os.path.join(out_dir, "images")
        lbl_out = os.path.join(out_dir, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        names = []
        for root, _, files in os.walk(tmp):
            if "obj.names" in files:
                with open(os.path.join(root, "obj.names"), encoding="utf-8") as fh:
                    names = [ln.strip() for ln in fh if ln.strip()]
                break
        if names:
            with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(names) + "\n")
        frames = {}       # image basename -> frame path inside obj_<subset>_data
        subset = None
        n = 0
        for root, _, files in os.walk(tmp):
            for f in files:
                if os.path.splitext(f)[1].lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                full = os.path.join(root, f)
                parts = os.path.relpath(full, tmp).split(os.sep)
                frame = f
                for i, p in enumerate(parts):
                    pl = p.lower()
                    if pl.startswith("obj_") and pl.endswith("_data"):
                        if subset is None:
                            subset = p[4:-5]           # 'obj_Train_data' -> 'Train'
                        frame = "/".join(parts[i + 1:])
                        break
                shutil.copy(full, os.path.join(img_out, f))
                stem = os.path.splitext(f)[0]
                lp = os.path.join(root, os.path.splitext(os.path.basename(full))[0] + ".txt")
                if os.path.exists(lp):
                    shutil.copy(lp, os.path.join(lbl_out, stem + ".txt"))
                frames[f] = frame
                n += 1
        return n, names, frames, subset
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def _read_export_labels(zip_path):
    """From an annotations-only YOLO export zip, return {image_stem: label_text}
    for every .txt under obj_<subset>_data/, plus '__names__' -> obj.names list."""
    import zipfile
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cvat_lbl_")
    out = {}
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        for root, _, files in os.walk(tmp):
            if "obj.names" in files:
                with open(os.path.join(root, "obj.names"), encoding="utf-8") as fh:
                    out["__names__"] = [ln.strip() for ln in fh if ln.strip()]
                break
        for root, _, files in os.walk(tmp):
            parts = os.path.relpath(root, tmp).split(os.sep)
            inside = any(p.lower().startswith("obj_") and p.lower().endswith("_data")
                         for p in parts)
            if not inside:
                continue                       # skip train.txt / data.yaml at the root
            for f in files:
                if not f.lower().endswith(".txt"):
                    continue
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    out[os.path.splitext(f)[0]] = fh.read()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _update_labels_only(task_id, status=None):
    """Refresh ONLY the label .txt files (and labels.txt) of an existing local
    copy from CVAT — images are left untouched. Returns {out_dir, updated}.
    `updated` is False when the local copy already matched CVAT (no download)."""
    import tempfile
    def say(m):
        if status:
            status(m)
    dirs = _local_task_dirs(task_id)
    if not dirs:
        raise RuntimeError("task is not imported locally")
    out_dir = dirs[0]
    say("checking task…")
    session = _cvat_session()
    info = session.get(f"{CVAT_URL}/api/tasks/{task_id}").json()
    updated = info.get("updated_date")
    link_path = os.path.join(out_dir, ".cvat_task.json")
    try:
        with open(link_path, encoding="utf-8") as fh:
            link = json.load(fh)
    except (OSError, ValueError):
        link = {}
    if updated and _norm_dt(link.get("updated_date")) == _norm_dt(updated):
        return {"out_dir": out_dir, "updated": False}        # already current
    fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_lblexp_")
    os.close(fd)
    try:
        say("downloading annotations…")
        _cvat_export_task(session, task_id, zip_path, save_images=False)
        say("writing labels…")
        labels = _read_export_labels(zip_path)
        names = labels.pop("__names__", None)
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(lbl_dir, exist_ok=True)
        # rewrite every image's label to mirror CVAT exactly (handles removals)
        for img in os.listdir(img_dir):
            if os.path.splitext(img)[1].lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = os.path.splitext(img)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(labels.get(stem, ""))
        if names:
            with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(names) + "\n")
        link["updated_date"] = updated                       # mark our copy current
        try:
            with open(link_path, "w", encoding="utf-8") as fh:
                json.dump(link, fh)
        except OSError:
            pass
        return {"out_dir": out_dir, "updated": True}
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def _cvat_session():
    """Authenticated requests session (org-scoped). The REST export API is more
    reliable across server versions than the SDK's export_dataset()."""
    import requests
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        raise RuntimeError("CVAT credentials missing in .env")
    s = requests.Session()
    if CVAT_ORG:
        s.headers.update({"X-Organization": CVAT_ORG})
    s.headers.update({"Referer": CVAT_URL})
    r = s.post(f"{CVAT_URL}/api/auth/login",
               json={"username": CVAT_USER, "password": CVAT_PASS},
               headers={"Content-Type": "application/json"})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"CVAT login failed: {r.status_code}")
    csrf = s.cookies.get("csrftoken")
    if csrf:
        s.headers.update({"X-CSRFToken": csrf})
    return s


def _cvat_export_task(session, task_id, zip_path, save_images=True):
    """Export a task as 'YOLO 1.1' and download the zip. With save_images=False
    only the annotation .txt files are exported (no images) — used for fast
    labels-only refreshes."""
    import time
    # location=local is required for app.cvat.ai to populate result_url
    r = session.post(f"{CVAT_URL}/api/tasks/{task_id}/dataset/export",
                     params={"format": "YOLO 1.1",
                             "save_images": "true" if save_images else "false",
                             "location": "local"})
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"export init failed {r.status_code}: {r.text[:160]}")
    rq_id = r.json().get("rq_id")
    if not rq_id:
        raise RuntimeError(f"no rq_id from export: {r.text[:160]}")
    waited = 0
    while waited < 1800:
        st = session.get(f"{CVAT_URL}/api/requests/{rq_id}").json()
        status = (st.get("status") or "").lower()
        if status == "finished":
            url = st.get("result_url")
            if not url:
                raise RuntimeError("export finished but no result_url")
            if not url.startswith("http"):
                url = f"{CVAT_URL}{url}"
            resp = session.get(url, stream=True)
            if resp.status_code != 200:
                raise RuntimeError(f"download failed {resp.status_code}")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            return
        if status == "failed":
            raise RuntimeError(f"export failed: {st.get('message')}")
        time.sleep(3)
        waited += 3
    raise RuntimeError("export timed out")


def _local_task_dirs(task_id):
    """Existing imported folders for a task id (have an images/ subdir)."""
    return [d for d in glob.glob(os.path.join(IMPORTS_DIR, f"{task_id}_*"))
            if os.path.isdir(os.path.join(d, "images"))]


def _classes_in(d):
    try:
        with open(os.path.join(d, "labels.txt"), encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        return []


def _cvat_import_task(task_id, status=None, force=False):
    """Return a local copy of a task as {out_dir, count, classes, frames, subset,
    name, project_id, cached}. Re-uses an existing folder if it's up to date with
    CVAT (matching updated_date); otherwise downloads fresh. `force` always
    downloads. `status` is an optional callable(message) for progress."""
    import tempfile
    import shutil
    def say(m):
        if status:
            status(m)
    say("checking task…")
    session = _cvat_session()
    info = session.get(f"{CVAT_URL}/api/tasks/{task_id}").json()
    tname = info.get("name") or f"task_{task_id}"
    updated = info.get("updated_date")
    # reuse an up-to-date local copy (no download)
    if not force:
        for d in _local_task_dirs(task_id):
            try:
                with open(os.path.join(d, ".cvat_task.json"), encoding="utf-8") as fh:
                    link = json.load(fh)
            except (OSError, ValueError):
                link = {}
            if updated and link.get("updated_date") == updated:
                imgs = os.listdir(os.path.join(d, "images"))
                return {"out_dir": d, "count": len(imgs), "classes": _classes_in(d),
                        "frames": link.get("frames") or {}, "subset": link.get("subset"),
                        "name": tname, "project_id": info.get("project_id"), "cached": True}
    # download fresh
    zip_path = None
    try:
        fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_task_")
        os.close(fd)
        say(f"downloading '{tname}' (images + annotations)…")
        _cvat_export_task(session, task_id, zip_path)
        for d in glob.glob(os.path.join(IMPORTS_DIR, f"{task_id}_*")):  # drop stale copies
            shutil.rmtree(d, ignore_errors=True)
        out_dir = os.path.join(IMPORTS_DIR, f"{task_id}_{_safe_name(tname)}")
        os.makedirs(out_dir, exist_ok=True)
        say("extracting…")
        n, names, frames, subset = _extract_yolo_export(zip_path, out_dir)
        try:
            with open(os.path.join(out_dir, ".cvat_task.json"), "w", encoding="utf-8") as fh:
                json.dump({"task_id": int(task_id), "task_name": tname,
                           "project_id": info.get("project_id"), "frames": frames,
                           "subset": subset, "updated_date": updated}, fh)
        except OSError:
            pass
        return {"out_dir": out_dir, "count": n, "classes": names, "frames": frames,
                "subset": subset, "name": tname, "project_id": info.get("project_id"),
                "cached": False}
    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def _do_cvat_import(task_id, force=False, labels_only=False):
    try:
        _set_imp(state="exporting", message="checking task…")
        if labels_only:
            # refresh annotations in place; keep the already-downloaded images
            r = _update_labels_only(task_id, status=lambda m: _set_imp(message=m))
            cnt = len([f for f in os.listdir(os.path.join(r["out_dir"], "images"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")])
            msg = ("labels already up to date ✓" if not r.get("updated")
                   else "labels updated from CVAT ✓ (images kept)")
            _set_imp(running=False, state="done", path=r["out_dir"], count=cnt, message=msg)
            return
        r = _cvat_import_task(task_id, status=lambda m: _set_imp(message=m), force=force)
        msg = (f"already downloaded & up to date — {r['count']} images ✓"
               if r.get("cached")
               else f"downloaded {r['count']} images, {len(r['classes'])} classes ✓")
        _set_imp(running=False, state="done", path=r["out_dir"], count=r["count"], message=msg)
    except Exception as e:
        _set_imp(running=False, state="error", error=str(e),
                 message=f"import failed: {e}")


@app.route("/api/cvat/import", methods=["POST"])
def api_cvat_import():
    data = request.get_json(force=True)
    task_id = data.get("task_id")
    force = bool(data.get("force"))
    labels_only = bool(data.get("labels_only"))
    if not task_id:
        return jsonify({"error": "no task selected"}), 400
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        return jsonify({"error": "CVAT credentials missing in .env"}), 400
    with _imp_lock:
        if _imp_job["running"]:
            return jsonify({"error": "an import is already running"}), 409
        _imp_job.update(running=True, state="starting", message="starting…",
                        path=None, count=0, error=None)
    threading.Thread(target=_do_cvat_import, args=(task_id,),
                     kwargs={"force": force, "labels_only": labels_only},
                     daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/cvat/taskstatus")
def api_cvat_taskstatus():
    """For an already-imported task: is the local copy still current with CVAT?
    Makes a single lightweight CVAT call. Used when opening an imported task."""
    tid = request.args.get("task_id")
    if not tid:
        return jsonify({"error": "no task id"}), 400
    if not _local_task_dirs(tid):
        return jsonify({"imported": False})
    try:
        session = _cvat_session()
        info = session.get(f"{CVAT_URL}/api/tasks/{tid}").json()
        updated = info.get("updated_date")
    except Exception as e:
        return jsonify({"imported": True, "up_to_date": None, "error": str(e)})
    st = _local_task_status(tid, updated)
    return jsonify({"imported": True, "up_to_date": st["up_to_date"],
                    "updated_date": updated})


@app.route("/api/cvat/importstatus")
def api_cvat_importstatus():
    with _imp_lock:
        return jsonify(dict(_imp_job))


# --------------------------------------------------------------------------- #
# Update-all: refresh every imported task in a project that's out of date.
# Reuses _cvat_import_task (non-force), which skips copies already current.
# --------------------------------------------------------------------------- #
_updall_job = {"running": False, "state": "idle", "message": "", "total": 0,
               "done": 0, "updated": 0, "uptodate": 0, "failed": 0, "error": None}
_updall_lock = threading.Lock()


def _set_updall(**kw):
    with _updall_lock:
        _updall_job.update(kw)


def _do_update_all(task_ids):
    try:
        total = len(task_ids)
        updated = uptodate = failed = 0
        for i, tid in enumerate(task_ids, 1):
            _set_updall(done=i - 1, message=f"checking task {tid} ({i}/{total})…")
            try:
                # labels-only: refresh annotations in place, never re-download images
                r = _update_labels_only(
                    tid, status=lambda m: _set_updall(message=f"[{i}/{total}] {m}"))
                if r.get("updated"):
                    updated += 1
                else:
                    uptodate += 1
            except Exception as e:
                failed += 1
                _set_updall(message=f"task {tid} failed: {e}")
            _set_updall(done=i, updated=updated, uptodate=uptodate, failed=failed)
        msg = f"done ✓ — {updated} updated, {uptodate} already current"
        if failed:
            msg += f", {failed} failed"
        _set_updall(running=False, state="done", message=msg)
    except Exception as e:
        _set_updall(running=False, state="error", error=str(e),
                    message=f"update-all failed: {e}")


@app.route("/api/cvat/updateall", methods=["POST"])
def api_cvat_updateall():
    data = request.get_json(force=True) or {}
    task_ids = [t for t in (data.get("task_ids") or []) if _local_task_dirs(t)]
    if not task_ids:
        return jsonify({"error": "no imported tasks to update"}), 400
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        return jsonify({"error": "CVAT credentials missing in .env"}), 400
    with _updall_lock:
        if _updall_job["running"]:
            return jsonify({"error": "an update is already running"}), 409
        _updall_job.update(running=True, state="starting", message="starting…",
                           total=len(task_ids), done=0, updated=0, uptodate=0,
                           failed=0, error=None)
    threading.Thread(target=_do_update_all, args=(task_ids,), daemon=True).start()
    return jsonify({"started": True, "count": len(task_ids)})


@app.route("/api/cvat/updateall_status")
def api_cvat_updateall_status():
    with _updall_lock:
        return jsonify(dict(_updall_job))


# --------------------------------------------------------------------------- #
# Auto-annotation pipeline: import task -> run model -> update CVAT
# --------------------------------------------------------------------------- #
_ap_job = {"running": False, "state": "idle", "message": "", "done": 0, "total": 0,
           "cur_task": 0, "n_tasks": 0, "task_id": None, "task_url": None,
           "added": 0, "done_tasks": 0, "error": None}
_ap_lock = threading.Lock()


def _set_ap(**kw):
    with _ap_lock:
        _ap_job.update(kw)


def _autopipeline_one(task_id, model_path, conf, mode, name_mapping, label):
    """Import one task, run the model, and push annotations back. Returns the
    number of boxes added. `label` prefixes progress messages (e.g. 'task 2/5')."""
    import tempfile
    zip_path = None
    try:
        _set_ap(state="importing", message=f"{label}: importing…")
        r = _cvat_import_task(task_id, status=lambda m: _set_ap(message=f"{label} import: {m}"))
        out_dir = r["out_dir"]
        classes = r["classes"]
        frames = r["frames"]
        subset = r["subset"]
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        images = sorted(os.listdir(img_dir)) if os.path.isdir(img_dir) else []
        if not classes:
            raise RuntimeError(f"task {task_id} has no classes (labels.txt)")
        name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(classes)}
        idx_map = {}
        for k, v in (name_mapping or {}).items():
            try:
                mi = int(k)
            except (TypeError, ValueError):
                continue
            ci = name_to_idx.get(str(v).strip().lower())
            if ci is not None:
                idx_map[mi] = ci
        _set_ap(state="annotating", message=f"{label}: loading model…", total=len(images), done=0)
        _load_model(model_path)
        added = 0
        for i, img in enumerate(images):
            _set_ap(message=f"{label}: annotating {i+1}/{len(images)}…", done=i)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img)[0] + ".txt")
            has_existing = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
            if mode == "skip" and has_existing:
                continue
            try:
                dets, _ = _infer(model_path, os.path.join(img_dir, img), conf)
            except Exception:
                continue
            new_boxes = []
            for d in dets:
                ci = idx_map.get(d.get("cls_model"))
                if ci is None:
                    continue
                new_boxes.append({"cls": ci, "cx": d["cx"], "cy": d["cy"],
                                  "w": d["w"], "h": d["h"]})
            boxes = (_read_label_file(lbl_path) + new_boxes) if mode == "append" else new_boxes
            _write_label_file(lbl_path, boxes)
            added += len(new_boxes)
        _set_ap(state="uploading", message=f"{label}: updating CVAT…", done=len(images))
        fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_ap_")
        os.close(fd)
        fmap = frames if frames else {n: n for n in images}
        _build_update_zip(lbl_dir, classes, fmap, subset, zip_path)
        with _cvat_client() as client:
            task = client.tasks.retrieve(int(task_id))
            task.remove_annotations()
            _cvat_import_annotations(client, int(task_id), zip_path)
        return added
    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def _do_cvat_autopipeline(task_ids, model_path, conf, mode, name_mapping):
    try:
        n = len(task_ids)
        total_added = 0
        for ti, task_id in enumerate(task_ids):
            _set_ap(cur_task=ti + 1, n_tasks=n, done_tasks=ti)
            total_added += _autopipeline_one(
                task_id, model_path, conf, mode, name_mapping, f"task {ti+1}/{n}")
        last = task_ids[-1] if task_ids else None
        _set_ap(running=False, state="done", done_tasks=n, added=total_added,
                task_id=int(last) if last is not None else None,
                task_url=f"{CVAT_URL}/tasks/{last}" if last is not None else None,
                message=f"done ✓ added {total_added} boxes across {n} task(s)")
    except Exception as e:
        _set_ap(running=False, state="error", error=str(e), message=f"failed: {e}")


@app.route("/api/cvat/autopipeline", methods=["POST"])
def api_cvat_autopipeline():
    data = request.get_json(force=True)
    task_ids = data.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        single = data.get("task_id")
        task_ids = [single] if single else []
    task_ids = [t for t in task_ids if t]
    model_path = _safe_model_path(data.get("model", ""))
    if not task_ids:
        return jsonify({"error": "no tasks selected"}), 400
    if not model_path:
        return jsonify({"error": "invalid model"}), 400
    try:
        conf = float(data.get("conf", 0.25) or 0.25)
    except (TypeError, ValueError):
        conf = 0.25
    mode = data.get("mode")
    if mode not in ("skip", "append", "replace"):
        mode = "append"
    mapping = data.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return jsonify({"error": "map at least one class"}), 400
    with _ap_lock:
        if _ap_job["running"]:
            return jsonify({"error": "a pipeline run is already going"}), 409
        _ap_job.update(running=True, state="starting", message="starting…", done=0,
                       total=0, cur_task=0, n_tasks=len(task_ids), task_id=None,
                       task_url=None, added=0, done_tasks=0, error=None)
    args = (task_ids, model_path, conf, mode, mapping)
    threading.Thread(target=_do_cvat_autopipeline, args=args, daemon=True).start()
    return jsonify({"started": True, "count": len(task_ids)})


@app.route("/api/cvat/autopipeline_status")
def api_cvat_autopipeline_status():
    with _ap_lock:
        return jsonify(dict(_ap_job))


# --------------------------------------------------------------------------- #
# Class count: per-class annotation counts in a project (project- and task-wise)
# --------------------------------------------------------------------------- #
def _cvat_get(session, url, params=None, retries=10, on_throttle=None):
    """GET that respects CVAT's rate limit (HTTP 429): on throttling it waits the
    server-suggested Retry-After and retries. app.cvat.ai throttles ~20 req/min,
    so without this a project's per-job annotation fetches get dropped."""
    import time
    resp = None
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=180)
        if resp.status_code != 429:
            return resp
        wait = resp.headers.get("Retry-After")
        try:
            wait = float(wait)
        except (TypeError, ValueError):
            wait = min(2 ** attempt, 30)
        wait = min(max(wait, 1), 60) + 0.5
        if on_throttle:
            on_throttle(wait)
        time.sleep(wait)
    return resp


def _cvat_paginated(session, url, params=None):
    params = dict(params or {})
    params.setdefault("page_size", 100)
    out, page = [], 1
    while True:
        params["page"] = page
        resp = _cvat_get(session, url, params=params)
        if resp.status_code != 200:
            break
        d = resp.json()
        out += d.get("results", [])
        if not d.get("next"):
            break
        page += 1
    return out


_cc_job = {"running": False, "state": "idle", "message": "", "done": 0, "total": 0,
           "error": None, "result": None}
_cc_lock = threading.Lock()


def _set_cc(**kw):
    with _cc_lock:
        _cc_job.update(kw)


def _do_classcount(project_id):
    from collections import defaultdict
    try:
        _set_cc(state="loading", message="loading labels, tasks & jobs…")
        s = _cvat_session()
        labels = _cvat_paginated(s, f"{CVAT_URL}/api/labels", {"project_id": project_id})
        id_to_name = {l["id"]: l["name"] for l in labels}
        tasks = _cvat_paginated(s, f"{CVAT_URL}/api/tasks", {"project_id": project_id})
        task_list = [{"id": t["id"], "name": t.get("name") or f"task_{t['id']}"} for t in tasks]
        # Count from JOBS, not the task-level /annotations endpoint (which returns
        # nothing for many tasks on this server). Skip ground-truth (validation)
        # jobs so only real annotation work is counted.
        jobs = _cvat_paginated(s, f"{CVAT_URL}/api/jobs", {"project_id": project_id})
        ann_jobs = [j for j in jobs if j.get("type") != "ground_truth"]
        m = len(ann_jobs)
        counts = {name: {"total": 0, "tasks": {}} for name in id_to_name.values()}
        task_totals = defaultdict(int)
        for k, j in enumerate(ann_jobs):
            tid = j.get("task_id")
            _set_cc(state="counting", done=k, total=m,
                    message=f"counting job {k+1}/{m} (task {tid})…")
            try:
                resp = _cvat_get(
                    s, f"{CVAT_URL}/api/jobs/{j['id']}/annotations",
                    on_throttle=lambda w, k=k: _set_cc(
                        message=f"CVAT rate limit — waiting {int(w)}s… (job {k+1}/{m})"))
                if resp.status_code != 200:
                    continue
                ann = resp.json()
            except Exception:
                continue
            per = defaultdict(int)
            for sh in ann.get("shapes", []):
                per[sh["label_id"]] += 1
            for tg in ann.get("tags", []):
                per[tg["label_id"]] += 1
            for tr in ann.get("tracks", []):       # each track keyframe = one instance
                per[tr["label_id"]] += len(tr.get("shapes", []))
            for lid, c in per.items():
                name = id_to_name.get(lid, f"label_{lid}")
                counts.setdefault(name, {"total": 0, "tasks": {}})
                counts[name]["total"] += c
                counts[name]["tasks"][str(tid)] = counts[name]["tasks"].get(str(tid), 0) + c
                task_totals[str(tid)] += c
        classes = sorted(counts.keys(), key=lambda x: x.lower())
        grand = sum(c["total"] for c in counts.values())
        result = {"classes": classes, "tasks": task_list, "counts": counts,
                  "task_totals": dict(task_totals), "grand_total": grand,
                  "project_id": int(project_id)}
        _set_cc(running=False, state="done", done=m, result=result,
                message=f"done ✓ {grand} annotations across {len(task_list)} tasks, {len(classes)} classes")
    except Exception as e:
        _set_cc(running=False, state="error", error=str(e), message=f"failed: {e}")


@app.route("/api/cvat/classcount", methods=["POST"])
def api_cvat_classcount():
    data = request.get_json(force=True)
    project_id = data.get("project_id")
    if not project_id:
        return jsonify({"error": "no project selected"}), 400
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        return jsonify({"error": "CVAT credentials missing in .env"}), 400
    with _cc_lock:
        if _cc_job["running"]:
            return jsonify({"error": "a count is already running"}), 409
        _cc_job.update(running=True, state="starting", message="starting…",
                       done=0, total=0, error=None, result=None)
    threading.Thread(target=_do_classcount, args=(project_id,), daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/cvat/classcount_status")
def api_cvat_classcount_status():
    with _cc_lock:
        return jsonify(dict(_cc_job))


# --------------------------------------------------------------------------- #
# Automatic annotation (YOLO models from MODELS_DIR)
# --------------------------------------------------------------------------- #
_models_cache = {}
_models_lock = threading.Lock()


def list_models():
    """All .pt files under MODELS_DIR. `name` is the top-level folder (e.g.
    'head_v2'); collisions get the file stem appended to stay unique."""
    from collections import Counter
    raw = []
    if os.path.isdir(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if f.lower().endswith(".pt"):
                    rel = os.path.relpath(os.path.join(root, f), MODELS_DIR)
                    parts = rel.split(os.sep)
                    disp = parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]
                    raw.append({"path": rel, "name": disp,
                                "stem": os.path.splitext(f)[0]})
    dup = Counter(m["name"] for m in raw)
    for m in raw:
        if dup[m["name"]] > 1:
            m["name"] = f'{m["name"]} / {m["stem"]}'
    out = [{"path": m["path"], "name": m["name"]} for m in raw]
    out.sort(key=lambda m: m["name"].lower())
    return out


def _safe_model_path(rel):
    """Resolve a model path inside MODELS_DIR (no traversal)."""
    if not rel:
        return None
    base = os.path.abspath(MODELS_DIR)
    full = os.path.abspath(os.path.join(MODELS_DIR, rel))
    try:
        if os.path.commonpath([base, full]) != base:
            return None
    except ValueError:
        return None
    return full if os.path.isfile(full) else None


def _load_model(path):
    with _models_lock:
        m = _models_cache.get(path)
        if m is None:
            from ultralytics import YOLO
            m = YOLO(path)
            _models_cache[path] = m
        return m


def _infer(model_path, img_path, conf):
    """Run a model on one image. Returns (detections, model_names) where each
    detection is {name, cls_model, cx, cy, w, h, conf} (normalised box)."""
    model = _load_model(model_path)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    res = model.predict(img_path, conf=conf, imgsz=640, verbose=False)
    dets = []
    if res and res[0].boxes is not None and len(res[0].boxes):
        b = res[0].boxes
        xywhn = b.xywhn.cpu().numpy()
        cls = b.cls.cpu().numpy() if b.cls is not None else [0] * len(xywhn)
        confs = b.conf.cpu().numpy() if b.conf is not None else [1.0] * len(xywhn)
        for (xc, yc, w, h), c, cf in zip(xywhn, cls, confs):
            ci = int(c)
            dets.append({"name": names.get(ci, str(ci)), "cls_model": ci,
                         "cx": float(xc), "cy": float(yc),
                         "w": float(w), "h": float(h), "conf": float(cf)})
    model_names = [names[k] for k in sorted(names)]
    return dets, model_names


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models(), "dir": MODELS_DIR})


@app.route("/api/modelclasses")
def api_modelclasses():
    """Class names of a model (ordered by index), for the mapping step."""
    model_path = _safe_model_path(request.args.get("model", ""))
    if not model_path:
        return jsonify({"error": "invalid model"}), 400
    try:
        model = _load_model(model_path)
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        return jsonify({"classes": [names[k] for k in sorted(names)]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/autoannotate/<int:idx>")
def api_autoannotate(idx):
    """Run the chosen model on ONE image; return detections for the client to
    map (by class name) onto the active class list and review."""
    if not (0 <= idx < len(IMAGES)):
        return jsonify({"error": "out of range"}), 404
    model_path = _safe_model_path(request.args.get("model", ""))
    if not model_path:
        return jsonify({"error": "invalid model"}), 400
    try:
        conf = float(request.args.get("conf", "0.25") or 0.25)
    except ValueError:
        conf = 0.25
    try:
        dets, model_names = _infer(model_path, os.path.join(IMG_DIR, IMAGES[idx]), conf)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"idx": idx, "detections": dets, "model_names": model_names})


_aa_job = {"running": False, "state": "idle", "done": 0, "total": 0,
           "written": 0, "skipped": 0, "unmapped": 0, "message": "", "error": None}
_aa_lock = threading.Lock()


def _set_aa(**kw):
    with _aa_lock:
        _aa_job.update(kw)


def _do_autoannotate_all(model_specs, images, img_dir, lbl_dir, classes, conf, mode):
    """model_specs: list of (model_path, mapping) — each model's class-index ->
    here class-index. All selected models run on every image; their detections
    are pooled, then the mode (append/replace/skip) is applied once per image."""
    global CLASSES
    try:
        # if the folder has no classes, adopt the first model's and persist labels.txt
        if not classes and model_specs:
            m0 = _load_model(model_specs[0][0])
            names0 = m0.names if isinstance(m0.names, dict) else dict(enumerate(m0.names))
            classes = [names0[k] for k in sorted(names0)]
            data_dir = os.path.dirname(img_dir)
            try:
                with open(os.path.join(data_dir, "labels.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(classes) + "\n")
                if data_dir == DATA:
                    CLASSES = list(classes)
            except OSError:
                pass
        name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(classes)}
        for mp, _ in model_specs:           # warm each model once up front
            _load_model(mp)
        written = skipped = unmapped = added = 0
        for i, img in enumerate(images):
            _set_aa(done=i, message=f"annotating {i+1}/{len(images)}…")
            stem = os.path.splitext(img)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            has_existing = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
            if mode == "skip" and has_existing:
                skipped += 1
                continue
            new_boxes = []
            for mp, mapping in model_specs:            # pool detections from all models
                try:
                    dets, _ = _infer(mp, os.path.join(img_dir, img), conf)
                except Exception:
                    continue
                for d in dets:
                    if mapping:
                        ci = mapping.get(d.get("cls_model"))
                    else:                              # no map -> match by class name
                        ci = name_to_idx.get(str(d["name"]).strip().lower())
                    if ci is None or not (0 <= ci < len(classes)):
                        unmapped += 1
                        continue
                    new_boxes.append({"cls": ci, "cx": d["cx"], "cy": d["cy"],
                                      "w": d["w"], "h": d["h"]})
            # append keeps the existing annotations and adds the detections
            boxes = (_read_label_file(lbl_path) + new_boxes) if mode == "append" else new_boxes
            _write_label_file(lbl_path, boxes)
            written += 1
            added += len(new_boxes)
        msg = f"done ✓ {written} images, {added} boxes added"
        if skipped:
            msg += f", {skipped} skipped (already labelled)"
        if unmapped:
            msg += f", {unmapped} dets unmapped"
        _set_aa(running=False, state="done", done=len(images),
                written=written, skipped=skipped, unmapped=unmapped, message=msg)
    except Exception as e:
        _set_aa(running=False, state="error", error=str(e), message=f"failed: {e}")


def _parse_mapping(raw):
    m = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                m[int(k)] = int(v)
            except (TypeError, ValueError):
                pass
    return m


@app.route("/api/autoannotate_all", methods=["POST"])
def api_autoannotate_all():
    data = request.get_json(force=True)
    if not IMAGES:
        return jsonify({"error": "no images in the current folder"}), 400
    # new multi-model form: models=[{model, mapping}]; also accept single model+mapping
    model_specs = []
    raw_models = data.get("models")
    if isinstance(raw_models, list) and raw_models:
        for m in raw_models:
            mp = _safe_model_path((m or {}).get("model", ""))
            if mp:
                model_specs.append((mp, _parse_mapping((m or {}).get("mapping"))))
    else:
        mp = _safe_model_path(data.get("model", ""))
        if mp:
            model_specs.append((mp, _parse_mapping(data.get("mapping"))))
    if not model_specs:
        return jsonify({"error": "select at least one model"}), 400
    try:
        conf = float(data.get("conf", 0.25) or 0.25)
    except (TypeError, ValueError):
        conf = 0.25
    classes = data.get("classes")
    if not isinstance(classes, list):
        classes = list(CLASSES)
    mode = data.get("mode")
    if mode not in ("skip", "append", "replace"):
        mode = "append"
    with _aa_lock:
        if _aa_job["running"]:
            return jsonify({"error": "an auto-annotation run is already going"}), 409
        _aa_job.update(running=True, state="starting", done=0, total=len(IMAGES),
                       written=0, skipped=0, unmapped=0,
                       message="loading models…", error=None)
    args = (model_specs, list(IMAGES), IMG_DIR, LBL_DIR, list(classes), conf, mode)
    threading.Thread(target=_do_autoannotate_all, args=args, daemon=True).start()
    return jsonify({"started": True, "count": len(IMAGES), "models": len(model_specs)})


@app.route("/api/autoannotate_status")
def api_autoannotate_status():
    with _aa_lock:
        return jsonify(dict(_aa_job))


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/demo")
def demo():
    return Response(DEMO_HTML, mimetype="text/html")


# --------------------------------------------------------------------------- #
# Front-end (single page)
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>annotation editor</title>
<script>
  // apply the saved theme before first paint (default: light); remembers last choice
  (function(){ try{ document.documentElement.setAttribute('data-theme',
    localStorage.getItem('theme')||'light'); }catch(e){ document.documentElement.setAttribute('data-theme','light'); } })();
</script>
<style>
  :root{
    --bg:#0a0a0a; --surface:#171717; --surface-2:#222; --surface-3:#1c1c1c;
    --border:#333; --border-2:#444;
    --text:#f5f5f5; --text-muted:#a3a3a3; --text-dim:#6f6f6f;
    --accent:#6ea8fe; --accent-soft:rgba(110,168,254,.14); --accent-fg:#0a0a0a; --accent-hover:#86b8ff;
    --ok:#34d399; --ok-soft:rgba(52,211,153,.14);
    --danger:#f87171; --danger-soft:rgba(248,113,113,.12); --danger-border:rgba(248,113,113,.32);
    --warn:#fbbf24; --canvas-bg:#000;
    --r:8px; --r-lg:12px;
    --sh-sm:0 1px 2px rgba(0,0,0,.5);
    --sh-md:0 6px 18px rgba(0,0,0,.5);
    --sh-lg:0 18px 44px rgba(0,0,0,.6);
    --ring:0 0 0 3px rgba(110,168,254,.22);
  }
  :root[data-theme="light"]{
    --bg:#f7f7f8; --surface:#ffffff; --surface-2:#eef0f2; --surface-3:#f2f3f5;
    --border:#e2e4e8; --border-2:#cfd2d8;
    --text:#171717; --text-muted:#5b6470; --text-dim:#9aa0aa;
    --accent:#2563eb; --accent-soft:rgba(37,99,235,.10); --accent-fg:#ffffff; --accent-hover:#1d4ed8;
    --ok:#059669; --ok-soft:rgba(5,150,105,.12);
    --danger:#dc2626; --danger-soft:rgba(220,38,38,.08); --danger-border:rgba(220,38,38,.30);
    --warn:#b45309; --canvas-bg:#d7d9dd;
    --sh-sm:0 1px 2px rgba(16,24,40,.06);
    --sh-md:0 6px 18px rgba(16,24,40,.10);
    --sh-lg:0 18px 44px rgba(16,24,40,.16);
    --ring:0 0 0 3px rgba(37,99,235,.20);
  }
  .theme-btn{display:inline-flex;align-items:center;justify-content:center;}
  .home-theme{position:absolute;top:18px;right:18px;z-index:5;width:40px;height:40px;
    background:var(--surface);border:1px solid var(--border);color:var(--text-muted);
    border-radius:var(--r);cursor:pointer;transition:background .15s,color .15s,border-color .15s;}
  .home-theme:hover{background:var(--surface-2);color:var(--text);border-color:var(--border-2);}
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
            -webkit-font-smoothing:antialiased;}
  body{display:flex;flex-direction:column;}
  *{scrollbar-width:thin;scrollbar-color:var(--border-2) transparent;}
  *::-webkit-scrollbar{width:8px;height:8px;}
  *::-webkit-scrollbar-track{background:transparent;}
  *::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:8px;border:2px solid transparent;background-clip:padding-box;}
  *::-webkit-scrollbar-thumb:hover{background:var(--text-dim);background-clip:padding-box;}
  svg{display:block;}
  .ic{width:16px;height:16px;flex:none;}
  /* top navigation bar with the image scrubber */
  #topnav{display:flex;align-items:center;gap:9px;padding:9px 14px;background:var(--surface);
          border-bottom:1px solid var(--border);flex:none;}
  #topnav button{display:inline-flex;align-items:center;gap:6px;background:transparent;
          color:var(--text);border:1px solid var(--border);padding:7px 12px;border-radius:var(--r);
          cursor:pointer;font-size:13px;font-weight:500;white-space:nowrap;
          transition:background .15s,border-color .15s,color .15s;}
  #topnav button:hover{background:var(--surface-2);border-color:var(--border-2);}
  #topnav button:disabled{opacity:.35;cursor:default;background:transparent;border-color:var(--border);}
  #topnav input[type=number]{width:74px;padding:7px 9px;background:var(--bg);color:var(--text);
          border:1px solid var(--border);border-radius:var(--r);outline:none;transition:border-color .15s,box-shadow .15s;}
  #topnav input[type=number]:focus{border-color:var(--accent);box-shadow:var(--ring);}
  #scrub{flex:1;min-width:120px;cursor:pointer;accent-color:var(--accent);height:5px;}
  #navpos{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--text-muted);
          white-space:nowrap;min-width:92px;text-align:center;}
  #topnav .brand{display:inline-flex;align-items:center;gap:8px;font-weight:700;font-size:14px;
                 color:var(--text);letter-spacing:-.2px;white-space:nowrap;margin-right:2px;}
  .brand-logo{width:24px;height:24px;border-radius:7px;display:block;}
  .mode{font-size:11px;padding:3px 10px;border-radius:999px;white-space:nowrap;
        font-weight:600;letter-spacing:.2px;}
  .mode.local{background:var(--ok-soft);color:var(--ok);border:1px solid rgba(52,211,153,.3);}
  .mode.cvat{background:var(--accent-soft);color:var(--accent);border:1px solid rgba(110,168,254,.32);}
  #meta{display:flex;gap:16px;align-items:center;padding:7px 14px;
        background:var(--surface-3);border-bottom:1px solid var(--border);flex:none;}
  #status{font-size:12.5px;color:var(--ok);}
  #name{font-size:12.5px;color:var(--text-muted);font-family:ui-monospace,Menlo,monospace;}
  #activeclass{font-size:12.5px;color:var(--text);display:flex;align-items:center;gap:6px;}
  #activeclass .sw{width:13px;height:13px;border-radius:4px;display:inline-block;
                   box-shadow:0 0 0 1px rgba(0,0,0,.4);}
  #main{flex:1;display:flex;min-height:0;}
  #wrap{position:relative;flex:1;min-width:0;overflow:auto;display:flex;background:var(--bg);}
  canvas#cv{background:var(--canvas-bg);box-shadow:0 8px 40px rgba(0,0,0,.6);margin:auto;border-radius:2px;}
  .dirty{color:var(--warn) !important;}
  kbd{background:var(--surface-2);border:1px solid var(--border-2);border-radius:5px;
      padding:1px 6px;font-size:11px;font-family:ui-monospace,Menlo,monospace;}
  /* ---- left control sidebar ---- */
  #left{width:240px;flex:none;overflow-y:auto;background:var(--surface);
        border-right:1px solid var(--border);}
  .lp-sec{padding:13px 14px;border-bottom:1px solid var(--border);}
  .lp-sec h4{margin:0 0 9px;font-size:10.5px;letter-spacing:.7px;color:var(--text-muted);
             text-transform:uppercase;font-weight:600;}
  .tick{display:inline-flex;align-items:center;gap:7px;cursor:pointer;font-size:13px;color:var(--text);}
  .tick input{accent-color:var(--accent);width:15px;height:15px;margin:0;cursor:pointer;}
  .lp-sec input[type=text],.lp-sec input[type=number]{
     width:100%;box-sizing:border-box;padding:8px 10px;background:var(--bg);color:var(--text);
     border:1px solid var(--border);border-radius:var(--r);font-size:13px;margin-bottom:7px;outline:none;
     transition:border-color .15s,box-shadow .15s;}
  .lp-sec input:focus{border-color:var(--accent);box-shadow:var(--ring);}
  .lp-row{display:flex;gap:7px;margin-bottom:7px;}
  .lp-row:last-child{margin-bottom:0;}
  .lp-row input[type=number]{width:auto;flex:1;min-width:0;margin-bottom:0;}
  .lp-row button{flex:none;}
  .lp-row button.grow{flex:1;min-width:0;}
  .lp-sec button{display:inline-flex;align-items:center;justify-content:center;gap:7px;
     background:transparent;color:var(--text);border:1px solid var(--border);padding:8px 12px;
     border-radius:var(--r);cursor:pointer;font-size:13px;font-weight:500;white-space:nowrap;
     transition:background .15s,border-color .15s,color .15s,box-shadow .15s;}
  .lp-sec button:hover{background:var(--surface-2);border-color:var(--border-2);}
  .lp-sec button.wide{display:flex;width:100%;margin-bottom:7px;box-sizing:border-box;}
  .lp-sec button.wide:last-child{margin-bottom:0;}
  .lp-sec button.grow{flex:1;}
  .lp-sec button.danger{color:var(--danger);border-color:var(--danger-border);}
  .lp-sec button.danger:hover{background:var(--danger-soft);border-color:var(--danger);}
  .lp-sec button.ok{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);font-weight:600;}
  .lp-sec button.ok:hover{background:var(--accent-hover);border-color:var(--accent-hover);box-shadow:var(--ring);}
  .lp-sec select{width:100%;box-sizing:border-box;padding:8px 10px;background:var(--bg);
     color:var(--text);border:1px solid var(--border);border-radius:var(--r);font-size:13px;outline:none;
     transition:border-color .15s,box-shadow .15s;}
  .lp-sec select:focus{border-color:var(--accent);box-shadow:var(--ring);}
  /* custom dropdowns — styled menu with hover-coloured options */
  .dropdown{position:relative;display:block;width:100%;}
  .selrow .dropdown,.lp-row .dropdown,.maprow .dropdown,.row .dropdown{flex:1 1 auto;min-width:0;}
  .browse-bar .dropdown{width:260px;flex:none;}
  .row .dropdown-trigger{padding:5px 8px !important;font-size:12px;background:var(--bg);}
  .dropdown-trigger{display:flex !important;align-items:center;justify-content:space-between !important;
    gap:10px;width:100% !important;box-sizing:border-box;
    background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 11px !important;
    border-radius:var(--r);font-size:13px;cursor:pointer;text-align:left;
    transition:border-color .15s,box-shadow .15s;}
  .dropdown-trigger:hover,.dropdown-trigger.open{border-color:var(--accent);box-shadow:var(--ring);}
  .dropdown-trigger.disabled{opacity:.6;cursor:default;box-shadow:none;border-color:var(--border);}
  .dropdown-value{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .dropdown-trigger svg{flex:none;color:var(--text-muted);transition:transform .15s;}
  .dropdown-trigger.open svg{transform:rotate(180deg);}
  .dropdown-menu{position:fixed;background:var(--surface);border:1px solid var(--border-2);
    border-radius:var(--r);box-shadow:var(--sh-lg);padding:5px;z-index:300;overflow:auto;
    animation:ddrop .12s ease-out;}
  @keyframes ddrop{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
  .dropdown-item{display:block;width:100%;background:transparent;border:none;text-align:left;padding:8px 11px;
    border-radius:6px;color:var(--text);font-size:13px;cursor:pointer;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;transition:background .1s,color .1s;}
  .dropdown-item:hover{background:var(--accent-soft);color:var(--accent);}
  .dropdown-item.active{background:var(--surface-2);color:var(--accent);font-weight:600;}
  .dropdown-search{position:sticky;top:-5px;z-index:2;display:block;box-sizing:border-box;
    width:calc(100% + 10px);margin:-5px -5px 5px;padding:9px 11px;
    background:var(--surface);border:none;border-bottom:1px solid var(--border-2);
    color:var(--text);font-size:13px;outline:none;}
  .dropdown-search::placeholder{color:var(--text-muted);}
  .dropdown-search:focus{border-bottom-color:var(--accent);}
  .dropdown-noresult{padding:10px 11px;color:var(--text-muted);font-size:12.5px;text-align:center;}
  /* per-class visibility rows */
  .collapse-h{cursor:pointer;user-select:none;display:flex;align-items:center;gap:7px;}
  .collapse-h #viscaret{color:var(--text-muted);display:inline-flex;width:12px;transition:transform .15s;}
  #visiblelist{display:flex;flex-direction:column;gap:1px;max-height:230px;overflow:auto;}
  .vis-row{display:flex;align-items:center;gap:8px;padding:5px 6px;cursor:pointer;
           font-size:12.5px;border-radius:6px;}
  .vis-row:hover{background:var(--surface-2);}
  .vis-row input{accent-color:var(--accent);width:15px;height:15px;margin:0;cursor:pointer;flex:none;}
  .vis-row .sw{width:13px;height:13px;border-radius:4px;box-shadow:0 0 0 1px rgba(0,0,0,.4);flex:none;}
  .vis-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .lp-toggles{display:flex;flex-direction:column;gap:6px;}
  .toggle{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;
          border-radius:999px;background:var(--surface-2);border:1px solid var(--border);cursor:pointer;
          font-size:12.5px;color:var(--text-muted);user-select:none;white-space:nowrap;
          transition:background .15s,border-color .15s,color .15s;}
  .toggle:hover{border-color:var(--border-2);}
  .opacity-row{margin-top:10px;}
  .opacity-row label{display:flex;justify-content:space-between;align-items:center;
    font-size:12px;color:var(--text-muted);margin-bottom:6px;}
  .opacity-row #fillopval{color:var(--accent);font-weight:600;font-variant-numeric:tabular-nums;}
  .opacity-row input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;height:5px;margin:0;}
  .toggle input{appearance:none;-webkit-appearance:none;width:0;height:0;margin:0;}
  .toggle:has(input:checked){background:var(--ok-soft);border-color:rgba(52,211,153,.4);color:var(--ok);}
  .toggle:has(input:checked)::before{content:'\25CF';color:var(--ok);font-size:9px;}
  .toggle::before{content:'\25CB';color:var(--text-dim);font-size:9px;}
  /* right boxes panel */
  #panel{width:288px;flex:none;overflow:auto;background:var(--surface);
         border-left:1px solid var(--border);font-size:13px;}
  #panel h3{margin:0;padding:11px 14px;background:var(--surface-3);font-size:10.5px;
            color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;font-weight:600;
            border-bottom:1px solid var(--border);}
  .row{display:flex;align-items:center;gap:8px;padding:8px 12px;
       border-bottom:1px solid var(--border);cursor:pointer;transition:background .12s;}
  .row:hover{background:var(--surface-2);}
  .selrow{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent);}
  .row .ix{font-family:ui-monospace,Menlo,monospace;width:28px;flex:none;font-size:12px;}
  .row select{flex:1;min-width:0;padding:5px 7px;background:var(--bg);color:var(--text);
             border:1px solid var(--border);border-radius:6px;font-size:12px;outline:none;}
  .row select:focus{border-color:var(--accent);}
  .row .del{margin-left:auto;background:transparent;color:var(--danger);border:1px solid var(--danger-border);
            border-radius:6px;cursor:pointer;padding:3px 9px;flex:none;display:inline-flex;align-items:center;
            transition:background .12s;}
  .row .del:hover{background:var(--danger-soft);}
  .sec{padding:8px 12px;background:var(--surface-3);color:var(--text-muted);font-size:10.5px;
       text-transform:uppercase;letter-spacing:.6px;font-weight:600;}
  #help{font-size:11px;color:var(--text-muted);padding:7px 14px;background:var(--surface);
        border-top:1px solid var(--border);flex:none;display:flex;flex-wrap:wrap;gap:4px 2px;align-items:center;}
  /* radial class picker overlay (covers viewport, never eats mouse events) */
  #radial{position:fixed;inset:0;z-index:50;display:none;pointer-events:none;}
  #cvatstatus{font-size:11.5px;color:var(--text-muted);margin-top:8px;word-break:break-word;line-height:1.45;}
  .lp-row #cvatproj{flex:1;width:auto;min-width:0;}
  #cvatproj:disabled{opacity:.7;}
  /* minimal icon-only lock: no button chrome, colour reflects state */
  .lp-sec .lockbtn{flex:none;width:38px;padding:0;background:transparent;border:1px solid var(--border);
                 border-radius:var(--r);color:var(--text-dim);cursor:pointer;display:flex;
                 align-items:center;justify-content:center;transition:color .15s,border-color .15s,background .15s;}
  .lp-sec .lockbtn:hover{background:var(--surface-2);color:var(--text-muted);}
  .lp-sec .lockbtn.on{color:var(--ok);border-color:rgba(52,211,153,.4);background:var(--ok-soft);}
  /* modals + progress */
  .modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.66);backdrop-filter:blur(2px);z-index:100;
            display:none;align-items:center;justify-content:center;}
  /* the auto-annotation pipeline is a standalone full-screen page (header bar +
     centred form card), consistent with the Import / Class-count screens */
  #apmodal{z-index:195;}
  .ap-page{flex:1;overflow:auto;display:flex;justify-content:center;align-items:flex-start;padding:30px 20px;}
  /* minimized auto-annotation: floating progress pill (above everything) */
  .apw{position:fixed;right:18px;bottom:18px;width:300px;z-index:210;background:var(--surface);
    border:1px solid var(--border-2);border-radius:var(--r-lg);box-shadow:var(--sh-lg);
    padding:12px 13px;animation:ddrop .14s ease-out;}
  .apw-top{display:flex;align-items:center;gap:8px;margin-bottom:9px;}
  .apw-top .spacer{flex:1;}
  .apw-dot{width:8px;height:8px;border-radius:50%;background:var(--warn);flex:none;
    box-shadow:0 0 0 3px rgba(251,191,36,.2);animation:apwpulse 1.2s ease-in-out infinite;}
  @keyframes apwpulse{0%,100%{opacity:1;}50%{opacity:.35;}}
  .apw-title{font-size:12.5px;font-weight:600;color:var(--text);}
  .apw-icon{flex:none;width:26px;height:26px;display:flex;align-items:center;justify-content:center;
    background:transparent;border:1px solid var(--border);border-radius:7px;color:var(--text-muted);
    cursor:pointer;transition:background .12s,color .12s;}
  .apw-icon:hover{background:var(--surface-2);color:var(--text);}
  .apw-bar{height:6px;background:var(--bg);border-radius:99px;overflow:hidden;}
  .apw-bar>div{height:100%;width:0;background:var(--accent);border-radius:99px;transition:width .3s;}
  .apw-text{font-size:11.5px;color:var(--text-muted);margin-top:7px;line-height:1.35;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .apw.done .apw-dot{background:var(--ok);box-shadow:0 0 0 3px rgba(52,211,153,.2);animation:none;}
  .apw.done .apw-bar>div{background:var(--ok);}
  .modal{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);width:360px;
         max-width:92vw;box-shadow:var(--sh-lg);}
  .modal-h{padding:15px 18px;font-size:15px;font-weight:600;border-bottom:1px solid var(--border);}
  .modal-body{padding:16px 18px;display:flex;flex-direction:column;gap:7px;}
  .modal-body label{font-size:10.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;
                    font-weight:600;margin-top:2px;}
  #aacfg{display:flex;flex-direction:column;gap:7px;}
  .modal select,.modal input[type=number],.modal input[type=text]{padding:9px 11px;background:var(--bg);
         color:var(--text);border:1px solid var(--border);border-radius:var(--r);font-size:13px;width:100%;
         box-sizing:border-box;outline:none;transition:border-color .15s,box-shadow .15s;}
  .modal select:focus,.modal input:focus{border-color:var(--accent);box-shadow:var(--ring);}
  .modal .selrow{display:flex;gap:7px;align-items:center;}
  .modal .selrow select{flex:1;min-width:0;}
  /* direct child only, so it doesn't hit the custom-dropdown trigger button */
  .modal .selrow > button{flex:none;width:40px;padding:9px 0;background:transparent;color:var(--text-muted);
         border:1px solid var(--border);border-radius:var(--r);cursor:pointer;display:flex;
         align-items:center;justify-content:center;transition:background .15s,color .15s;}
  .modal .selrow > button:hover{background:var(--surface-2);color:var(--text);}
  .modal-f{padding:14px 18px;border-top:1px solid var(--border);display:flex;
           justify-content:flex-end;gap:9px;}
  .modal-f button#apmin{display:inline-flex;align-items:center;gap:6px;}
  .modal-f button{padding:9px 18px;border-radius:var(--r);border:1px solid var(--border);
                  background:transparent;color:var(--text);cursor:pointer;font-size:13px;font-weight:500;
                  transition:background .15s,border-color .15s;}
  .modal-f button:hover{background:var(--surface-2);border-color:var(--border-2);}
  .modal-f button.ok{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);font-weight:600;}
  .modal-f button.ok:hover{background:var(--accent-hover);}
  .modal-f button.danger{background:var(--danger);color:#fff;border-color:var(--danger);font-weight:600;}
  .modal-f button.danger:hover{filter:brightness(1.06);}
  .modal-f button:disabled{opacity:.45;cursor:default;}
  .bar{height:8px;background:var(--surface-2);border-radius:999px;overflow:hidden;margin:8px 0 12px;}
  .bar>div{height:100%;width:0;background:var(--accent);border-radius:999px;transition:width .3s ease;}
  .bar.indet>div{width:35%;transition:none;animation:indet 1.1s ease-in-out infinite;}
  @keyframes indet{0%{margin-left:-35%}100%{margin-left:100%}}
  #aaprogtext,#cvprogtext,#impprogtext{font-size:12.5px;color:var(--text);line-height:1.5;}
  .aamsg{font-size:11.5px;color:var(--warn);min-height:14px;}
  .maphint{font-size:11.5px;color:var(--text-muted);margin-bottom:8px;line-height:1.45;}
  #aamaplist{display:flex;flex-direction:column;gap:6px;max-height:320px;overflow:auto;}
  .maprow{display:flex;align-items:center;gap:8px;}
  .maprow .mc{flex:0 0 42%;font-size:12px;color:var(--text);overflow:hidden;
              text-overflow:ellipsis;white-space:nowrap;}
  .maprow .arr{color:var(--text-dim);}
  .maprow select{flex:1;min-width:0;padding:7px 9px;background:var(--bg);color:var(--text);
                 border:1px solid var(--border);border-radius:var(--r);font-size:12px;}
  /* multi-select task list (auto-annotation pipeline) */
  .aptasks{max-height:180px;overflow:auto;border:1px solid var(--border);border-radius:var(--r);
           background:var(--bg);padding:5px;}
  .aptasks .apt-empty{color:var(--text-dim);padding:7px;font-size:12px;}
  .aptasks .trow{display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:6px;
                 cursor:pointer;font-size:12.5px;}
  .aptasks .trow:hover{background:var(--surface-2);}
  .aptasks .trow input{accent-color:var(--accent);width:15px;height:15px;margin:0;flex:none;}
  .aptasks .tid{font-family:ui-monospace,Menlo,monospace;color:var(--ok);flex:none;}
  .aptasks .tname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .aptaskcount{font-size:11px;color:var(--accent);text-transform:none;letter-spacing:0;}
  .aa-mgroup{margin-bottom:11px;}
  .aa-mgroup:last-child{margin-bottom:0;}
  .aa-mname{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
    color:var(--accent);margin:2px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--border);}
  .aprow{display:flex;gap:7px;align-items:center;margin:7px 0;}
  .aprow .spacer{flex:1;}
  .aprow button{flex:none;padding:7px 13px;background:transparent;color:var(--text);
                border:1px solid var(--border);border-radius:var(--r);cursor:pointer;font-size:12px;
                display:inline-flex;align-items:center;gap:6px;transition:background .15s,border-color .15s;}
  .aprow button:hover{background:var(--surface-2);border-color:var(--border-2);}
  .cvtarget{font-size:13px;color:var(--text);background:var(--bg);border:1px solid var(--border);
            border-radius:var(--r);padding:9px 11px;word-break:break-word;}
  /* landing / home screen */
  #home{position:fixed;inset:0;z-index:200;display:flex;align-items:center;justify-content:center;
        background:radial-gradient(circle at 18% 0%,rgba(110,168,254,.10),transparent 42%),
                   radial-gradient(circle at 85% 100%,rgba(52,211,153,.08),transparent 45%),var(--bg);}
  .home-inner{text-align:center;max-width:610px;padding:24px;}
  .home-logo{width:60px;height:60px;border-radius:16px;margin:0 auto 16px;display:block;
             filter:drop-shadow(0 10px 28px rgba(110,168,254,.25));}
  .home-title{font-size:32px;margin:0 0 6px;color:var(--text);font-weight:700;letter-spacing:-.5px;}
  .home-sub{color:var(--text-muted);margin:0 0 32px;font-size:14.5px;}
  .continue-card{display:flex;align-items:center;gap:15px;width:100%;box-sizing:border-box;
    background:var(--surface);border:1px solid var(--accent);border-radius:14px;padding:15px 18px;
    margin-bottom:18px;cursor:pointer;text-align:left;box-shadow:var(--sh-sm);font:inherit;color:inherit;
    transition:transform .12s,border-color .12s,box-shadow .12s,background .12s;}
  .continue-card:hover{transform:translateY(-2px);box-shadow:var(--sh-md);background:var(--surface-2);}
  .continue-card .cont-icon{flex:none;width:44px;height:44px;border-radius:11px;display:flex;
    align-items:center;justify-content:center;background:var(--accent-soft);color:var(--accent);}
  .continue-card .cont-body{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px;}
  .cont-eyebrow{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--accent);}
  .cont-name-row{display:flex;align-items:center;gap:9px;min-width:0;}
  .cont-name{font-size:15.5px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .cont-badge{flex:none;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
    padding:2px 8px;border-radius:999px;line-height:1.5;}
  .cont-badge.local{background:var(--accent-soft);color:var(--accent);}
  .cont-badge.cvat{background:var(--ok-soft);color:var(--ok);}
  .cont-meta{font-size:12.5px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .continue-card .cont-arrow{flex:none;color:var(--text-muted);transition:color .12s,transform .12s;}
  .continue-card:hover .cont-arrow{color:var(--accent);transform:translateX(3px);}
  .home-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;}
  .home-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
             padding:22px;cursor:pointer;text-align:left;box-shadow:var(--sh-sm);
             display:flex;flex-direction:column;
             transition:transform .14s,border-color .14s,box-shadow .14s;}
  .home-card:hover{transform:translateY(-4px);border-color:var(--border-2);box-shadow:var(--sh-md);}
  .hc-icon{width:46px;height:46px;border-radius:13px;display:flex;align-items:center;justify-content:center;
           margin-bottom:15px;background:var(--accent-soft);color:var(--accent);}
  .home-card.cvat .hc-icon{background:var(--ok-soft);color:var(--ok);}
  .home-card.auto .hc-icon{background:rgba(251,191,36,.14);color:var(--warn);}
  .home-card.count .hc-icon{background:rgba(167,139,250,.15);color:#a78bfa;}
  /* class-count table */
  #ccbody{flex:1;overflow:auto;padding:18px 20px;}
  .cc-summary{font-size:13px;color:var(--text-muted);margin-bottom:14px;}
  .cc-summary b{color:var(--text);}
  .cc-table{border-collapse:separate;border-spacing:0;font-size:12.5px;width:max-content;min-width:100%;}
  .cc-table th,.cc-table td{padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap;}
  .cc-table thead th{position:sticky;top:0;background:var(--surface-3);color:var(--text-muted);
                     text-transform:uppercase;font-size:10.5px;letter-spacing:.5px;font-weight:600;
                     text-align:right;z-index:2;}
  .cc-table thead th .tkid{display:block;font-size:9.5px;color:var(--text-dim);font-weight:500;text-transform:none;}
  .cc-table th.cc-class,.cc-table td.cc-class{position:sticky;left:0;background:var(--surface);
                     text-align:left;z-index:1;font-weight:500;color:var(--text);border-right:1px solid var(--border);}
  .cc-table thead th.cc-class{z-index:3;background:var(--surface-3);}
  .cc-table td{text-align:right;color:var(--text-muted);font-family:ui-monospace,Menlo,monospace;}
  .cc-table td.cc-total{color:var(--accent);font-weight:600;}
  .cc-table tbody tr:hover td{background:var(--surface-2);}
  .cc-table tbody tr:hover td.cc-class{background:var(--surface-2);}
  .cc-table .cc-totalrow td{border-top:2px solid var(--border-2);color:var(--text);font-weight:700;background:var(--surface-3);}
  .cc-table .cc-totalrow td.cc-class{background:var(--surface-3);}
  .cc-table td.zero{color:var(--text-dim);}
  .home-card:hover .hc-icon{transform:scale(1.05);transition:transform .14s;}
  .hc-title{font-size:17px;font-weight:600;color:var(--text);margin-bottom:8px;letter-spacing:-.2px;}
  .hc-desc{font-size:13px;color:var(--text-muted);line-height:1.55;}
  #homeBtn{padding:7px 9px;}
  /* CVAT browser (project/task cards) */
  .browse{position:fixed;inset:0;z-index:190;background:var(--bg);display:none;flex-direction:column;}
  .browse-bar{display:flex;align-items:center;gap:10px;padding:12px 18px;
              background:var(--surface);border-bottom:1px solid var(--border);flex:none;}
  .browse-bar .spacer{flex:1;}
  .browse-bar button{display:inline-flex;align-items:center;gap:7px;background:transparent;color:var(--text);
              border:1px solid var(--border);padding:8px 14px;border-radius:var(--r);cursor:pointer;font-size:13px;
              font-weight:500;transition:background .15s,border-color .15s;}
  .browse-bar button:hover{background:var(--surface-2);border-color:var(--border-2);}
  .browse-title{font-size:15px;font-weight:600;color:var(--text);letter-spacing:-.2px;}
  .browse-grid{flex:1;overflow:auto;display:grid;align-content:start;gap:14px;padding:20px;
               grid-template-columns:repeat(auto-fill,minmax(240px,1fr));}
  .browse-empty{color:var(--text-dim);padding:20px;font-size:14px;}
  .bcard{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:18px;
         cursor:pointer;box-shadow:var(--sh-sm);display:flex;flex-direction:column;gap:6px;
         transition:transform .12s,border-color .12s,box-shadow .12s;}
  .bc-badge{position:absolute;top:12px;right:12px;display:inline-flex;align-items:center;gap:4px;
    font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
    padding:3px 8px 3px 6px;border-radius:999px;line-height:1;}
  .bc-badge svg{flex:none;}
  .bc-badge.ok{background:rgba(52,211,153,.14);color:var(--ok);border:1px solid rgba(52,211,153,.35);}
  .bc-badge.warn{background:rgba(251,191,36,.15);color:var(--warn);border:1px solid rgba(251,191,36,.38);}
  .bc-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--warn);
    margin-left:5px;box-shadow:0 0 0 2px rgba(251,191,36,.25);}
  .upd-banner{display:flex;align-items:center;gap:9px;margin-bottom:10px;padding:9px 11px;
    background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);border-radius:var(--r);}
  .upd-banner .upd-ic{flex:none;width:16px;height:16px;color:var(--warn);}
  .upd-banner .upd-text{flex:1;min-width:0;font-size:12px;color:var(--text);line-height:1.35;}
  .upd-banner .upd-btn{flex:none;background:var(--warn);color:#1a1300;border:none;border-radius:6px;
    padding:6px 12px;font-size:12px;font-weight:700;cursor:pointer;transition:filter .12s;}
  .upd-banner .upd-btn:hover{filter:brightness(1.08);}
  .bcard:hover{transform:translateY(-3px);border-color:var(--accent);box-shadow:var(--sh-md);}
  .bcard .bc-id{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--ok);}
  .bcard .bc-name{font-size:15px;font-weight:600;color:var(--text);word-break:break-word;letter-spacing:-.2px;flex:1;}
  .bcard .bc-sub{font-size:12px;color:var(--text-muted);display:inline-flex;align-items:center;gap:5px;margin-top:2px;}
  .browse-prog{position:absolute;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(2px);display:flex;
               align-items:center;justify-content:center;}
  .bp-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:26px;
           width:400px;max-width:90vw;text-align:center;box-shadow:var(--sh-lg);}
  #browseProgText{font-size:13px;color:var(--text);margin:12px 0 16px;}
  .bp-close{background:transparent;color:var(--text);border:1px solid var(--border);padding:8px 18px;
            border-radius:var(--r);cursor:pointer;font-size:13px;transition:background .15s;}
  .bp-close:hover{background:var(--surface-2);}
</style>
</head>
<body>
<div id="home">
  <button class="theme-btn home-theme" onclick="toggleTheme()" title="toggle theme"></button>
  <div class="home-inner">
    <svg class="home-logo" viewBox="0 0 64 64"><defs><linearGradient id="alg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs><rect width="64" height="64" rx="15" fill="url(#alg2)"/><g stroke="#0a0a0a" stroke-width="3.4" stroke-linecap="round" fill="none" opacity=".88"><path d="M16 25v-7a2 2 0 0 1 2-2h7"/><path d="M48 25v-7a2 2 0 0 0-2-2h-7"/><path d="M16 39v7a2 2 0 0 0 2 2h7"/><path d="M48 39v7a2 2 0 0 1-2 2h-7"/></g><circle cx="32" cy="32" r="4.6" fill="#0a0a0a"/></svg>
    <h1 class="home-title">Annotation Studio</h1>
    <p class="home-sub">Choose how to start</p>
    <button id="continueCard" class="continue-card" style="display:none" onclick="continueLastSession()">
      <span class="cont-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="22" height="22" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M10 9l5 3-5 3z" fill="currentColor" stroke="none"/></svg></span>
      <span class="cont-body">
        <span class="cont-eyebrow">Continue last session</span>
        <span class="cont-name-row"><span class="cont-name" id="contName"></span><span class="cont-badge" id="contBadge"></span></span>
        <span class="cont-meta" id="contMeta"></span>
      </span>
      <svg class="cont-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="20" height="20" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>
    </button>
    <div class="home-cards">
      <div class="home-card local" onclick="enterLocal()">
        <div class="hc-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg></div>
        <div class="hc-title">Annotate locally</div>
        <div class="hc-desc">Open a folder of images + YOLO labels on this machine and start annotating.</div>
      </div>
      <div class="home-card cvat" onclick="enterImport()">
        <div class="hc-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24" stroke-linecap="round" stroke-linejoin="round"><path d="M7 18a4 4 0 0 1-.5-7.97A6 6 0 0 1 18 9a3.5 3.5 0 0 1 0 9z"/><path d="M12 11v6m0 0l-2.4-2.4M12 17l2.4-2.4"/></svg></div>
        <div class="hc-title">Import from CVAT</div>
        <div class="hc-desc">Pull a task's images + annotations from CVAT and edit them here.</div>
      </div>
      <div class="home-card auto" onclick="enterAuto()">
        <div class="hc-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 4.1 4.1.6-3 2.9.7 4.4L12 17l-3.7 2 .7-4.4-3-2.9 4.1-.6z"/><path d="M5 21h14"/></svg></div>
        <div class="hc-title">Automatic annotations</div>
        <div class="hc-desc">Pick a CVAT project + task, run a model on it, and push the annotations back to CVAT.</div>
      </div>
      <div class="home-card count" onclick="enterCount()">
        <div class="hc-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="11" width="3" height="6" rx="1"/><rect x="12.5" y="7" width="3" height="10" rx="1"/><rect x="18" y="13" width="3" height="4" rx="1"/></svg></div>
        <div class="hc-title">Class count</div>
        <div class="hc-desc">Count annotations per class in a CVAT project — project-wide and broken down by task.</div>
      </div>
    </div>
  </div>
</div>
<div id="topnav">
  <button id="homeBtn" onclick="goHome()" title="home"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/></svg></button>
  <button id="tasksBtn" onclick="returnToTasks()" title="back to tasks" style="display:none;"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></button>
  <span class="brand"><svg class="brand-logo" viewBox="0 0 64 64"><defs><linearGradient id="alg1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs><rect width="64" height="64" rx="15" fill="url(#alg1)"/><g stroke="#0a0a0a" stroke-width="3.4" stroke-linecap="round" fill="none" opacity=".88"><path d="M16 25v-7a2 2 0 0 1 2-2h7"/><path d="M48 25v-7a2 2 0 0 0-2-2h-7"/><path d="M16 39v7a2 2 0 0 0 2 2h7"/><path d="M48 39v7a2 2 0 0 1-2 2h-7"/></g><circle cx="32" cy="32" r="4.6" fill="#0a0a0a"/></svg>Annotation&nbsp;Studio</span>
  <span id="modebadge" class="mode local">Local</span>
  <button onclick="go(-1)" title="prev (A)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>Prev</button>
  <button onclick="go(1)" title="next (D)">Next<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></button>
  <button id="undoBtn" onclick="undo()" title="undo (Ctrl+Z)" disabled><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/></svg></button>
  <button id="redoBtn" onclick="redo()" title="redo (Ctrl+Y)" disabled><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 14 5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/></svg></button>
  <input id="scrub" type="range" min="0" max="0" value="0"
         oninput="scrubTo(this.value)" onchange="scrubTo(this.value)"
         title="drag to scrub through all images">
  <span id="navpos">0 / 0</span>
  <input id="jump" type="number" min="1" placeholder="#"
         onkeydown="if(event.key==='Enter')jump()">
  <button onclick="jump()">Go</button>
  <button class="theme-btn" onclick="toggleTheme()" title="toggle theme"></button>
</div>
<div id="meta">
  <span id="name"></span>
  <span id="activeclass"></span>
  <span id="status"></span>
</div>
<div id="main">
  <div id="left">
    <div class="lp-sec">
      <h4>Dataset folder</h4>
      <input id="folder" type="text" placeholder="folder with images/ + labels/ + labels.txt"
             onkeydown="if(event.key==='Enter')loadFolder()">
      <button class="wide" onclick="loadFolder()">Load folder</button>
    </div>
    <div class="lp-sec">
      <h4>Active class &mdash; hold <kbd>C</kbd> for wheel</h4>
      <select id="classsel" onchange="setActiveClass(parseInt(this.value,10))"></select>
      <button class="wide" style="margin-top:7px;" onclick="openClsModal()">Import classes from CVAT</button>
    </div>
    <div class="lp-sec">
      <h4 class="collapse-h" onclick="toggleVisSec()"><span id="viscaret"><svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></span> Visible labels</h4>
      <div id="visbody" style="display:none;">
        <div class="lp-row" style="margin-bottom:6px;">
          <button class="grow" onclick="setAllVisible(true)">All</button>
          <button class="grow" onclick="setAllVisible(false)">None</button>
        </div>
        <div id="visiblelist"></div>
      </div>
    </div>
    <div class="lp-sec">
      <h4>Automatic annotation</h4>
      <button class="wide ok" onclick="openAaModal()">Automatic annotation…</button>
    </div>
    <div class="lp-sec" id="cvatsec">
      <h4>CVAT</h4>
      <div id="cvupdatebanner" class="upd-banner" style="display:none">
        <svg class="upd-ic" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/></svg>
        <span class="upd-text" id="cvupdtext">A newer version of this task is available on CVAT.</span>
        <button class="upd-btn" id="cvupdbtn" onclick="updateTaskNow()">Update</button>
      </div>
      <div id="cvatProjWrap">
        <div class="lp-row">
          <select id="cvatproj" onchange="onCvatProjPick()"><option value="">— loading projects… —</option></select>
          <button id="cvatlock" class="lockbtn" title="lock project" onclick="toggleCvatLock()"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-1.5"/></svg></button>
        </div>
        <button class="wide" onclick="loadCvatProjects(true)">Refresh list</button>
        <button class="wide" onclick="openCvatBrowse()">Import from CVAT…</button>
        <div id="cvatstatus"></div>
      </div>
      <button class="wide ok" onclick="openCvModal()">Upload to CVAT…</button>
    </div>
  </div>

  <div id="wrap"><canvas id="cv"></canvas></div>

  <div id="panel">
    <div class="lp-sec">
      <h4>Tools</h4>
      <div class="lp-toggles">
        <label class="toggle"><input type="checkbox" id="autosave" checked> &#10515; autosave</label>
      </div>
      <div class="opacity-row">
        <label for="fillop">Box fill opacity <span id="fillopval">20%</span></label>
        <input type="range" id="fillop" min="0" max="60" step="5" value="20" oninput="setFillOpacity(this.value)">
      </div>
    </div>
    <div class="lp-sec">
      <h4>Actions</h4>
      <div class="lp-row">
        <button class="grow danger" onclick="delSel()">Delete box</button>
        <button class="grow danger" onclick="clearAll()">Clear all</button>
      </div>
      <button class="wide ok" onclick="save()">Save (S)</button>
      <button class="wide danger" onclick="deleteImage()">Delete this image</button>
    </div>
    <h3>Boxes &mdash; class &amp; delete</h3>
    <div id="list"></div>
  </div>
</div>
<div id="help">
  drag empty = add &nbsp; drag inside = move &nbsp; drag handle = resize &nbsp;|&nbsp;
  <b>scroll = zoom</b> &nbsp;|&nbsp; <kbd>A</kbd>/<kbd>D</kbd> prev/next &nbsp;
  <kbd>0-9</kbd> class &nbsp; <b>hold <kbd>C</kbd> = class wheel</b> &nbsp;
  <kbd>Del</kbd> delete &nbsp; <kbd>S</kbd> save &nbsp;|&nbsp;
  <b>right-click a box = delete it</b>
</div>
<canvas id="radial"></canvas>

<div id="aamodal" class="modal-bg">
  <div class="modal">
    <div class="modal-h">Automatic annotation</div>
    <div class="modal-body">
      <div id="aacfg">
        <label>Models <span class="aptaskcount" id="aamodelcount"></span></label>
        <div id="aamodellist" class="aptasks"><div class="apt-empty">loading models…</div></div>
        <div class="aprow">
          <button type="button" onclick="aaSelectAllModels(true)">All</button>
          <button type="button" onclick="aaSelectAllModels(false)">None</button>
        </div>
        <label>Confidence</label>
        <input id="aaconf" type="number" min="0" max="1" step="0.05" value="0.25">
        <label>If a label already exists</label>
        <select id="aamode">
          <option value="append">add detections to it</option>
          <option value="replace">replace it</option>
          <option value="skip">skip the image</option>
        </select>
      </div>
      <div id="aamap" style="display:none;">
        <div class="maphint">Map each <b>model class</b> → a class here (auto-matched by name; pick <i>skip</i> to drop one):</div>
        <div id="aamaplist"></div>
      </div>
      <div id="aaprog" style="display:none;">
        <div class="bar"><div id="aabar"></div></div>
        <div id="aaprogtext"></div>
      </div>
      <div id="aamsg" class="aamsg"></div>
    </div>
    <div class="modal-f">
      <button id="aacancel" onclick="closeAaModal()">Cancel</button>
      <button id="aaback" onclick="aaShowConfig()" style="display:none;">Back</button>
      <button id="aanext" class="ok" onclick="aaNext()">Next</button>
      <button id="aarun" class="ok" onclick="runAutoAnnotate()" style="display:none;">Annotate</button>
    </div>
  </div>
</div>

<div id="clsmodal" class="modal-bg">
  <div class="modal">
    <div class="modal-h">Import classes from CVAT</div>
    <div class="modal-body">
      <label>Project</label>
      <div class="selrow">
        <select id="clsproj"><option value="">— select project —</option></select>
        <button onclick="loadClsProjects(true)" title="refresh projects"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg></button>
      </div>
      <div class="maphint">The project's labels become the annotation classes for this folder.</div>
      <div id="clsmsg" class="aamsg"></div>
    </div>
    <div class="modal-f">
      <button onclick="closeClsModal()">Cancel</button>
      <button class="ok" id="clsrun" onclick="runImportClasses()">Import classes</button>
    </div>
  </div>
</div>

<div id="cvmodal" class="modal-bg">
  <div class="modal">
    <div class="modal-h">Upload to CVAT</div>
    <div class="modal-body">
      <div id="cvcfg">
        <div id="cvlinkwrap" style="display:none;">
          <label class="tick" style="margin-bottom:8px;"><input type="checkbox" id="cvupdate" checked onchange="cvUpdateToggle()"> Update source task <b id="cvlinkedname"></b></label>
        </div>
        <div id="cvnewwrap">
          <label>Project</label>
          <div class="selrow">
            <select id="cvUploadProj" onchange="updateCvRunState()"><option value="">— select project —</option></select>
            <button onclick="loadCvUploadProjects(true)" title="refresh projects"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg></button>
          </div>
          <label>Task name</label>
          <input id="cvattask" type="text" placeholder="task name">
        </div>
      </div>
      <div id="cvprog" style="display:none;">
        <div class="bar indet" id="cvbarwrap"><div id="cvbar"></div></div>
        <div id="cvprogtext"></div>
      </div>
      <div id="cvmsg" class="aamsg"></div>
    </div>
    <div class="modal-f">
      <button id="cvcancel" onclick="closeCvModal()">Cancel</button>
      <button id="cvrun" class="ok" onclick="runCvatUpload()">Upload</button>
    </div>
  </div>
</div>

<div id="cvatbrowse" class="browse">
  <div class="browse-bar">
    <button onclick="goHome()" title="home"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/></svg></button>
    <button id="browseBack" onclick="browseProjects()" style="display:none;"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>Projects</button>
    <span id="browseTitle" class="browse-title">Import from CVAT — projects</span>
    <span class="spacer"></span>
    <button id="browseUpdateAll" onclick="updateAllTasks()" title="update all imported tasks in this project" style="display:none;"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/></svg>Update all</button>
    <button onclick="browseRefresh()" title="refresh"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>Refresh</button>
  </div>
  <div id="browseGrid" class="browse-grid"></div>
  <div id="browseProg" class="browse-prog" style="display:none;">
    <div class="bp-card">
      <div class="bar indet"><div></div></div>
      <div id="browseProgText"></div>
      <button class="bp-close" onclick="browseCancelProg()">Close</button>
    </div>
  </div>
</div>

<div id="ccview" class="browse">
  <div class="browse-bar">
    <button onclick="goHome()" title="home"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/></svg></button>
    <span class="browse-title">Class count</span>
    <span class="spacer"></span>
    <select id="ccproj" style="width:auto;min-width:220px;max-width:340px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:13px;"><option value="">— select project —</option></select>
    <button onclick="ccLoadProjects(true)" title="refresh projects"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg></button>
    <button class="ok" id="ccrun" onclick="ccRun()" style="background:var(--accent);color:var(--accent-fg);border-color:var(--accent);font-weight:600;">Count</button>
  </div>
  <div id="ccbody">
    <div id="ccempty" class="browse-empty">Select a CVAT project and click <b>Count</b>.</div>
    <div id="ccprog" class="bp-card" style="display:none;margin:40px auto;">
      <div class="bar indet"><div></div></div>
      <div id="ccprogtext" style="font-size:13px;color:var(--text);"></div>
    </div>
    <div id="ccresult"></div>
  </div>
</div>

<div id="apmodal" class="browse">
  <div class="browse-bar">
    <button onclick="closeApModal()" title="home"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/></svg></button>
    <span class="browse-title">Automatic annotations &rarr; CVAT</span>
    <span class="spacer"></span>
    <button onclick="loadApProjects(true)" title="refresh projects"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>Refresh</button>
  </div>
  <div class="ap-page">
    <div class="modal">
    <div class="modal-body">
      <div id="apcfg">
        <label>Project</label>
        <div class="selrow">
          <select id="approj" onchange="loadApTasks()"><option value="">— select project —</option></select>
          <button onclick="loadApProjects(true)" title="refresh projects"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg></button>
        </div>
        <label>Tasks <span id="aptaskcount" class="aptaskcount"></span></label>
        <div id="aptasklist" class="aptasks"><div class="apt-empty">— select a project first —</div></div>
        <div class="aprow">
          <button onclick="apSelectAllTasks(true)">All</button>
          <button onclick="apSelectAllTasks(false)">None</button>
          <span class="spacer"></span>
          <button onclick="loadApTasks(true)" title="refresh tasks"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>refresh</button>
        </div>
        <label>Model</label>
        <select id="apmodel"><option value="">— loading models… —</option></select>
        <label>Confidence</label>
        <input id="apconf" type="number" min="0" max="1" step="0.05" value="0.25">
        <label>Existing annotations</label>
        <select id="apmode">
          <option value="append">add detections to them</option>
          <option value="replace">replace them</option>
          <option value="skip">skip already-labelled</option>
        </select>
      </div>
      <div id="apmap" style="display:none;">
        <div class="maphint">Map each <b>model class</b> &rarr; a project class (auto-matched by name):</div>
        <div id="apmaplist"></div>
      </div>
      <div id="approg" style="display:none;">
        <div class="bar" id="apbarwrap"><div id="apbar"></div></div>
        <div id="approgtext"></div>
      </div>
      <div id="apmsg" class="aamsg"></div>
    </div>
    <div class="modal-f">
      <button id="apcancel" onclick="closeApModal()">Cancel</button>
      <button id="apmin" onclick="minimizeAuto()" style="display:none;"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/></svg>Minimize</button>
      <button id="apback" onclick="apShowConfig()" style="display:none;">Back</button>
      <button id="apnext" class="ok" onclick="apNext()">Next</button>
      <button id="aprun" class="ok" onclick="runAutoPipeline()" style="display:none;">Run</button>
    </div>
    </div>
  </div>
</div>

<div id="confirmModal" class="modal-bg" style="z-index:400;">
  <div class="modal" style="width:390px;">
    <div class="modal-h" id="confirmTitle">Confirm</div>
    <div class="modal-body">
      <div id="confirmMsg" style="font-size:13.5px;color:var(--text);line-height:1.55;white-space:pre-line;"></div>
    </div>
    <div class="modal-f">
      <button id="confirmCancel" onclick="_confirmResolve(false)">Cancel</button>
      <button id="confirmOk" class="ok" onclick="_confirmResolve(true)">OK</button>
    </div>
  </div>
</div>

<div id="apwidget" class="apw" style="display:none;">
  <div class="apw-top">
    <span class="apw-dot"></span>
    <span class="apw-title">Auto-annotating…</span>
    <span class="spacer"></span>
    <button class="apw-icon" onclick="restoreAuto()" title="open full view"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg></button>
    <button class="apw-icon" onclick="dismissAutoWidget()" title="hide"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
  </div>
  <div class="apw-bar"><div id="apwbar"></div></div>
  <div id="apwtext" class="apw-text"></div>
</div>

<script>
let idx = 0, count = 0, name = "";
let classes = [];        // class names from labels.txt; index = class id
let activeClass = 0;     // class assigned to NEW boxes
let hiddenClasses = new Set();   // class ids hidden from the canvas (view filter)
let boxes = [];          // editable boxes for THIS image (seeded from the file)
let origBoxes = [];      // on-disk snapshot at load time (reference only)
let img = new Image();
let dirty = false;
let touched = false;
let sel = -1;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const HANDLE = 8;
let drag = null;

// distinct, evenly-spread palette; class id maps to a stable colour
// CVAT's own label-colour palette (used when a class has no explicit CVAT colour)
const CVAT_PALETTE=['#33ddff','#fa3253','#ff007c','#ff6037','#f3787e','#b83df5',
  '#66ff66','#aaf0d1','#fafa37','#5986b3','#ff6a4d','#f078f0','#cc3366','#cc9933',
  '#fa32b7','#ff355e','#8271d4','#20f1f1','#e1f93f','#34d1b7','#3399ff','#b25050'];
let classColors=[];    // exact per-class CVAT colours when known, else the palette
function classColor(i){
  if(i>=0 && classColors[i]) return classColors[i];   // exact colour from CVAT
  if(i<0) return '#888';
  return CVAT_PALETTE[i % CVAT_PALETTE.length];
}
function hexToRgba(hex,a){
  hex=String(hex).replace('#','');
  if(hex.length===3) hex=hex.split('').map(x=>x+x).join('');
  const n=parseInt(hex,16)||0;
  return 'rgba('+((n>>16)&255)+','+((n>>8)&255)+','+(n&255)+','+a+')';
}
// same colour as the box outline, low alpha -> a translucent fill to tell boxes apart
function classFill(i,a){ return hexToRgba(classColor(i), a); }
let fillOpacity=20;    // box-fill opacity in %, adjustable from the Tools slider (0 = off)
function setFillOpacity(v){
  fillOpacity=parseInt(v,10)||0;
  const e=document.getElementById('fillopval'); if(e) e.textContent=fillOpacity+'%';
  draw();              // apply immediately
}
// ---- light / dark theme ----
const _SUN='<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const _MOON='<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function currentTheme(){ return document.documentElement.getAttribute('data-theme')||'light'; }
function updateThemeIcons(){
  const dark=currentTheme()==='dark';
  document.querySelectorAll('.theme-btn').forEach(b=>{
    b.innerHTML = dark?_SUN:_MOON;      // moon in light (-> go dark), sun in dark (-> go light)
    b.title = dark?'switch to light mode':'switch to dark mode';
  });
}
function toggleTheme(){
  const next = currentTheme()==='light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  try{ localStorage.setItem('theme', next); }catch(e){}   // remember the last choice
  updateThemeIcons();
}
// ---- in-app confirm dialog (replaces the browser's confirm popup) ----
let _confirmCb=null;
function appConfirm(message, opts){
  opts=opts||{};
  return new Promise(resolve=>{
    _confirmCb=resolve;
    document.getElementById('confirmMsg').textContent=message;
    document.getElementById('confirmTitle').textContent=opts.title||'Confirm';
    const ok=document.getElementById('confirmOk');
    ok.textContent=opts.ok||'OK';
    ok.classList.toggle('danger', !!opts.danger);
    ok.classList.toggle('ok', !opts.danger);
    document.getElementById('confirmModal').style.display='flex';
    setTimeout(()=>{ try{ ok.focus(); }catch(e){} },0);
  });
}
function _confirmResolve(v){
  document.getElementById('confirmModal').style.display='none';
  const cb=_confirmCb; _confirmCb=null; if(cb) cb(v);
}
function _confirmOpen(){ return document.getElementById('confirmModal').style.display==='flex'; }
function className(i){
  if(i>=0 && i<classes.length) return classes[i];
  return 'class '+i;
}

function setStatus(t, cls){ const s=document.getElementById('status');
  s.textContent=t; s.className=cls||''; }
function markDirty(d){ dirty=d;
  document.getElementById('name').className = d ? 'dirty' : ''; }
function updateName(){
  document.getElementById('name').textContent = '['+(idx+1)+'/'+count+'] '+name;
  updateNav(); updateModeBadge();
}
// keep the top scrubber + position label in sync with the current image
function updateNav(){
  const s=document.getElementById('scrub');
  if(s){ s.max=Math.max(0, count-1); s.value=idx; }
  const np=document.getElementById('navpos');
  if(np) np.textContent=(count?idx+1:0)+' / '+count;
}

// ---- class selection UI ----
function buildClassUI(){
  const sel=document.getElementById('classsel');
  sel.innerHTML = classes.length
    ? classes.map((n,i)=>'<option value="'+i+'">'+i+': '+escapeHtml(n)+'</option>').join('')
    : '<option value="0">0: class 0</option>';
  const names = classes.length ? classes : ['class 0'];
  if(activeClass>=names.length) activeClass=0;
  hiddenClasses.clear();           // a new class set starts all-visible
  buildVisibleUI();
  setActiveClass(activeClass);
}
// per-class show/hide checkboxes
function buildVisibleUI(){
  const el=document.getElementById('visiblelist');
  if(!el) return;
  const names = classes.length ? classes : ['class 0'];
  el.innerHTML = names.map((n,i)=>
    '<label class="vis-row"><input type="checkbox" '+(hiddenClasses.has(i)?'':'checked')
    +' onchange="toggleClassVis('+i+',this.checked)">'
    +'<span class="sw" style="background:'+classColor(i)+'"></span>'
    +'<span class="vis-name">'+escapeHtml(n)+'</span></label>').join('');
}
function toggleVisSec(){
  const b=document.getElementById('visbody');
  const open = b.style.display==='none';
  b.style.display = open ? 'block' : 'none';
  document.getElementById('viscaret').style.transform = open ? 'rotate(90deg)' : '';  // chevron
}
function toggleClassVis(i, on){
  if(on) hiddenClasses.delete(i); else hiddenClasses.add(i);
  draw();
}
function setAllVisible(on){
  const names = classes.length ? classes : ['class 0'];
  hiddenClasses.clear();
  if(!on) for(let i=0;i<names.length;i++) hiddenClasses.add(i);
  buildVisibleUI(); draw();
}
function setActiveClass(i){
  if(i<0) return;
  const max = (classes.length||1)-1;
  if(i>max) return;
  activeClass=i;
  document.getElementById('classsel').value=String(i); syncDD(document.getElementById('classsel'));
  const ac=document.getElementById('activeclass');
  ac.innerHTML='class: <span class="sw" style="background:'+classColor(i)+'"></span> '
    +'<b>'+escapeHtml(className(i))+'</b> ('+i+')';
  // NB: this only sets the class for NEW boxes. The selected box's class is
  // changed only via the right-panel dropdown (setCls).
  saveSession();
}
function escapeHtml(s){ return String(s).replace(/[&<>"]/g,c=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// ---- custom dropdowns: replace a native <select> with a styled menu whose
//      options highlight on hover. The native select stays (hidden) as the
//      source of truth so existing value reads / change handlers keep working.
const _DD_CHEV='<svg viewBox="0 0 12 7" width="11" height="7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 1 6 6 11 1"/></svg>';
let _ddCloseOpen=null;
function syncDD(sel){ if(sel && sel._ddSync) sel._ddSync(); }
// drop menus whose trigger left the DOM (re-rendered lists), so none leak
function _ddCleanup(){ document.querySelectorAll('body > .dropdown-menu').forEach(m=>{
  if(m._trg && !m._trg.isConnected) m.remove(); }); }
function enhanceSelect(sel){
  if(!sel || sel.dataset.dd) return; sel.dataset.dd='1'; _ddCleanup();
  sel.style.display='none';
  const wrap=document.createElement('div'); wrap.className='dropdown';
  const trg=document.createElement('button'); trg.type='button'; trg.className='dropdown-trigger';
  trg.innerHTML='<span class="dropdown-value"></span>'+_DD_CHEV;
  wrap.appendChild(trg);
  sel.parentNode.insertBefore(wrap, sel);
  // the menu lives at body level (fixed) so scrollable panels never clip it
  const menu=document.createElement('div'); menu.className='dropdown-menu'; menu.style.display='none';
  menu._trg=trg; document.body.appendChild(menu);
  let searchInp=null, itemsWrap=null;
  function sync(){
    const cur=sel.options[sel.selectedIndex];
    trg.querySelector('.dropdown-value').textContent = cur?cur.textContent:'';
    trg.classList.toggle('disabled', !!sel.disabled);
    if(!itemsWrap) return;
    const items=itemsWrap.querySelectorAll('.dropdown-item');
    for(let i=0;i<items.length;i++) items[i].classList.toggle('active', i===sel.selectedIndex);
  }
  function applyFilter(q){
    if(!itemsWrap) return;
    q=(q||'').trim().toLowerCase();
    let any=false;
    itemsWrap.querySelectorAll('.dropdown-item').forEach(it=>{
      const m=!q || it.textContent.toLowerCase().includes(q);
      it.style.display=m?'block':'none'; if(m)any=true;
    });
    let nr=itemsWrap.querySelector('.dropdown-noresult');
    if(!any){ if(!nr){ nr=document.createElement('div'); nr.className='dropdown-noresult'; nr.textContent='No matches'; itemsWrap.appendChild(nr);} nr.style.display='block'; }
    else if(nr){ nr.style.display='none'; }
  }
  function rebuild(){
    menu.innerHTML=''; searchInp=null;
    if(sel.options.length>8){
      searchInp=document.createElement('input'); searchInp.type='text'; searchInp.className='dropdown-search';
      searchInp.placeholder='Search by id or name…';
      searchInp.addEventListener('input',()=>applyFilter(searchInp.value));
      searchInp.addEventListener('mousedown',e=>e.stopPropagation());
      searchInp.addEventListener('click',e=>e.stopPropagation());
      searchInp.addEventListener('keydown',e=>{ if(e.key==='Escape')close(); e.stopPropagation(); });
      menu.appendChild(searchInp);
    }
    itemsWrap=document.createElement('div'); itemsWrap.className='dropdown-items';
    for(let i=0;i<sel.options.length;i++){
      const o=sel.options[i];
      const it=document.createElement('button'); it.type='button'; it.className='dropdown-item';
      it.textContent=o.textContent;
      it.addEventListener('click',()=>{
        if(sel.value!==o.value){ sel.value=o.value; sel.dispatchEvent(new Event('change',{bubbles:true})); }
        close(); sync();
      });
      itemsWrap.appendChild(it);
    }
    menu.appendChild(itemsWrap);
    sync();
  }
  function place(){
    const r=trg.getBoundingClientRect();
    menu.style.left=r.left+'px'; menu.style.width=r.width+'px';
    const below=window.innerHeight-r.bottom-8, above=r.top-8;
    if(below<160 && above>below){           // not enough room below -> open upward
      menu.style.top='auto'; menu.style.bottom=(window.innerHeight-r.top+6)+'px';
      menu.style.maxHeight=Math.min(300,above)+'px';
    } else {
      menu.style.bottom='auto'; menu.style.top=(r.bottom+6)+'px';
      menu.style.maxHeight=Math.max(120,Math.min(300,below))+'px';
    }
  }
  function close(){ menu.style.display='none'; trg.classList.remove('open'); if(_ddCloseOpen===close)_ddCloseOpen=null; }
  function open(){ if(sel.disabled) return; if(_ddCloseOpen)_ddCloseOpen(); place(); menu.style.display='block'; trg.classList.add('open'); _ddCloseOpen=close;
    if(searchInp){ searchInp.value=''; applyFilter(''); setTimeout(()=>{try{searchInp.focus();}catch(e){}},0); } menu.scrollTop=0; }
  // mousedown must not reach the document handler (it would close the menu a tick
  // before the click toggles it, making an open menu immediately reopen)
  trg.addEventListener('mousedown',e=>e.stopPropagation());
  trg.addEventListener('click',e=>{ e.stopPropagation(); (menu.style.display==='none')?open():close(); });
  menu.addEventListener('mousedown',e=>e.stopPropagation());
  new MutationObserver(rebuild).observe(sel,{childList:true,subtree:true});
  sel.addEventListener('change',sync);
  sel._ddSync=sync;
  rebuild();
}
function enhanceSelects(ids){ ids.forEach(id=>{ const e=document.getElementById(id); if(e) enhanceSelect(e); }); }
document.addEventListener('mousedown',(e)=>{ if(_ddCloseOpen && !(e.target.closest && e.target.closest('.dropdown-menu'))) _ddCloseOpen(); });
window.addEventListener('scroll',(e)=>{ if(!_ddCloseOpen) return; if(e.target && e.target.closest && e.target.closest('.dropdown-menu')) return; _ddCloseOpen(); }, true);

// ---- normalised <-> pixel helpers ----
function toPix(b){
  return { x:(b.cx-b.w/2)*cv.width, y:(b.cy-b.h/2)*cv.height,
           w:b.w*cv.width, h:b.h*cv.height };
}
function fromPix(x,y,w,h,cls){
  return { cls:cls||0,
           cx:(x+w/2)/cv.width, cy:(y+h/2)/cv.height,
           w:w/cv.width, h:h/cv.height };
}

async function load(i, opts){
  opts=opts||{};
  if(i<0||i>=count) return;
  const r = await fetch('/api/item/'+i+'?t='+Date.now()).then(r=>r.json());
  idx=r.idx; name=r.name; count=r.count; origBoxes=r.boxes;
  if(opts.boxes){                // restoring a history snapshot (undo/redo)
    boxes = opts.boxes.map(b=>({...b})); touched=true; markDirty(true);
  } else {                       // normal open: existing labels are editable boxes
    boxes = origBoxes.map(b=>({...b})); touched=false; markDirty(false);
  }
  sel=-1;
  noteBaseline();                // remember this image's opening state (global history is kept)
  updateName();
  document.getElementById('jump').value = idx+1;
  img = new Image();
  img.onload = ()=>{ fit(); draw(); };
  img.src = '/api/image/'+i+'?t='+Date.now();
  setStatus(opts.boxes ? 'restored' : 'loaded');
  saveSession();
  if(opts.boxes) maybeAutosave();
}

let zoom=1, baseW=0, baseH=0;
function fit(){
  const wrap=document.getElementById('wrap');
  const maxW=wrap.clientWidth-20, maxH=wrap.clientHeight-20;
  let w=img.naturalWidth, h=img.naturalHeight;
  const s=Math.min(maxW/w, maxH/h, 3);
  baseW=Math.round(w*s); baseH=Math.round(h*s);
  applyCanvasSize();
}
function applyCanvasSize(){
  cv.width=Math.round(baseW*zoom);
  cv.height=Math.round(baseH*zoom);
}
cv.addEventListener('wheel', e=>{
  e.preventDefault();
  const wrap=document.getElementById('wrap');
  const wr=wrap.getBoundingClientRect();
  const rect=cv.getBoundingClientRect();
  const fx=(e.clientX-rect.left)/cv.width;
  const fy=(e.clientY-rect.top)/cv.height;
  const factor = e.deltaY<0 ? 1.1 : 1/1.1;
  zoom=Math.min(Math.max(zoom*factor, 0.2), 12);
  applyCanvasSize();
  draw();
  wrap.scrollLeft = fx*cv.width  - (e.clientX-wr.left) + cv.offsetLeft;
  wrap.scrollTop  = fy*cv.height - (e.clientY-wr.top)  + cv.offsetTop;
}, {passive:false});

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  boxes.forEach((b,i)=>{
    if(hiddenClasses.has(b.cls)) return;    // class hidden by the visibility filter
    const p=toPix(b);
    const col = classColor(b.cls);          // box colour always matches its class
    // translucent same-colour fill; selected box a bit stronger (Tools slider sets the base %)
    const fa = (i===sel ? Math.min(fillOpacity*2,80) : fillOpacity)/100;
    if(fa>0){ ctx.fillStyle = classFill(b.cls, fa); ctx.fillRect(p.x,p.y,p.w,p.h); }
    ctx.lineWidth = (i===sel)?3:1.5;         // selection shown by a thicker border + handles
    ctx.strokeStyle = col;
    ctx.strokeRect(p.x,p.y,p.w,p.h);
    ctx.fillStyle = col;
    ctx.font='12px monospace';
    const lbl='#'+(i+1)+' '+className(b.cls);
    ctx.fillText(lbl, p.x+2, p.y-3<8?p.y+12:p.y-3);
    if(i===sel) drawHandles(p);
  });
  renderPanel();
}

function renderPanel(){
  const list=document.getElementById('list');
  if(!list) return;
  // skip rebuild when nothing the panel shows changed (draw() runs on every drag move)
  const sig=boxes.map(b=>b.cls).join(',')+'|'+sel+'|'+[...hiddenClasses].join(',')+'|'+classes.length;
  if(sig===list._sig) return; list._sig=sig;
  const opts=(cur)=>{
    const names = classes.length ? classes : ['class 0'];
    return names.map((n,i)=>'<option value="'+i+'"'+(i===cur?' selected':'')+'>'
      +i+': '+escapeHtml(n)+'</option>').join('');
  };
  // only list boxes whose class is currently visible (mirrors the canvas)
  const shownN=boxes.filter(b=>!hiddenClasses.has(b.cls)).length;
  const hiddenN=boxes.length-shownN;
  let html='<div class="sec">Boxes on this image ('+shownN
    +(hiddenN?(' shown / '+boxes.length+' total'):'')+')</div>';
  if(!boxes.length){
    html+='<div style="color:var(--text-dim);padding:8px 10px;">none &mdash; draw a box</div>';
  } else if(!shownN){
    html+='<div style="color:var(--text-dim);padding:8px 10px;">all classes hidden</div>';
  } else {
    boxes.forEach((b,i)=>{
      if(hiddenClasses.has(b.cls)) return;
      html+='<div class="row'+(i===sel?' selrow':'')+'" onclick="selectBox('+i+')">'
        +'<span class="ix" style="color:'+classColor(b.cls)+'">#'+(i+1)+'</span>'
        +'<select onclick="event.stopPropagation()" onchange="setCls('+i+',this.value)">'+opts(b.cls)+'</select>'
        +'<button class="del" onclick="event.stopPropagation();removeBox('+i+')" title="delete box"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>'
        +'</div>';
    });
  }
  list.innerHTML=html;
  list.querySelectorAll('select').forEach(enhanceSelect);
}

function selectBox(i){ sel=i; draw(); }
function setCls(i, v){
  boxes[i].cls = parseInt(v,10) || 0;
  touched=true; markDirty(true); draw(); recordHistory(); maybeAutosave();
}
function removeBox(i){
  boxes.splice(i,1);
  if(sel===i) sel=-1; else if(sel>i) sel--;
  touched=true; markDirty(true); draw(); recordHistory(); maybeAutosave();
}

// ---- landing screen + mode ----
let appMode='local';   // 'local' = no CVAT options, 'cvat' = import/upload available
function applyMode(){
  // Local mode is clean: only the "Upload to CVAT" button (the upload modal picks
  // the project itself). CVAT mode also shows project select / import.
  document.getElementById('cvatsec').style.display='block';
  document.getElementById('cvatProjWrap').style.display = appMode==='cvat' ? 'block' : 'none';
  const tb=document.getElementById('tasksBtn'); if(tb) tb.style.display = appMode==='cvat' ? '' : 'none';
  updateModeBadge();
}
function updateModeBadge(){
  const el=document.getElementById('modebadge'); if(!el) return;
  if(appMode==='cvat' && linkedTask){
    el.textContent='CVAT task #'+linkedTask.task_id; el.className='mode cvat';
  } else if(appMode==='cvat'){
    el.textContent='CVAT'; el.className='mode cvat';
  } else { el.textContent='Local'; el.className='mode local'; }
}
// only one full-screen surface at a time
function hideAllScreens(){
  ['home','cvatbrowse','ccview','apmodal'].forEach(id=>{
    const e=document.getElementById(id); if(e) e.style.display='none'; });
}
function enterLocal(){
  appMode='local'; applyMode();
  hideAllScreens();
  if(img.complete && img.naturalWidth){ fit(); draw(); }
}
function enterImport(){
  appMode='cvat'; applyMode();
  hideAllScreens();
  openCvatBrowse();
}
function goHome(){ hideAllScreens(); document.getElementById('home').style.display='flex'; }
// resume the last opened folder, in the mode it belongs to (local, or CVAT if it
// was an imported task). The folder is already loaded server-side at startup.
function continueLastSession(){
  if(!count){ setStatus('no previous session'); return; }
  appMode = linkedTask ? 'cvat' : 'local'; applyMode();
  hideAllScreens();
  if(img.complete && img.naturalWidth){ fit(); draw(); }
  else { load(typeof idx==='number'?idx:0); }
  if(linkedTask && linkedTask.task_id) checkTaskUpdate(linkedTask.task_id);
}
// fill in the "continue last session" banner on the home page (if any)
function showContinueCard(m){
  const cc=document.getElementById('continueCard'); if(!cc) return;
  if(!(m && m.count && m.path)){ cc.style.display='none'; return; }
  const base=(String(m.path).split(/[\\/]/).filter(Boolean).pop())||m.path;
  document.getElementById('contName').textContent=base;
  const isCvat=!!m.linked_task, b=document.getElementById('contBadge');
  b.textContent=isCvat?'From CVAT':'Local'; b.className='cont-badge '+(isCvat?'cvat':'local');
  let meta=m.count+' image'+(m.count===1?'':'s');
  if(isCvat){ const lt=m.linked_task; meta+=' · task '+(lt.task_name?lt.task_name:('#'+lt.task_id)); }
  document.getElementById('contMeta').textContent=meta;
  cc.style.display='flex';
}

// remember where we are (current image + active class) so a restart resumes here.
// throttled, and never writes an empty image (avoids clobbering during boot).
let _sessTimer=null;
function saveSession(){
  clearTimeout(_sessTimer);
  _sessTimer=setTimeout(()=>{
    const body={active_class:activeClass};
    if(name) body.image=name;
    fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).catch(()=>{});
  }, 350);
}
// persist an imported class list to the folder's labels.txt so it survives a restart
async function persistClasses(){
  try{ await fetch('/api/classes',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({classes})}); }catch(e){}
}

// delete the current image (and its label) from disk, then advance
async function deleteImage(){
  if(!count){ setStatus('no image to delete'); return; }
  const extra = (appMode==='cvat' && linkedTask)
    ? '\\nIt will also be removed from the CVAT task on the next update.' : '';
  if(!(await appConfirm('Delete image "'+name+'" and its label from disk?\\nThis cannot be undone.'+extra,
       {title:'Delete image', ok:'Delete', danger:true}))) return;
  const di=idx;
  touched=false; markDirty(false);     // don't autosave the image we're deleting
  try{
    const r=await fetch('/api/delete/'+di,{method:'POST'}).then(r=>r.json());
    if(r.error){ setStatus('error: '+r.error); return; }
    count=r.count;
    setStatus('deleted '+r.deleted);
    if(!count){ name=''; boxes=[]; origBoxes=[]; sel=-1; updateName();
      const ctx2=cv.getContext('2d'); ctx2.clearRect(0,0,cv.width,cv.height); return; }
    load(Math.min(di, count-1));        // lands on the next image (or last)
  }catch(e){ setStatus('delete request failed'); }
}

function drawHandles(p){
  // white squares with a dark outline so they're visible over any class colour
  handlePts(p).forEach(pt=>{
    ctx.fillStyle='#fff';
    ctx.fillRect(pt.x-HANDLE/2, pt.y-HANDLE/2, HANDLE, HANDLE);
    ctx.lineWidth=1; ctx.strokeStyle='#000';
    ctx.strokeRect(pt.x-HANDLE/2, pt.y-HANDLE/2, HANDLE, HANDLE);
  });
}
function handlePts(p){
  const x=p.x,y=p.y,w=p.w,h=p.h;
  return [
    {x:x,     y:y,     id:'nw'}, {x:x+w/2, y:y,     id:'n'},
    {x:x+w,   y:y,     id:'ne'}, {x:x+w,   y:y+h/2, id:'e'},
    {x:x+w,   y:y+h,   id:'se'}, {x:x+w/2, y:y+h,   id:'s'},
    {x:x,     y:y+h,   id:'sw'}, {x:x,     y:y+h/2, id:'w'},
  ];
}
function mousePos(e){
  const r=cv.getBoundingClientRect();
  return { x:e.clientX-r.left, y:e.clientY-r.top };
}
function hitHandle(p, m){
  for(const pt of handlePts(p)){
    if(Math.abs(m.x-pt.x)<=HANDLE && Math.abs(m.y-pt.y)<=HANDLE) return pt.id;
  }
  return null;
}
function inside(p,m){ return m.x>=p.x && m.x<=p.x+p.w && m.y>=p.y && m.y<=p.y+p.h; }

cv.addEventListener('mousedown', e=>{
  const m=mousePos(e);
  if(e.button===2){
    for(let i=boxes.length-1;i>=0;i--){
      if(hiddenClasses.has(boxes[i].cls)) continue;
      if(inside(toPix(boxes[i]),m)){
        boxes.splice(i,1);
        if(sel===i) sel=-1; else if(sel>i) sel--;
        touched=true; markDirty(true); draw(); recordHistory(); maybeAutosave();
        setStatus('deleted box #'+(i+1)+' (right-click)');
        return;
      }
    }
    return;
  }
  if(sel>=0){
    const p=toPix(boxes[sel]);
    const h=hitHandle(p,m);
    if(h){ drag={mode:'resize', box:sel, handle:h, orig:{...p}}; return; }
  }
  for(let i=boxes.length-1;i>=0;i--){
    if(hiddenClasses.has(boxes[i].cls)) continue;
    if(inside(toPix(boxes[i]),m)){
      sel=i; drag={mode:'move', box:i, startX:m.x, startY:m.y, orig:toPix(boxes[i])};
      setActiveClassSilent(boxes[i].cls);
      draw(); return;
    }
  }
  sel=-1;
  drag={mode:'new', startX:m.x, startY:m.y, cur:{x:m.x,y:m.y,w:0,h:0}};
  draw();
});
// update the active-class indicator to match a clicked box, without re-applying
function setActiveClassSilent(i){
  if(i<0) return; const max=(classes.length||1)-1; if(i>max) return;
  activeClass=i; document.getElementById('classsel').value=String(i); syncDD(document.getElementById('classsel'));
  const ac=document.getElementById('activeclass');
  ac.innerHTML='class: <span class="sw" style="background:'+classColor(i)+'"></span> '
    +'<b>'+escapeHtml(className(i))+'</b> ('+i+')';
}

// suppress the browser context menu over the image so right-click can delete a box
document.getElementById('wrap').addEventListener('contextmenu', e=>e.preventDefault());

window.addEventListener('mousemove', e=>{
  if(!drag) return;
  const m=mousePos(e);
  m.x=Math.max(0, Math.min(m.x, cv.width));
  m.y=Math.max(0, Math.min(m.y, cv.height));
  if(drag.mode==='move'){
    const dx=m.x-drag.startX, dy=m.y-drag.startY;
    const o=drag.orig;
    const nx=Math.max(0, Math.min(o.x+dx, cv.width  - o.w));
    const ny=Math.max(0, Math.min(o.y+dy, cv.height - o.h));
    boxes[drag.box]=fromPix(nx, ny, o.w, o.h, boxes[drag.box].cls);
    touched=true; markDirty(true); draw();
  } else if(drag.mode==='resize'){
    let {x,y,w,h}=drag.orig;
    let x2=x+w, y2=y+h;
    const id=drag.handle;
    if(id.includes('w')) x=m.x;
    if(id.includes('e')) x2=m.x;
    if(id.includes('n')) y=m.y;
    if(id.includes('s')) y2=m.y;
    const nx=Math.min(x,x2), ny=Math.min(y,y2);
    const nw=Math.abs(x2-x), nh=Math.abs(y2-y);
    boxes[drag.box]=fromPix(nx,ny,nw,nh, boxes[drag.box].cls);
    touched=true; markDirty(true); draw();
  } else if(drag.mode==='new'){
    drag.cur={x:Math.min(drag.startX,m.x), y:Math.min(drag.startY,m.y),
              w:Math.abs(m.x-drag.startX), h:Math.abs(m.y-drag.startY)};
    draw();
    ctx.save();
    if(fillOpacity>0){ ctx.fillStyle=classFill(activeClass,fillOpacity/100);
      ctx.fillRect(drag.cur.x,drag.cur.y,drag.cur.w,drag.cur.h); }
    ctx.strokeStyle=classColor(activeClass);
    ctx.lineWidth=1.5;
    ctx.strokeRect(drag.cur.x,drag.cur.y,drag.cur.w,drag.cur.h);
    ctx.restore();
  }
});

window.addEventListener('mouseup', e=>{
  if(!drag) return;
  if(drag.mode==='new'){
    const c=drag.cur;
    if(c.w>3 && c.h>3){
      boxes.push(fromPix(c.x,c.y,c.w,c.h,activeClass));   // new box -> active class
      sel=boxes.length-1; touched=true; markDirty(true);
      // don't let a freshly drawn box be invisible because its class is hidden
      if(hiddenClasses.has(activeClass)){ hiddenClasses.delete(activeClass); buildVisibleUI(); }
    }
  }
  drag=null; draw();
  recordHistory();          // one history entry per completed add / move / resize
  maybeAutosave();
});

function maybeAutosave(){
  if(document.getElementById('autosave').checked && touched) save();
}

// ---- undo / redo history (GLOBAL across images) ----
// Each entry is {idx, boxes}. Undo/redo can cross image boundaries: it navigates
// to the entry's image and restores that snapshot, so edits stay undoable even
// after you move to another image.
let hist=[], histIdx=-1; const HIST_CAP=40;   // plenty of steps across images
let _loadBaseline=null;                        // boxes as this image was opened
function _snap(){ return boxes.map(b=>({...b})); }
function noteBaseline(){ _loadBaseline=_snap(); updateUndoButtons(); }
function recordHistory(){
  const snap=_snap();
  const sameTop = histIdx>=0 && hist[histIdx].idx===idx;
  const ref = sameTop ? hist[histIdx].boxes : _loadBaseline;
  if(ref && JSON.stringify(ref)===JSON.stringify(snap)) return;   // nothing actually changed
  hist=hist.slice(0,histIdx+1);                 // drop any redo branch
  if(!sameTop && _loadBaseline)                 // first edit on this image -> keep its baseline
    hist.push({idx, boxes:_loadBaseline});
  hist.push({idx, boxes:snap});
  while(hist.length>HIST_CAP) hist.shift();
  histIdx=hist.length-1;
  updateUndoButtons();
}
async function _applyHist(){
  const st=hist[histIdx];
  if(st.idx!==idx){ await load(st.idx, {boxes:st.boxes}); }   // jump to that image + restore
  else { boxes=st.boxes.map(b=>({...b})); sel=-1; touched=true; markDirty(true); draw(); maybeAutosave(); }
  updateUndoButtons();
}
async function undo(){ if(histIdx>0){ histIdx--; await _applyHist(); setStatus('undo'); } }
async function redo(){ if(histIdx<hist.length-1){ histIdx++; await _applyHist(); setStatus('redo'); } }
function updateUndoButtons(){
  const u=document.getElementById('undoBtn'), r=document.getElementById('redoBtn');
  if(u) u.disabled = histIdx<=0;
  if(r) r.disabled = histIdx>=hist.length-1;
}
function delSel(){
  if(sel<0){ setStatus('no box selected'); return; }
  boxes.splice(sel,1); sel=-1; touched=true; markDirty(true); draw(); recordHistory(); maybeAutosave();
}
async function clearAll(){
  if(!boxes.length) return;
  if(!(await appConfirm('Delete ALL boxes on this image?',
       {title:'Clear all', ok:'Delete all', danger:true}))) return;
  boxes=[]; sel=-1; touched=true; markDirty(true); draw(); recordHistory(); maybeAutosave();
}

async function save(){
  if(!touched){ setStatus('no changes — file left as is'); return; }
  const r=await fetch('/api/save/'+idx,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({boxes})}).then(r=>r.json());
  markDirty(false);
  setStatus('saved '+r.n+' box(es) ✓');
}

async function loadFolder(){
  const path=document.getElementById('folder').value.trim();
  if(!path){ setStatus('enter a folder path'); return; }
  const r=await fetch('/api/setfolder',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path})}).then(r=>r.json());
  if(r.error){ setStatus('error: '+r.error); return; }
  count=r.count; classes=r.classes||[]; activeClass=0;
  linkedTask=r.linked_task||null;
  classColors=[];                 // palette by default; exact CVAT colours below
  hideUpdateBanner();
  buildClassUI();
  setStatus('loaded '+r.count+' images, '+classes.length+' classes from '+r.path);
  if(count) load(0); else { name=''; updateName(); }
  if(linkedTask && linkedTask.project_id) applyCvatColorsFor(linkedTask.project_id);
}
// pull a CVAT project's label colours and match them to this folder's classes by name
async function applyCvatColorsFor(pid){
  if(!pid) return;
  try{
    const r=await fetch('/api/cvat/projectlabels?project_id='+encodeURIComponent(pid)
      +'&t='+Date.now()).then(r=>r.json());
    if(r.error || !r.classes) return;
    const map={}; r.classes.forEach((n,i)=>{ map[String(n).trim().toLowerCase()]=(r.colors||[])[i]||''; });
    classColors = classes.map(n=>map[String(n).trim().toLowerCase()]||'');
    buildClassUI(); draw();
  }catch(e){}
}

// ---- automatic annotation (modal + progress) ----
function setAaMsg(t){ const e=document.getElementById('aamsg'); if(e) e.textContent=t||''; }
function aaProgText(t){ const e=document.getElementById('aaprogtext'); if(e) e.textContent=t||''; }
function setBar(pct){ const e=document.getElementById('aabar');
  if(e) e.style.width=Math.max(0,Math.min(100,pct))+'%'; }
async function loadModelsInto(selId){
  const sel=document.getElementById(selId);
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/models?t='+Date.now()).then(r=>r.json());
    if(!r.models.length){ sel.innerHTML='<option value="">no .pt models found</option>'; return; }
    sel.innerHTML='<option value="">— select model —</option>'
      +r.models.map(m=>'<option value="'+escapeHtml(m.path)+'">'+escapeHtml(m.name)
      +'</option>').join('');
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; }
}
async function loadModels(){
  const host=document.getElementById('aamodellist');
  host.innerHTML='<div class="apt-empty">loading models…</div>';
  try{
    const r=await fetch('/api/models?t='+Date.now()).then(r=>r.json());
    if(!r.models||!r.models.length){ host.innerHTML='<div class="apt-empty">no .pt models found</div>'; aaModelCount(); return; }
    host.innerHTML=r.models.map(m=>'<label class="trow"><input type="checkbox" value="'+escapeHtml(m.path)
      +'" data-name="'+escapeHtml(m.name)+'" onchange="aaModelCount()"><span class="tname">'
      +escapeHtml(m.name)+'</span></label>').join('');
    aaModelCount();
  }catch(e){ host.innerHTML='<div class="apt-empty">error loading models</div>'; }
}
// three views: config -> map -> progress, with matching footer buttons
function aaButtons(cancel,back,next,run){
  document.getElementById('aacancel').style.display=cancel?'':'none';
  document.getElementById('aaback').style.display=back?'':'none';
  document.getElementById('aanext').style.display=next?'':'none';
  const r=document.getElementById('aarun'); r.style.display=run?'':'none'; r.disabled=false;
}
function aaShowConfig(){
  document.getElementById('aacfg').style.display='block';
  document.getElementById('aamap').style.display='none';
  document.getElementById('aaprog').style.display='none';
  document.getElementById('aacancel').textContent='Cancel';
  aaButtons(true,false,true,false); aaProgText(''); setBar(0);
}
function aaShowMap(){
  document.getElementById('aacfg').style.display='none';
  document.getElementById('aamap').style.display='block';
  document.getElementById('aaprog').style.display='none';
  aaButtons(false,true,false,true);
}
function showAaProgress(){
  document.getElementById('aacfg').style.display='none';
  document.getElementById('aamap').style.display='none';
  document.getElementById('aaprog').style.display='block';
  document.getElementById('aacancel').textContent='Close';
  aaButtons(true,false,false,false);
}
let aaModels=[];   // [{path,name,classes}] for each selected local model
function aaModelCount(){
  const n=document.querySelectorAll('#aamodellist input:checked').length;
  const e=document.getElementById('aamodelcount'); if(e) e.textContent=n?('· '+n+' selected'):'';
}
function aaSelectAllModels(on){
  document.querySelectorAll('#aamodellist input[type=checkbox]').forEach(c=>c.checked=on);
  aaModelCount();
}
function aaCheckedModels(){
  return [...document.querySelectorAll('#aamodellist input:checked')]
    .map(c=>({path:c.value, name:c.getAttribute('data-name')||c.value}));
}
async function aaNext(){
  const sel=aaCheckedModels();
  if(!sel.length){ setAaMsg('select at least one model'); return; }
  setAaMsg('loading model classes…');
  try{
    aaModels=[];
    for(const m of sel){
      const r=await fetch('/api/modelclasses?model='+encodeURIComponent(m.path)
        +'&t='+Date.now()).then(r=>r.json());
      if(r.error){ setAaMsg('error ('+m.name+'): '+r.error); return; }
      aaModels.push({path:m.path, name:m.name, classes:r.classes||[]});
    }
    if(!aaModels.some(m=>m.classes.length)){ setAaMsg('selected models expose no classes'); return; }
    buildAaMap(); setAaMsg(''); aaShowMap();
  }catch(e){ setAaMsg('failed to load model classes'); }
}
// build per-model mapping groups: each model class -> a class here (auto-matched)
function buildAaMap(){
  const host=document.getElementById('aamaplist');
  const names = classes.length ? classes : [];
  const lower={}; names.forEach((n,i)=>{ lower[String(n).trim().toLowerCase()]=i; });
  const opts=(selIdx)=>'<option value="">— skip —</option>'
    + names.map((n,i)=>'<option value="'+i+'"'+(i===selIdx?' selected':'')+'>'
        +i+': '+escapeHtml(n)+'</option>').join('');
  host.innerHTML = aaModels.map(m=>{
    const rows = m.classes.map((mn,mi)=>{
      const idx=lower[String(mn).trim().toLowerCase()];
      return '<div class="maprow"><span class="mc" title="'+escapeHtml(mn)+'">'+mi+': '
        +escapeHtml(mn)+'</span><span class="arr">→</span>'
        +'<select data-mi="'+mi+'">'+opts(idx===undefined?-1:idx)+'</select></div>';
    }).join('');
    return '<div class="aa-mgroup" data-model="'+escapeHtml(m.path)+'">'
      +'<div class="aa-mname">'+escapeHtml(m.name)+'</div>'+rows+'</div>';
  }).join('');
  host.querySelectorAll('select').forEach(enhanceSelect);
}
async function openAaModal(){
  document.getElementById('aamodal').style.display='flex';
  setAaMsg(''); aaShowConfig(); loadModels();
  // if a run is already in progress (modal was closed), jump back to it
  try{
    const s=await fetch('/api/autoannotate_status?t='+Date.now()).then(r=>r.json());
    if(s.running){ showAaProgress(); pollAutoAnnotate(); }
  }catch(e){}
}
function closeAaModal(){ document.getElementById('aamodal').style.display='none'; }
let aaPoll=null;
async function runAutoAnnotate(){
  const conf=parseFloat(document.getElementById('aaconf').value)||0.25;
  const mode=document.getElementById('aamode').value;
  // gather a per-model class mapping from each model's group of dropdowns
  const models=[];
  document.querySelectorAll('#aamaplist .aa-mgroup').forEach(g=>{
    const mapping={};
    g.querySelectorAll('select').forEach(s=>{
      if(s.value!=='') mapping[s.getAttribute('data-mi')]=parseInt(s.value,10);
    });
    models.push({model:g.getAttribute('data-model'), mapping});
  });
  if(!models.length){ setAaMsg('select at least one model'); return; }
  if(!models.some(m=>Object.keys(m.mapping).length)){ setAaMsg('map at least one class (or all are set to skip)'); return; }
  setAaMsg(''); showAaProgress();
  setBar(0); aaProgText('starting… ('+count+' images, '+models.length+' model'+(models.length>1?'s':'')+')');
  try{
    const r=await fetch('/api/autoannotate_all',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({models, conf, mode, classes})}).then(r=>r.json());
    if(r.error){ aaProgText('error: '+r.error); return; }
    pollAutoAnnotate();
  }catch(e){ aaProgText('request failed'); }
}
function pollAutoAnnotate(){
  clearInterval(aaPoll);
  aaPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/autoannotate_status?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    const pct = s.total ? Math.round((s.done/s.total)*100) : 0;
    setBar(s.running?pct:100);
    aaProgText((s.message||s.state||'')+(s.running&&s.total?(' — '+s.done+'/'+s.total):''));
    if(!s.running){
      clearInterval(aaPoll);
      setBar(100);
      document.getElementById('aacancel').textContent='Done';
      load(idx);                                    // refresh current image's labels
    }
  }, 800);
}

// ---- CVAT upload ----
let cvatLocked=false;
let linkedTask=null;     // {task_id,task_name,project_id} if this folder was imported
const LOCK_OPEN='<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-1.5"/></svg>';
const LOCK_SHUT='<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>';
function setCvatStatus(t){ document.getElementById('cvatstatus').textContent=t; }
// lock pins the chosen project so it can't change (and survives a list refresh)
function toggleCvatLock(){
  const sel=document.getElementById('cvatproj');
  if(!cvatLocked && !sel.value){ setCvatStatus('pick a project before locking'); return; }
  cvatLocked=!cvatLocked;
  sel.disabled=cvatLocked;
  const b=document.getElementById('cvatlock');
  b.innerHTML=cvatLocked?LOCK_SHUT:LOCK_OPEN;       // closed/green when locked
  b.classList.toggle('on',cvatLocked);
  b.title=cvatLocked?'unlock project':'lock project';
  setCvatStatus(cvatLocked?('locked to project '+sel.value):'project unlocked');
}
async function loadCvatProjects(refresh){
  const sel=document.getElementById('cvatproj');
  const prev=sel.value;                       // keep current pick across a refresh
  sel.innerHTML='<option value="">loading…</option>';
  setCvatStatus(refresh?'refreshing projects…':'loading projects…');
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>';
      setCvatStatus('error: '+r.error); return; }
    if(!r.projects.length){ sel.innerHTML='<option value="">no projects</option>';
      setCvatStatus('no projects in org "'+(r.org||'')+'"'); return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '
      +escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;                   // restore selection if still present
    setCvatStatus(r.projects.length+' projects'+(r.cached?' (cached)':'')
      +(cvatLocked?' — locked to '+sel.value:' — pick one'));
  }catch(e){ setCvatStatus('request failed'); }
}
// type/pick a project ID and pull its classes down to annotate with
async function fetchCvatClasses(){
  const pid=document.getElementById('cvatproj').value;
  if(!pid){ setCvatStatus('select a project'); return; }
  setCvatStatus('fetching classes for project '+pid+'…');
  try{
    const r=await fetch('/api/cvat/projectlabels?project_id='+encodeURIComponent(pid)
      +'&t='+Date.now()).then(r=>r.json());
    if(r.error){ setCvatStatus('error: '+r.error); return; }
    if(!r.classes.length){ setCvatStatus('project '+pid+' has no labels'); return; }
    classes=r.classes; classColors=r.colors||[]; activeClass=0; buildClassUI(); draw();   // recolour with CVAT's own colours
    persistClasses();
    setCvatStatus('using '+classes.length+' classes from CVAT project '+pid);
  }catch(e){ setCvatStatus('request failed'); }
}
function onCvatProjPick(){
  if(document.getElementById('cvatproj').value) fetchCvatClasses();
}
// ---- import classes from a CVAT project (available in local mode) ----
function openClsModal(){
  document.getElementById('clsmodal').style.display='flex';
  document.getElementById('clsmsg').textContent='';
  loadClsProjects();
}
function closeClsModal(){ document.getElementById('clsmodal').style.display='none'; }
async function loadClsProjects(refresh){
  const sel=document.getElementById('clsproj'); const prev=sel.value;
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; document.getElementById('clsmsg').textContent='error: '+r.error; return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '+escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; }
}
async function runImportClasses(){
  const pid=document.getElementById('clsproj').value;
  const msg=document.getElementById('clsmsg');
  if(!pid){ msg.textContent='select a project'; return; }
  msg.textContent='fetching classes for project '+pid+'…';
  try{
    const r=await fetch('/api/cvat/projectlabels?project_id='+encodeURIComponent(pid)
      +'&t='+Date.now()).then(r=>r.json());
    if(r.error){ msg.textContent='error: '+r.error; return; }
    if(!r.classes.length){ msg.textContent='project '+pid+' has no labels'; return; }
    classes=r.classes; classColors=r.colors||[]; activeClass=0; buildClassUI(); draw();
    persistClasses();
    setStatus('imported '+classes.length+' classes from CVAT project '+pid+' ✓');
    closeClsModal();
  }catch(e){ msg.textContent='request failed'; }
}
// ---- CVAT upload modal + progress ----
let cvatPoll=null, _lastUploadProj=null;
function cvMsg(t){ const e=document.getElementById('cvmsg'); if(e) e.textContent=t||''; }
function cvProgText(t){ const e=document.getElementById('cvprogtext'); if(e) e.textContent=t||''; }
function cvIndet(on){
  const w=document.getElementById('cvbarwrap'), b=document.getElementById('cvbar');
  if(on){ w.classList.add('indet'); b.style.width=''; b.style.marginLeft=''; }
  else { w.classList.remove('indet'); b.style.marginLeft='0'; b.style.width='100%'; }
}
function showCvConfig(){
  document.getElementById('cvcfg').style.display='block';
  document.getElementById('cvprog').style.display='none';
  const run=document.getElementById('cvrun'); run.style.display=''; run.disabled=false;
  document.getElementById('cvcancel').textContent='Cancel';
  cvProgText('');
}
function showCvProgress(){
  document.getElementById('cvcfg').style.display='none';
  document.getElementById('cvprog').style.display='block';
  document.getElementById('cvrun').style.display='none';
  document.getElementById('cvcancel').textContent='Close';
}
// the "update source task" option only applies in CVAT mode (imported folder)
function cvIsUpdate(){
  return appMode==='cvat' && linkedTask && document.getElementById('cvupdate').checked;
}
// updating an imported task hides the "create new" (project + task name) fields
function cvUpdateToggle(){
  document.getElementById('cvnewwrap').style.display = cvIsUpdate() ? 'none' : 'block';
  updateCvRunState();
}
function updateCvRunState(){
  const run=document.getElementById('cvrun');
  if(cvIsUpdate()){ run.disabled=false; cvMsg(''); return; }
  if(document.getElementById('cvUploadProj').value){ run.disabled=false; cvMsg(''); }
  else { run.disabled=true; cvMsg('select a project to upload into'); }
}
// the upload modal carries its own project dropdown (cached list)
async function loadCvUploadProjects(refresh){
  const sel=document.getElementById('cvUploadProj'); const prev=sel.value;
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '+escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; }
  updateCvRunState();
}
async function openCvModal(){
  document.getElementById('cvmodal').style.display='flex';
  cvMsg(''); showCvConfig();
  // linked-task (imported folder) update option — only in CVAT mode
  const lw=document.getElementById('cvlinkwrap');
  if(appMode==='cvat' && linkedTask){
    lw.style.display='block';
    document.getElementById('cvlinkedname').textContent='#'+linkedTask.task_id
      +(linkedTask.task_name?(' ('+linkedTask.task_name+')'):'');
    document.getElementById('cvupdate').checked=true;
  } else lw.style.display='none';
  // populate the modal's project dropdown; preselect sidebar pick if any
  const side=document.getElementById('cvatproj').value;
  await loadCvUploadProjects();
  if(side){ document.getElementById('cvUploadProj').value=side; syncDD(document.getElementById('cvUploadProj')); }
  cvUpdateToggle();
  // if an upload is already running, jump straight to its progress
  try{
    const s=await fetch('/api/cvat/uploadstatus?t='+Date.now()).then(r=>r.json());
    if(s.running){ showCvProgress(); cvIndet(true); pollCvatUpload(); }
  }catch(e){}
}
function closeCvModal(){ document.getElementById('cvmodal').style.display='none'; }
async function runCvatUpload(){
  const updating = cvIsUpdate();
  let body;
  if(updating){
    body={task_id:linkedTask.task_id, classes};
    _lastUploadProj = linkedTask.project_id;
  } else {
    const pid=document.getElementById('cvUploadProj').value;
    const tname=document.getElementById('cvattask').value.trim();
    if(!pid){ cvMsg('select a project to upload into'); return; }
    if(!tname){ cvMsg('enter a task name'); return; }
    body={project_id:pid, task_name:tname, classes};
    _lastUploadProj = pid;
  }
  cvMsg(''); showCvProgress(); cvIndet(true);
  cvProgText(updating ? ('updating annotations on task #'+linkedTask.task_id+'…')
                      : ('starting upload of '+count+' images…'));
  try{
    const r=await fetch('/api/cvat/upload',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).then(r=>r.json());
    if(r.error){ cvProgText('error: '+r.error); cvIndet(false); return; }
    cvProgText(updating ? 'uploading annotations…'
      : ('uploading '+r.count+' images'+(r.annotations?' + annotations':'')+'…'));
    pollCvatUpload();
  }catch(e){ cvProgText('request failed'); cvIndet(false); }
}
function pollCvatUpload(){
  clearInterval(cvatPoll);
  cvatPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/uploadstatus?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    cvProgText(s.message||s.state||'');
    if(!s.running){
      clearInterval(cvatPoll);
      cvIndet(false);
      document.getElementById('cvcancel').textContent='Done';
      if(s.error){ cvProgText('failed: '+s.error); }
      else {
        cvProgText('done ✓'+(s.task_id?(' — task '+s.task_id):'')
          +(s.task_url?' ('+s.task_url+')':'')+' — opening tasks…');
        // hand the user back to the tasks page for that project
        setTimeout(()=>{ closeCvModal(); returnToTasks(true, _lastUploadProj); }, 1100);
      }
    }
  }, 1500);
}

// ---- CVAT browser: project cards -> task cards -> import ----
let impPoll=null, bProjects=[], bTasks=[], browseState='projects',
    browseProjectId=null, browseProjectName='';
function bCard(id,name,sub,onclick,badge){
  return '<div class="bcard" onclick="'+onclick+'">'+(badge||'')+'<div class="bc-id">#'+id+'</div>'
    +'<div class="bc-name">'+escapeHtml(name)+'</div>'
    +(sub?'<div class="bc-sub">'+escapeHtml(sub)+'</div>':'')+'</div>';
}
const _ICO_CHECK='<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
const _ICO_SYNC='<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/></svg>';
function taskBadge(t){
  // grid signals whether a task is imported; a small yellow dot beside it hints
  // that a newer version exists on CVAT (full update prompt shows on open).
  if(!t.imported) return '';
  const dot = t.up_to_date===false ? '<span class="bc-dot" title="update available on CVAT"></span>' : '';
  return '<span class="bc-badge ok">'+_ICO_CHECK+'imported'+dot+'</span>';
}
function openCvatBrowse(){
  document.getElementById('cvatbrowse').style.display='flex';
  browseProjects();
}
function closeCvatBrowse(){ document.getElementById('cvatbrowse').style.display='none'; }
// jump back to the tasks grid for the current task's project (topnav button +
// used after an upload). pidArg overrides which project's tasks to show.
async function returnToTasks(refresh, pidArg){
  const pid = pidArg || (linkedTask && linkedTask.project_id) || browseProjectId;
  document.getElementById('cvatbrowse').style.display='flex';
  if(!pid){ browseProjects(); return; }          // no project context -> project list
  let pname = (String(browseProjectId)===String(pid) && browseProjectName) ? browseProjectName : '';
  if(!pname){
    let p=bProjects.find(x=>String(x.id)===String(pid));
    if(!p){
      try{ const r=await fetch('/api/cvat/projects?t='+Date.now()).then(r=>r.json());
        if(r.projects) bProjects=r.projects; p=bProjects.find(x=>String(x.id)===String(pid)); }catch(e){}
    }
    pname = p ? p.name : ('#'+pid);
  }
  openProject(pid, pname, refresh);
}
async function browseProjects(refresh){
  browseState='projects';
  document.getElementById('browseBack').style.display='none';
  document.getElementById('browseUpdateAll').style.display='none';
  document.getElementById('browseTitle').textContent='Import from CVAT — projects';
  const grid=document.getElementById('browseGrid');
  grid.innerHTML='<div class="browse-empty">loading projects…</div>';
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ grid.innerHTML='<div class="browse-empty">error: '+escapeHtml(r.error)+'</div>'; return; }
    bProjects=r.projects||[];
    grid.innerHTML = bProjects.length
      ? bProjects.map((p,i)=>bCard(p.id,p.name,'open tasks →','openProjectIdx('+i+')')).join('')
      : '<div class="browse-empty">no projects</div>';
  }catch(e){ grid.innerHTML='<div class="browse-empty">failed to load projects</div>'; }
}
function openProjectIdx(i){ const p=bProjects[i]; if(p) openProject(p.id, p.name); }
async function openProject(pid, pname, refresh){
  browseState='tasks'; browseProjectId=pid; browseProjectName=pname;
  document.getElementById('browseBack').style.display='';
  document.getElementById('browseTitle').textContent='Tasks in '+pname;
  const grid=document.getElementById('browseGrid');
  grid.innerHTML='<div class="browse-empty">loading tasks…</div>';
  try{
    const r=await fetch('/api/cvat/tasks?project_id='+encodeURIComponent(pid)
      +(refresh?'&refresh=1':'')+'&t='+Date.now()).then(r=>r.json());
    if(r.error){ grid.innerHTML='<div class="browse-empty">error: '+escapeHtml(r.error)+'</div>'; return; }
    bTasks=r.tasks||[];
    grid.innerHTML = bTasks.length
      ? bTasks.map((t,i)=>{
          const act=t.imported?'open →':'import →';
          return bCard(t.id,t.name,(t.size!=null?t.size+' images · ':'')+act,'openTaskIdx('+i+')',taskBadge(t));
        }).join('')
      : '<div class="browse-empty">no tasks in this project</div>';
    // offer "Update all" only when something is actually imported here
    document.getElementById('browseUpdateAll').style.display = bTasks.some(t=>t.imported) ? '' : 'none';
  }catch(e){ grid.innerHTML='<div class="browse-empty">failed to load tasks</div>'; }
}
function openTaskIdx(i){ const t=bTasks[i]; if(t) openTask(t); }
async function openTask(t){
  // already imported -> open the local copy right away (no re-download), then
  // check CVAT in the background and offer an update if a newer version exists.
  if(t.imported && t.local_path){
    closeCvatBrowse();
    appMode='cvat'; applyMode();
    document.getElementById('folder').value=t.local_path;
    await loadFolder();
    setStatus('opened local copy of "'+t.name+'"');
    checkTaskUpdate(t.id);
    return;
  }
  // not imported yet -> download then open (progress overlay)
  document.getElementById('browseProg').style.display='flex';
  document.getElementById('browseProgText').textContent='importing "'+t.name+'"…';
  try{
    const r=await fetch('/api/cvat/import',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:t.id})}).then(r=>r.json());
    if(r.error){ document.getElementById('browseProgText').textContent='error: '+r.error; return; }
    browsePollImport();
  }catch(e){ document.getElementById('browseProgText').textContent='request failed'; }
}
// ---- "update available" banner for an opened imported task ----
function hideUpdateBanner(){ const b=document.getElementById('cvupdatebanner'); if(b) b.style.display='none'; }
function showUpdateBanner(tid){
  const b=document.getElementById('cvupdatebanner'); if(!b) return;
  b.dataset.tid=tid;
  document.getElementById('cvupdtext').textContent='A newer version of this task is available on CVAT.';
  document.getElementById('cvupdbtn').style.display='';
  b.style.display='flex';
}
async function checkTaskUpdate(tid){
  hideUpdateBanner();
  if(!tid) return;
  try{
    const r=await fetch('/api/cvat/taskstatus?task_id='+tid+'&t='+Date.now()).then(r=>r.json());
    if(r && r.imported && r.up_to_date===false) showUpdateBanner(tid);
  }catch(e){}
}
async function updateTaskNow(){
  const b=document.getElementById('cvupdatebanner'); const tid=b&&b.dataset.tid;
  if(!tid) return;
  document.getElementById('cvupdbtn').style.display='none';
  document.getElementById('cvupdtext').textContent='Updating from CVAT…';
  try{
    const r=await fetch('/api/cvat/import',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:tid, labels_only:true})}).then(r=>r.json());
    if(r.error){ document.getElementById('cvupdtext').textContent='update failed: '+r.error; return; }
    clearInterval(impPoll);
    impPoll=setInterval(async()=>{
      let s; try{ s=await fetch('/api/cvat/importstatus?t='+Date.now()).then(r=>r.json()); }catch(e){ return; }
      document.getElementById('cvupdtext').textContent=s.message||s.state||'updating…';
      if(!s.running){
        clearInterval(impPoll);
        if(s.error){ document.getElementById('cvupdbtn').style.display=''; return; }
        if(s.path){
          hideUpdateBanner();
          document.getElementById('folder').value=s.path;
          await loadFolder();
          setStatus(s.message||'labels updated from CVAT ✓');
        }
      }
    }, 1000);
  }catch(e){ document.getElementById('cvupdtext').textContent='update request failed'; document.getElementById('cvupdbtn').style.display=''; }
}
function browsePollImport(){
  clearInterval(impPoll);
  impPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/importstatus?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    document.getElementById('browseProgText').textContent=s.message||s.state||'';
    if(!s.running){
      clearInterval(impPoll);
      if(s.error){ return; }                     // leave the error shown; user can close
      if(s.path){
        document.getElementById('browseProgText').textContent='imported '+s.count+' images ✓ — opening…';
        document.getElementById('browseProg').style.display='none';
        closeCvatBrowse();
        document.getElementById('folder').value=s.path;
        loadFolder();                            // switch to the annotation view
      }
    }
  }, 1000);
}
function browseRefresh(){
  if(browseState==='tasks') openProject(browseProjectId, browseProjectName, true);
  else browseProjects(true);
}
// update every imported task in this project that's out of date (others are
// skipped server-side, so it only downloads what actually changed).
async function updateAllTasks(){
  const ids = bTasks.filter(t=>t.imported).map(t=>t.id);
  const prog=document.getElementById('browseProg'), txt=document.getElementById('browseProgText');
  prog.style.display='flex';
  if(!ids.length){ txt.textContent='No imported tasks in this project to update.'; return; }
  txt.textContent='updating '+ids.length+' imported task(s)…';
  try{
    const r=await fetch('/api/cvat/updateall',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_ids:ids})}).then(r=>r.json());
    if(r.error){ txt.textContent='error: '+r.error; return; }
    pollUpdateAll();
  }catch(e){ txt.textContent='request failed'; }
}
function pollUpdateAll(){
  clearInterval(impPoll);
  impPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/updateall_status?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    let t=s.message||s.state||'';
    if(s.running && s.total) t+=' ('+s.done+'/'+s.total+')';
    document.getElementById('browseProgText').textContent=t;
    if(!s.running){
      clearInterval(impPoll);
      // refresh the grid so the dots/badges reflect the new state
      setTimeout(()=>{ document.getElementById('browseProg').style.display='none';
        openProject(browseProjectId, browseProjectName, true); }, 1300);
    }
  }, 1000);
}
function browseCancelProg(){
  document.getElementById('browseProg').style.display='none';
  clearInterval(impPoll);
}

// ---- Automatic annotations -> CVAT pipeline modal ----
let apModelClasses=[], apProjectClasses=[], apPoll=null;
function apMsg(t){ const e=document.getElementById('apmsg'); if(e) e.textContent=t||''; }
function apProgText(t){ const e=document.getElementById('approgtext'); if(e) e.textContent=t||''; }
function apBar(pct){ const e=document.getElementById('apbar'); if(e) e.style.width=Math.max(0,Math.min(100,pct))+'%'; }
function apButtons(cancel,back,next,run,min){
  document.getElementById('apcancel').style.display=cancel?'':'none';
  document.getElementById('apback').style.display=back?'':'none';
  document.getElementById('apnext').style.display=next?'':'none';
  document.getElementById('apmin').style.display=min?'':'none';
  const r=document.getElementById('aprun'); r.style.display=run?'':'none'; r.disabled=false;
}
function apShowConfig(){
  document.getElementById('apcfg').style.display='block';
  document.getElementById('apmap').style.display='none';
  document.getElementById('approg').style.display='none';
  document.getElementById('apcancel').textContent='Cancel';
  apButtons(true,false,true,false); apProgText(''); apBar(0);
}
function apShowMap(){
  document.getElementById('apcfg').style.display='none';
  document.getElementById('apmap').style.display='block';
  document.getElementById('approg').style.display='none';
  apButtons(false,true,false,true);
}
function apShowProgress(){
  document.getElementById('apcfg').style.display='none';
  document.getElementById('apmap').style.display='none';
  document.getElementById('approg').style.display='block';
  document.getElementById('apcancel').textContent='Close';
  apButtons(true,false,false,false,true);   // Minimize available while running
}
async function enterAuto(){
  appMode='cvat'; applyMode();
  hideAllScreens();
  apMinimized=false; apShowWidget(false);   // full view -> no floating widget
  document.getElementById('apmodal').style.display='flex';
  apMsg(''); apShowConfig();
  document.getElementById('aptasklist').innerHTML='<div class="apt-empty">— select a project first —</div>';
  apTaskCount();
  loadApProjects(); loadModelsInto('apmodel');
  try{
    const s=await fetch('/api/cvat/autopipeline_status?t='+Date.now()).then(r=>r.json());
    if(s.running){ apShowProgress(); pollAutoPipeline(); }
  }catch(e){}
}
function closeApModal(){
  document.getElementById('apmodal').style.display='none';
  goHome();                 // standalone flow: return to the landing screen
}
// ---- minimize the running pipeline to a floating widget ----
let apMinimized=false;
function apShowWidget(on){ const w=document.getElementById('apwidget'); if(w) w.style.display=on?'block':'none'; }
function apSetWidget(pct,text,done){
  const bar=document.getElementById('apwbar'), t=document.getElementById('apwtext'), w=document.getElementById('apwidget');
  if(bar) bar.style.width=Math.max(0,Math.min(100,pct))+'%';
  if(t) t.textContent=text||'';
  if(w){ w.classList.toggle('done',!!done);
    w.querySelector('.apw-title').textContent = done ? 'Auto-annotation done' : 'Auto-annotating…'; }
}
function minimizeAuto(){
  apMinimized=true;
  document.getElementById('apmodal').style.display='none';   // reveal whatever is behind (editor)
  apShowWidget(true);
}
function restoreAuto(){
  apMinimized=false;
  apShowWidget(false);
  appMode='cvat'; applyMode();
  hideAllScreens();
  document.getElementById('apmodal').style.display='flex';
  apShowProgress();
}
function dismissAutoWidget(){ apMinimized=false; apShowWidget(false); }
async function loadApProjects(refresh){
  const sel=document.getElementById('approj'); const prev=sel.value;
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; apMsg('error: '+r.error); return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '+escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;
    if(refresh) apMsg(r.projects.length+' projects'+(r.cached?' (cached)':' refreshed'));
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; apMsg('project list failed'); }
}
async function loadApTasks(refresh){
  const pid=document.getElementById('approj').value;
  const host=document.getElementById('aptasklist');
  if(!pid){ host.innerHTML='<div class="apt-empty">— select a project first —</div>'; apTaskCount(); return; }
  host.innerHTML='<div class="apt-empty">loading tasks…</div>';
  try{
    const r=await fetch('/api/cvat/tasks?project_id='+encodeURIComponent(pid)
      +(refresh?'&refresh=1':'')+'&t='+Date.now()).then(r=>r.json());
    if(r.error){ host.innerHTML='<div class="apt-empty">error: '+escapeHtml(r.error)+'</div>'; return; }
    if(!r.tasks.length){ host.innerHTML='<div class="apt-empty">no tasks in this project</div>'; return; }
    host.innerHTML=r.tasks.map(t=>'<label class="trow"><input type="checkbox" value="'+t.id
      +'" onchange="apTaskCount()"><span class="tid">#'+t.id+'</span>'
      +'<span class="tname">'+escapeHtml(t.name)+(t.size!=null?(' · '+t.size+' imgs'):'')
      +'</span></label>').join('');
    apTaskCount();
    if(refresh) apMsg(r.tasks.length+' tasks'+(r.cached?' (cached)':' refreshed'));
  }catch(e){ host.innerHTML='<div class="apt-empty">failed to load tasks</div>'; }
}
function apCheckedTasks(){
  return [...document.querySelectorAll('#aptasklist input:checked')].map(c=>c.value);
}
function apTaskCount(){
  const n=apCheckedTasks().length;
  document.getElementById('aptaskcount').textContent = n?('('+n+' selected)'):'';
}
function apSelectAllTasks(on){
  document.querySelectorAll('#aptasklist input[type=checkbox]').forEach(c=>c.checked=on);
  apTaskCount();
}
async function apNext(){
  const pid=document.getElementById('approj').value;
  const tasks=apCheckedTasks();
  const model=document.getElementById('apmodel').value;
  if(!pid){ apMsg('select a project'); return; }
  if(!tasks.length){ apMsg('select at least one task'); return; }
  if(!model){ apMsg('select a model'); return; }
  apMsg('loading model + project classes…');
  try{
    const [mc,pc]=await Promise.all([
      fetch('/api/modelclasses?model='+encodeURIComponent(model)+'&t='+Date.now()).then(r=>r.json()),
      fetch('/api/cvat/projectlabels?project_id='+encodeURIComponent(pid)+'&t='+Date.now()).then(r=>r.json())
    ]);
    if(mc.error){ apMsg('model: '+mc.error); return; }
    if(pc.error){ apMsg('project: '+pc.error); return; }
    apModelClasses=mc.classes||[]; apProjectClasses=pc.classes||[];
    if(!apModelClasses.length){ apMsg('model exposes no classes'); return; }
    if(!apProjectClasses.length){ apMsg('project has no labels'); return; }
    buildApMap(); apMsg(''); apShowMap();
  }catch(e){ apMsg('failed to load classes'); }
}
function buildApMap(){
  const host=document.getElementById('apmaplist');
  const lower={}; apProjectClasses.forEach(n=>{ lower[String(n).trim().toLowerCase()]=n; });
  const opts=(sel)=>'<option value="">— skip —</option>'
    + apProjectClasses.map(n=>'<option value="'+escapeHtml(n)+'"'+(n===sel?' selected':'')+'>'
        +escapeHtml(n)+'</option>').join('');
  host.innerHTML=apModelClasses.map((mn,mi)=>{
    const match=lower[String(mn).trim().toLowerCase()];
    return '<div class="maprow"><span class="mc" title="'+escapeHtml(mn)+'">'+mi+': '
      +escapeHtml(mn)+'</span><span class="arr">→</span>'
      +'<select data-mi="'+mi+'">'+opts(match||'')+'</select></div>';
  }).join('');
  host.querySelectorAll('select').forEach(enhanceSelect);
}
async function runAutoPipeline(){
  const tasks=apCheckedTasks();
  const model=document.getElementById('apmodel').value;
  const conf=parseFloat(document.getElementById('apconf').value)||0.25;
  const mode=document.getElementById('apmode').value;
  const mapping={};
  document.querySelectorAll('#apmaplist select').forEach(s=>{
    if(s.value!=='') mapping[s.getAttribute('data-mi')]=s.value;   // model idx -> class NAME
  });
  if(!tasks.length){ apMsg('select at least one task'); return; }
  if(!Object.keys(mapping).length){ apMsg('map at least one class'); return; }
  apMsg(''); apShowProgress(); apBar(0); apProgText('starting '+tasks.length+' task(s)…');
  try{
    const r=await fetch('/api/cvat/autopipeline',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_ids:tasks, model, conf, mode, mapping})}).then(r=>r.json());
    if(r.error){ apProgText('error: '+r.error); return; }
    pollAutoPipeline();
  }catch(e){ apProgText('request failed'); }
}
function pollAutoPipeline(){
  clearInterval(apPoll);
  apPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/autopipeline_status?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    // overall progress across all tasks: completed tasks + fraction of current
    const frac = (s.state==='annotating' && s.total) ? (s.done/s.total) : 0;
    const pct = s.n_tasks ? Math.round(((s.cur_task-1)+frac)/s.n_tasks*100) : (s.running?5:0);
    const barPct = s.running?Math.max(2,pct):100;
    let txt = (s.message||s.state||'')+(s.state==='annotating'&&s.total?(' — '+s.done+'/'+s.total):'');
    apBar(barPct); apProgText(txt);
    apSetWidget(barPct, txt, false);
    if(!s.running){
      clearInterval(apPoll); apBar(100);
      document.getElementById('apcancel').textContent='Done';
      if(s.error){ apProgText('failed: '+s.error); apSetWidget(100,'failed: '+s.error,true); }
      else {
        const done='done ✓ — '+s.added+' boxes across '+(s.done_tasks||s.n_tasks)+' task(s)'
          +(s.task_url?' · '+s.task_url:'');
        apProgText(done); apSetWidget(100,done,true);
      }
    }
  }, 1000);
}

// ---- Class count (project- and task-wise) ----
let ccPoll=null;
async function enterCount(){
  appMode='cvat'; applyMode();
  hideAllScreens();
  document.getElementById('ccview').style.display='flex';
  document.getElementById('ccresult').innerHTML='';
  document.getElementById('ccprog').style.display='none';
  document.getElementById('ccempty').style.display='block';
  ccLoadProjects();
  try{
    const s=await fetch('/api/cvat/classcount_status?t='+Date.now()).then(r=>r.json());
    if(s.running){ ccShowProgress(); pollClassCount(); }
    else if(s.result){ renderCcTable(s.result); }
  }catch(e){}
}
async function ccLoadProjects(refresh){
  const sel=document.getElementById('ccproj'); const prev=sel.value;
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/cvat/projects?'+(refresh?'refresh=1&':'')+'t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '+escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; }
}
function ccShowProgress(){
  document.getElementById('ccempty').style.display='none';
  document.getElementById('ccresult').innerHTML='';
  document.getElementById('ccprog').style.display='block';
}
async function ccRun(){
  const pid=document.getElementById('ccproj').value;
  if(!pid){ document.getElementById('ccempty').style.display='block';
    document.getElementById('ccempty').textContent='Select a project first.'; return; }
  ccShowProgress();
  document.getElementById('ccprogtext').textContent='starting…';
  try{
    const r=await fetch('/api/cvat/classcount',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({project_id:pid})}).then(r=>r.json());
    if(r.error){ document.getElementById('ccprogtext').textContent='error: '+r.error; return; }
    pollClassCount();
  }catch(e){ document.getElementById('ccprogtext').textContent='request failed'; }
}
function pollClassCount(){
  clearInterval(ccPoll);
  ccPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/classcount_status?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    document.getElementById('ccprogtext').textContent=(s.message||s.state||'')
      +(s.state==='counting'&&s.total?(' — '+s.done+'/'+s.total):'');
    if(!s.running){
      clearInterval(ccPoll);
      document.getElementById('ccprog').style.display='none';
      if(s.error){ document.getElementById('ccempty').style.display='block';
        document.getElementById('ccempty').textContent='Failed: '+s.error; return; }
      if(s.result) renderCcTable(s.result);
    }
  }, 1000);
}
function renderCcTable(res){
  const tasks=res.tasks||[], classes=res.classes||[], counts=res.counts||{};
  document.getElementById('ccempty').style.display='none';
  let h='<div class="cc-summary">Project <b>#'+res.project_id+'</b> · <b>'+classes.length
    +'</b> classes · <b>'+tasks.length+'</b> tasks · <b>'+res.grand_total
    +'</b> total annotations</div>';
  h+='<table class="cc-table"><thead><tr><th class="cc-class">Class</th>'
    +'<th>Total</th>'
    +tasks.map(t=>'<th>'+escapeHtml(t.name)+'<span class="tkid">#'+t.id+'</span></th>').join('')
    +'</tr></thead><tbody>';
  classes.forEach(cn=>{
    const c=counts[cn]||{total:0,tasks:{}};
    h+='<tr><td class="cc-class">'+escapeHtml(cn)+'</td>'
      +'<td class="cc-total'+(c.total?'':' zero')+'">'+(c.total||0)+'</td>'
      +tasks.map(t=>{ const v=(c.tasks||{})[String(t.id)]||0;
        return '<td class="'+(v?'':'zero')+'">'+v+'</td>'; }).join('')
      +'</tr>';
  });
  // totals row
  h+='<tr class="cc-totalrow"><td class="cc-class">TOTAL</td><td>'+res.grand_total+'</td>'
    +tasks.map(t=>'<td>'+((res.task_totals||{})[String(t.id)]||0)+'</td>').join('')
    +'</tr>';
  h+='</tbody></table>';
  document.getElementById('ccresult').innerHTML=h;
}

async function saveIfDirty(){
  if(touched){
    if(document.getElementById('autosave').checked) await save();
    else if(dirty && await appConfirm('Unsaved changes — save before moving?',
            {title:'Unsaved changes', ok:'Save'})) await save();
  }
}
async function navTo(i){
  if(i<0||i>=count) return;
  await saveIfDirty();
  load(i);
}
async function go(d){
  await saveIfDirty();
  const t=idx+d;
  if(t>=0 && t<count) load(t);
}
function jump(){
  const v=parseInt(document.getElementById('jump').value,10);
  if(isNaN(v)) return;
  navTo(v-1);
}

// ---- live scrubber ----
// The slider drives navigation in real time. Each input sets the latest target;
// a single consumer loads images sequentially, always jumping to the newest
// target and skipping any it raced past, so dragging never floods the server.
let scrubPending=null, scrubBusy=false;
function scrubTo(v){
  let i=parseInt(v,10); if(isNaN(i)) return;
  i=Math.max(0, Math.min(i, count-1));
  // update the label/jump box instantly so the bar feels live while dragging
  document.getElementById('navpos').textContent=(i+1)+' / '+count;
  document.getElementById('jump').value=i+1;
  scrubPending=i;
  pumpScrub();
}
async function pumpScrub(){
  if(scrubBusy) return;
  scrubBusy=true;
  try{
    while(scrubPending!==null){
      const t=scrubPending; scrubPending=null;
      if(t===idx) continue;
      await saveIfDirty();   // persist edits to the image we're leaving (no-op if untouched)
      await load(t);
    }
  } finally { scrubBusy=false; }
}

/* ---------------- radial class picker (hold C) ---------------- */
const rc = document.getElementById('radial');
const rctx = rc.getContext('2d');
let radial = null;        // {cx, cy, hover}  while the wheel is open
let lastMouse = {x:window.innerWidth/2, y:window.innerHeight/2};
document.addEventListener('mousemove', e=>{
  lastMouse={x:e.clientX, y:e.clientY};
  if(radial){ radial.hover = radialHoverAt(e.clientX, e.clientY); drawRadial(); }
});
function radialClassList(){ return classes.length ? classes : ['class 0']; }
// Wheel geometry adapts to the class count AND label lengths so words stay
// readable: more classes -> bigger ring (thinner-looking wedges but the labels
// don't crowd), longer words -> thicker ring + smaller font. Everything is
// clamped to the viewport so the wheel always fits on screen.
function radialDims(){
  const names=radialClassList();
  const n=names.length;
  const step=2*Math.PI/n;
  const vMin=Math.min(window.innerWidth, window.innerHeight);
  // longest label drives font size + ring thickness
  let maxLen=1; names.forEach(s=>maxLen=Math.max(maxLen, String(s).length));
  const font=Math.max(10, Math.min(16, Math.round(150/Math.max(8,maxLen))+8));
  const thick=Math.max(60, Math.min(170, maxLen*font*0.62 + 30));
  // label ring radius: keep adjacent labels at least ~ (font+10)px apart along
  // the circle, i.e. ring*step >= gap  ->  ring >= gap/step
  const gap=font+11;
  let ring=Math.max(82, gap/step);
  const maxRing=vMin/2 - thick/2 - 14;       // leave room for the ring thickness
  ring=Math.min(ring, Math.max(82, maxRing));
  let inner=Math.max(26, ring - thick/2);
  let outer=ring + thick/2;
  return {n, step, inner, outer, ring, font, thick};
}
function radialHoverAt(mx, my){
  const {n, inner, step} = radialDims();
  const dx=mx-radial.cx, dy=my-radial.cy;
  const dist=Math.hypot(dx,dy);
  if(dist < inner*0.55) return -1;           // dead zone in the middle
  let ang=Math.atan2(dy,dx);                 // -PI..PI, 0 = +x (right)
  // sectors are centred so id 0 sits at the top (-PI/2)
  let a = ang + Math.PI/2;                    // rotate so top = 0
  a = (a % (2*Math.PI) + 2*Math.PI) % (2*Math.PI);
  let i = Math.round(a/step) % n;
  return i;
}
function openRadial(){
  rc.width=window.innerWidth; rc.height=window.innerHeight;
  let cx=lastMouse.x, cy=lastMouse.y;
  const {outer}=radialDims();
  const pad=outer+12;                         // keep the whole wheel on-screen
  cx=Math.max(pad, Math.min(cx, window.innerWidth-pad));
  cy=Math.max(pad, Math.min(cy, window.innerHeight-pad));
  // set radial FIRST — radialHoverAt() reads radial.cx/cy, so it must exist
  radial={cx, cy, hover:-1};
  radial.hover=radialHoverAt(lastMouse.x, lastMouse.y);
  rc.style.display='block';
  drawRadial();
}
function closeRadial(commit){
  if(!radial) return;
  const pick=radial.hover;
  rc.style.display='none';
  rctx.clearRect(0,0,rc.width,rc.height);
  radial=null;
  if(commit && pick>=0){
    setActiveClass(pick);
    setStatus('class → '+className(pick)+' ('+pick+')');
  }
}
// Shrink a label (adding an ellipsis) until it fits maxw px at the current font.
function fitText(g, s, maxw){
  s=String(s);
  if(g.measureText(s).width<=maxw) return s;
  let t=s;
  while(t.length>1 && g.measureText(t+'…').width>maxw) t=t.slice(0,-1);
  return t+'…';
}
function drawRadial(){
  const {n, inner, outer, ring, font, thick}=radialDims();
  const names=radialClassList();
  rctx.clearRect(0,0,rc.width,rc.height);
  rctx.save();
  rctx.fillStyle='rgba(0,0,0,0.35)';
  rctx.fillRect(0,0,rc.width,rc.height);
  const step=2*Math.PI/n;
  const idR=inner+9;                 // radius for the small class id
  const nameLo=idR+9;                // inner bound of the name band
  const nameR=(nameLo+outer)/2;      // centre of the name band
  const nameMaxW=Math.max(20, outer-nameLo-6);
  for(let i=0;i<n;i++){
    const mid=-Math.PI/2 + i*step;            // sector centre angle
    const a0=mid-step/2, a1=mid+step/2;
    const on = i===radial.hover;
    rctx.beginPath();
    rctx.arc(radial.cx, radial.cy, outer, a0, a1);
    rctx.arc(radial.cx, radial.cy, inner, a1, a0, true);
    rctx.closePath();
    const col=classColor(i);
    rctx.fillStyle = on ? col : 'rgba(40,40,40,0.92)';
    rctx.fill();
    rctx.lineWidth = on ? 3 : 1.5;
    rctx.strokeStyle = on ? '#fff' : '#000';
    rctx.stroke();
    // label: rotate it to run along the radius so even long words fit and never
    // overlap their neighbours. Flip text on the left half so it stays upright.
    const flip = Math.cos(mid)<0;
    const sgn = flip ? -1 : 1;
    rctx.save();
    rctx.translate(radial.cx, radial.cy);
    rctx.rotate(mid + (flip?Math.PI:0));
    rctx.textAlign='center'; rctx.textBaseline='middle';
    rctx.fillStyle = on ? '#000' : '#eee';
    rctx.font = (on?'bold ':'')+font+'px system-ui,Arial';
    rctx.fillText(fitText(rctx, names[i], nameMaxW), sgn*nameR, 0);   // class name
    rctx.fillStyle = on ? '#222' : '#9ab';
    rctx.font='10px monospace';
    rctx.fillText(String(i), sgn*idR, 0);            // class id near the inner edge
    rctx.restore();
  }
  // centre hint
  rctx.fillStyle='rgba(20,20,20,0.95)';
  rctx.beginPath(); rctx.arc(radial.cx, radial.cy, inner*0.55, 0, 2*Math.PI); rctx.fill();
  rctx.fillStyle='#bbb'; rctx.font='11px system-ui,Arial';
  rctx.textAlign='center'; rctx.textBaseline='middle';
  rctx.fillText('release C', radial.cx, radial.cy);
  rctx.restore();
}

window.addEventListener('keydown', e=>{
  if(_confirmOpen()){            // confirm dialog captures keys: Enter=OK, Esc=Cancel
    if(e.key==='Enter'){ e.preventDefault(); _confirmResolve(true); }
    else if(e.key==='Escape'){ e.preventDefault(); _confirmResolve(false); }
    return;
  }
  if(e.target.tagName==='INPUT' || e.target.tagName==='SELECT' || e.target.tagName==='TEXTAREA') return;
  if(e.ctrlKey||e.metaKey){                       // undo / redo
    const k=e.key.toLowerCase();
    if(k==='z'){ e.preventDefault(); e.shiftKey?redo():undo(); return; }
    if(k==='y'){ e.preventDefault(); redo(); return; }
  }
  if(e.key==='c'||e.key==='C'){
    if(e.repeat) return;            // keep the wheel open while held
    if(!radial) openRadial();
    e.preventDefault();
    return;
  }
  if(radial){ if(e.key==='Escape') closeRadial(false); return; }   // ignore other keys while wheel is up
  if(/^[0-9]$/.test(e.key)){ setActiveClass(parseInt(e.key,10)); return; }
  if(e.key==='Delete'||e.key==='Backspace'){ e.preventDefault(); delSel(); }
  else if(e.key==='s'||e.key==='S'){ e.preventDefault(); save(); }
  else if(e.key==='d'||e.key==='D'||e.key==='ArrowRight'){ go(1); }
  else if(e.key==='a'||e.key==='A'||e.key==='ArrowLeft'){ go(-1); }
});
window.addEventListener('keyup', e=>{
  if(e.key==='c'||e.key==='C'){ closeRadial(true); e.preventDefault(); }
});
// if focus is lost while holding C, don't leave a stuck wheel
window.addEventListener('blur', ()=>{ if(radial) closeRadial(false); });

window.addEventListener('resize', ()=>{ if(img.complete){ fit(); draw(); } });

(async()=>{
  const m=await fetch('/api/meta').then(r=>r.json());
  count=m.count; classes=m.classes||[]; linkedTask=m.linked_task||null;
  activeClass = m.active_class||0;            // restore the class we were using
  const startIdx = m.last_image||0;           // resume on the exact image
  document.getElementById('folder').value = m.path || '';
  buildClassUI();              // clamps + applies the restored active class
  updateNav();
  applyMode();                 // default to Local (CVAT section hidden until chosen)
  updateThemeIcons();          // set the light/dark toggle icons
  showContinueCard(m);         // offer to resume the last session from the home page
  enhanceSelects(['classsel','cvatproj','aamode','apmodel',
                  'approj','apmode','cvUploadProj','ccproj','clsproj']);  // styled dropdowns
  loadCvatProjects();          // populate the CVAT project dropdown in the background
  if(!count){ setStatus('no images in '+(m.path||'')+' — set a folder above'); return; }
  load(startIdx);
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Wheel-design demo gallery: open /demo, hover each wheel, pick a number.
# --------------------------------------------------------------------------- #
DEMO_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>class-wheel designs</title>
<style>
  html,body{margin:0;background:#161616;color:#ddd;font-family:system-ui,Arial,sans-serif;}
  h1{font-size:18px;padding:14px 18px 4px;margin:0;}
  p.sub{padding:0 18px 12px;margin:0;color:#9aa;font-size:13px;}
  #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
        gap:14px;padding:14px 18px 40px;}
  .card{background:#1f1f1f;border:1px solid #000;border-radius:10px;overflow:hidden;
        display:flex;flex-direction:column;}
  .card h3{margin:0;padding:9px 12px;font-size:13px;background:#272727;
           border-bottom:1px solid #000;display:flex;align-items:center;gap:8px;}
  .card h3 .num{background:#3a6;color:#06210f;font-weight:700;border-radius:5px;
                padding:1px 8px;font-size:13px;}
  .card .desc{padding:6px 12px;color:#8aa;font-size:11px;min-height:30px;}
  .card canvas{display:block;background:#0c0c0c;width:100%;height:330px;cursor:crosshair;}
  .pick{font-size:12px;color:#ffd24d;padding:6px 12px;border-top:1px solid #000;
        min-height:18px;}
</style>
</head>
<body>
<h1>Class-wheel designs &mdash; hover any wheel to feel the selection</h1>
<p class="sub">16 sample classes (mixed lengths) so you can judge crowding. Move the cursor around each wheel; the box below shows what would be picked. <b>Tell me the number you want</b> and I'll wire it into the app.</p>
<div id="grid"></div>
<script>
const CL=["person","helmet","vest","no-helmet","no-vest","forklift","truck",
  "fire-extinguisher","spill","ladder","scaffold","gas-cylinder","hi-vis-vest",
  "welding-mask","confined-space","harness"];
const N=CL.length;
function col(i){ return i<0?'#888':'hsl('+((i*47)%360)+',70%,55%)'; }
function trunc(g,s,max){ if(g.measureText(s).width<=max) return s;
  let t=s; while(t.length>1 && g.measureText(t+'…').width>max) t=t.slice(0,-1);
  return t+'…'; }
function wedge(g,cx,cy,r0,r1,a0,a1){ g.beginPath(); g.arc(cx,cy,r1,a0,a1);
  g.arc(cx,cy,r0,a1,a0,true); g.closePath(); }
function sectorIdx(cx,cy,n,mx,my){
  let a=Math.atan2(my-cy,mx-cx)+Math.PI/2;
  a=((a%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
  return Math.round(a/(2*Math.PI/n))%n;
}
function centerHint(g,cx,cy,r){
  g.fillStyle='rgba(15,15,15,.95)'; g.beginPath(); g.arc(cx,cy,r*0.6,0,7); g.fill();
  g.fillStyle='#999'; g.font='10px system-ui'; g.textAlign='center'; g.textBaseline='middle';
  g.fillText('release C', cx, cy);
}

/* ---------- 1. Radial spokes (labels run along the radius) ---------- */
function d1(g,cx,cy,hv){
  const step=2*Math.PI/N, inner=44, outer=145;
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'rgba(38,38,38,.95)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const fl=Math.cos(mid)<0, s=fl?-1:1;
    g.save(); g.translate(cx,cy); g.rotate(mid+(fl?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle';
    g.fillStyle=on?'#000':'#eee'; g.font=(on?'bold ':'')+'12px system-ui';
    g.fillText(trunc(g,CL[i],outer-inner-12), s*((inner+outer)/2), 0);
    g.restore(); }
  centerHint(g,cx,cy,inner);
}
/* ---------- 2. Curved tangential text along the arc ---------- */
function d2(g,cx,cy,hv){
  const step=2*Math.PI/N, inner=52, outer=145, lr=(inner+outer)/2;
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'rgba(38,38,38,.95)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const bottom=Math.sin(mid)>0;
    g.save(); g.translate(cx,cy); g.rotate(mid+Math.PI/2+(bottom?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle';
    g.fillStyle=on?'#000':'#eee'; g.font=(on?'bold ':'')+'11px system-ui';
    g.fillText(trunc(g,CL[i],lr*step*0.95), 0, bottom?lr:-lr);
    g.restore(); }
  centerHint(g,cx,cy,inner);
}
/* ---------- 3. Callout leader lines, horizontal labels outside ---------- */
function d3(g,cx,cy,hv){
  const step=2*Math.PI/N, inner=34, outer=82;
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'rgba(40,40,40,.95)'; g.fill();
    g.lineWidth=on?2:1; g.strokeStyle=on?'#fff':'#111'; g.stroke(); }
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    const right=Math.cos(mid)>=0;
    const x1=cx+Math.cos(mid)*outer, y1=cy+Math.sin(mid)*outer;
    const x2=cx+Math.cos(mid)*(outer+14), y2=cy+Math.sin(mid)*(outer+14);
    const x3=right?x2+8:x2-8;
    g.strokeStyle=on?col(i):'#555'; g.lineWidth=on?2:1;
    g.beginPath(); g.moveTo(x1,y1); g.lineTo(x2,y2); g.lineTo(x3,y2); g.stroke();
    g.fillStyle=on?col(i):'#cdd'; g.font=(on?'bold ':'')+'11px system-ui';
    g.textAlign=right?'left':'right'; g.textBaseline='middle';
    g.fillText(trunc(g,CL[i],70), right?x3+3:x3-3, y2); }
  centerHint(g,cx,cy,inner);
}
/* ---------- 4. Minimal ring + BIG centre preview of hovered class ---------- */
function d4(g,cx,cy,hv){
  const step=2*Math.PI/N, inner=104, outer=145;
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'hsl('+((i*47)%360)+',45%,30%)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#0c0c0c'; g.stroke();
    g.save(); g.translate(cx,cy); g.rotate(mid+(Math.cos(mid)<0?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle'; g.fillStyle=on?'#000':'#cfd';
    g.font='10px monospace'; g.fillText(String(i), (Math.cos(mid)<0?-1:1)*(inner+outer)/2, 0);
    g.restore(); }
  if(hv>=0){ g.fillStyle=col(hv); g.beginPath(); g.arc(cx,cy-12,16,0,7); g.fill();
    g.fillStyle='#fff'; g.textAlign='center'; g.textBaseline='middle';
    g.font='bold 16px system-ui'; g.fillText(trunc(g,CL[hv],180), cx, cy+22);
    g.fillStyle='#9ab'; g.font='11px monospace'; g.fillText('class '+hv, cx, cy+42);
  } else { g.fillStyle='#777'; g.textAlign='center'; g.textBaseline='middle';
    g.font='13px system-ui'; g.fillText('hover a class', cx, cy); }
}
/* ---------- 5. Pill chips arranged on a ring ---------- */
function d5(g,cx,cy,hv){
  const ringR=112;
  g.strokeStyle='#2a2a2a'; g.lineWidth=1; g.beginPath(); g.arc(cx,cy,ringR,0,7); g.stroke();
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*(2*Math.PI/N), on=i===hv;
    const x=cx+Math.cos(mid)*ringR, y=cy+Math.sin(mid)*ringR;
    g.font=(on?'bold ':'')+'11px system-ui';
    const t=trunc(g,CL[i],90), w=g.measureText(t).width+14, h=on?22:18;
    g.fillStyle=on?col(i):'#2c2c2c'; g.strokeStyle=on?'#fff':'#444'; g.lineWidth=on?2:1;
    const rx=x-w/2, ry=y-h/2, r=h/2;
    g.beginPath(); g.moveTo(rx+r,ry); g.arcTo(rx+w,ry,rx+w,ry+h,r);
    g.arcTo(rx+w,ry+h,rx,ry+h,r); g.arcTo(rx,ry+h,rx,ry,r); g.arcTo(rx,ry,rx+w,ry,r);
    g.closePath(); g.fill(); g.stroke();
    g.fillStyle=on?'#000':'#ddd'; g.textAlign='center'; g.textBaseline='middle';
    g.fillText(t,x,y); }
  centerHint(g,cx,cy,40);
}
/* ---------- 6. Two concentric rings (half the classes each) ---------- */
function d6(g,cx,cy,hv){
  const half=Math.ceil(N/2);
  ring6(g,cx,cy,0,half,44,84,hv);
  ring6(g,cx,cy,half,N,90,140,hv);
}
function ring6(g,cx,cy,a,b,inner,outer,hv){
  const m=b-a, step=2*Math.PI/m;
  for(let k=0;k<m;k++){ const i=a+k, mid=-Math.PI/2+k*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'rgba(38,38,38,.95)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const fl=Math.cos(mid)<0;
    g.save(); g.translate(cx,cy); g.rotate(mid+(fl?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle';
    g.fillStyle=on?'#000':'#eee'; g.font=(on?'bold ':'')+'10px system-ui';
    g.fillText(trunc(g,CL[i],outer-inner-8), (fl?-1:1)*(inner+outer)/2, 0);
    g.restore(); }
}
/* ---------- 7. Fish-eye: hovered wedge swells for room ---------- */
function d7(g,cx,cy,hv){
  const inner=44, outer=145, base=2*Math.PI/N;
  const big=hv>=0?base*2.4:base, rest=hv>=0?(2*Math.PI-big)/(N-1):base;
  const midH=-Math.PI/2+(hv<0?0:hv)*base;
  let a=hv>=0 ? midH-hv*rest-big/2 : -Math.PI/2-base/2;
  for(let i=0;i<N;i++){ const w=(i===hv)?big:rest, a0=a, a1=a+w, mid=(a0+a1)/2, on=i===hv;
    wedge(g,cx,cy,inner,outer,a0,a1);
    g.fillStyle=on?col(i):'rgba(38,38,38,.95)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const fl=Math.cos(mid)<0;
    g.save(); g.translate(cx,cy); g.rotate(mid+(fl?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle';
    g.fillStyle=on?'#000':'#ccc'; g.font=(on?'bold ':'')+(on?'13px':'10px')+' system-ui';
    g.fillText(trunc(g,CL[i],outer-inner-10), (fl?-1:1)*(inner+outer)/2, 0);
    g.restore(); a=a1; }
  centerHint(g,cx,cy,inner);
}
/* ---------- 8. Thin ring + horizontal labels just outside (no leaders) ---------- */
function d8(g,cx,cy,hv){
  const step=2*Math.PI/N, inner=70, outer=92;
  for(let i=0;i<N;i++){ const mid=-Math.PI/2+i*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'hsl('+((i*47)%360)+',50%,34%)'; g.fill();
    g.lineWidth=on?2:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const right=Math.cos(mid)>=0;
    const lx=cx+Math.cos(mid)*(outer+6), ly=cy+Math.sin(mid)*(outer+6);
    g.fillStyle=on?col(i):'#cdd'; g.font=(on?'bold ':'')+'11px system-ui';
    g.textAlign=right?'left':'right'; g.textBaseline='middle';
    g.fillText(trunc(g,CL[i],62), lx, ly); }
  centerHint(g,cx,cy,inner);
}
/* ---------- 9. Half-wheel (top semicircle, roomy) ---------- */
function d9(g,cx,cy,hv){
  const inner=58, outer=150, step=Math.PI/N;
  for(let i=0;i<N;i++){ const mid=Math.PI+(i+0.5)*step, on=i===hv;
    wedge(g,cx,cy,inner,outer,mid-step/2,mid+step/2);
    g.fillStyle=on?col(i):'rgba(38,38,38,.95)'; g.fill();
    g.lineWidth=on?2.5:1; g.strokeStyle=on?'#fff':'#111'; g.stroke();
    const fl=Math.cos(mid)<0;
    g.save(); g.translate(cx,cy); g.rotate(mid+(fl?Math.PI:0));
    g.textAlign='center'; g.textBaseline='middle';
    g.fillStyle=on?'#000':'#eee'; g.font=(on?'bold ':'')+'11px system-ui';
    g.fillText(trunc(g,CL[i],outer-inner-10), (fl?-1:1)*(inner+outer)/2, 0);
    g.restore(); }
  g.fillStyle='#888'; g.textAlign='center'; g.textBaseline='middle'; g.font='11px system-ui';
  g.fillText('cursor enters from below', cx, cy+18);
}
function hv9(cx,cy,mx,my){
  const d=Math.hypot(mx-cx,my-cy); if(d<58||d>150) return -1;
  let a=Math.atan2(my-cy,mx-cx); a=((a%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
  if(a<Math.PI) return -1;                 // bottom half unused
  return Math.min(N-1, Math.floor((a-Math.PI)/(Math.PI/N)));
}
/* ---------- 10. Quick palette grid at the cursor (not a wheel) ---------- */
const G_COLS=2, G_X=18, G_Y=14, G_W=148, G_H=36;
function d10(g,cx,cy,hv){
  for(let i=0;i<N;i++){ const cI=i%G_COLS, rI=(i-cI)/G_COLS, on=i===hv;
    const x=G_X+cI*(G_W+8), y=G_Y+rI*(G_H+4);
    g.fillStyle=on?'#3a3320':'#222'; g.strokeStyle=on?col(i):'#3a3a3a'; g.lineWidth=on?2:1;
    g.beginPath(); g.rect(x,y,G_W,G_H); g.fill(); g.stroke();
    g.fillStyle=col(i); g.beginPath(); g.rect(x+8,y+G_H/2-7,14,14); g.fill();
    g.fillStyle=on?'#fff':'#ddd'; g.font=(on?'bold ':'')+'12px system-ui';
    g.textAlign='left'; g.textBaseline='middle';
    g.fillText(trunc(g,CL[i],G_W-40), x+30, y+G_H/2);
    g.fillStyle='#778'; g.font='9px monospace'; g.textAlign='right';
    g.fillText(String(i), x+G_W-6, y+G_H/2); }
}
function hv10(cx,cy,mx,my){
  for(let i=0;i<N;i++){ const cI=i%G_COLS, rI=(i-cI)/G_COLS;
    const x=G_X+cI*(G_W+8), y=G_Y+rI*(G_H+4);
    if(mx>=x&&mx<=x+G_W&&my>=y&&my<=y+G_H) return i; }
  return -1;
}

const VARIANTS=[
 {n:1,name:'Radial spokes',desc:'Labels run outward along each spoke (your current style, tuned).',draw:d1,cx:165,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<26?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:2,name:'Curved tangential',desc:'Text follows the circle. Classic radial-menu look.',draw:d2,cx:165,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<30?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:3,name:'Callout leaders',desc:'Wedges point out to horizontal labels — most readable for long names.',draw:d3,cx:150,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<20?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:4,name:'Centre preview',desc:'Thin colour ring; the hovered class shows BIG in the middle.',draw:d4,cx:165,cy:165,hv:(cx,cy,mx,my)=>sectorIdx(cx,cy,N,mx,my)},
 {n:5,name:'Pill chips',desc:'Each class is a labelled pill on a ring; hovered pill grows.',draw:d5,cx:165,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<28?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:6,name:'Two rings',desc:'Splits classes across two rings → more room per item.',draw:d6,cx:165,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy),half=Math.ceil(N/2);if(d<44)return -1;if(d<=86)return sectorIdx(cx,cy,half,mx,my);if(d<=142)return half+sectorIdx(cx,cy,N-half,mx,my);return -1;}},
 {n:7,name:'Fish-eye',desc:'Hovered wedge swells so its label is always big.',draw:d7,cx:165,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<26?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:8,name:'Outside labels',desc:'Thin ring, horizontal labels anchored just outside.',draw:d8,cx:155,cy:165,hv:(cx,cy,mx,my)=>{const d=Math.hypot(mx-cx,my-cy);return d<40?-1:sectorIdx(cx,cy,N,mx,my);}},
 {n:9,name:'Half-wheel',desc:'Top semicircle only — double the room per item.',draw:d9,cx:165,cy:225,hv:hv9},
 {n:10,name:'Palette grid',desc:'Not a wheel: a compact colour grid at the cursor. Fastest to scan.',draw:d10,cx:0,cy:0,hv:hv10},
];

const grid=document.getElementById('grid');
VARIANTS.forEach(v=>{
  const card=document.createElement('div'); card.className='card';
  card.innerHTML='<h3><span class="num">'+v.n+'</span>'+v.name+'</h3>'
    +'<div class="desc">'+v.desc+'</div>'
    +'<canvas width="340" height="330"></canvas>'
    +'<div class="pick">move cursor over the wheel…</div>';
  grid.appendChild(card);
  const cvs=card.querySelector('canvas'), g=cvs.getContext('2d');
  const pick=card.querySelector('.pick');
  let hover=-1;
  function render(){ g.clearRect(0,0,cvs.width,cvs.height); v.draw(g,v.cx,v.cy,hover); }
  cvs.addEventListener('mousemove',e=>{
    const r=cvs.getBoundingClientRect();
    const mx=(e.clientX-r.left)*(cvs.width/r.width);
    const my=(e.clientY-r.top)*(cvs.height/r.height);
    const h=v.hv(v.cx,v.cy,mx,my);
    if(h!==hover){ hover=h; render();
      pick.textContent = hover>=0 ? ('would pick → '+CL[hover]+'  (class '+hover+')')
                                  : 'move cursor over the wheel…'; }
  });
  cvs.addEventListener('mouseleave',()=>{ hover=-1; render();
    pick.textContent='move cursor over the wheel…'; });
  render();
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"Images dir : {IMG_DIR}")
    print(f"Labels dir : {LBL_DIR}")
    print(f"Classes    : {CLASSES if CLASSES else '(none — add labels.txt)'}")
    print(f"Found {len(IMAGES)} images.")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
