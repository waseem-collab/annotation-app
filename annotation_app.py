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


def _remembered_folder(default):
    try:
        with open(STATE_FILE) as fh:
            p = fh.read().strip()
        if p and os.path.isdir(os.path.join(p, "images")):
            return p
    except OSError:
        pass
    return default


def _remember_folder(path):
    try:
        with open(STATE_FILE, "w") as fh:
            fh.write(path)
    except OSError:
        pass


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
                with open(p) as fh:
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
        with open(p) as fh:
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
    with open(path, "w") as fh:
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
        with open(os.path.join(DATA, ".cvat_task.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _linked_task_public():
    """Linked-task info without the (potentially large) frame map, for the UI."""
    lt = _linked_task()
    return {k: v for k, v in lt.items() if k != "frames"} if lt else None


@app.route("/api/meta")
def api_meta():
    return jsonify({"count": len(IMAGES), "path": DATA, "classes": CLASSES,
                    "linked_task": _linked_task_public()})


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
                with open(lbl) as fh:
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
                with open(lp) as fh:
                    content = fh.read()
            frame_stem = os.path.splitext(frame)[0]
            z.writestr(f"{folder}/{frame_stem}.txt", content)
            lines.append(f"data/{folder}/{frame}")
        z.writestr(listname, "\n".join(lines) + "\n")


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
            fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_upd_")
            os.close(fd)
            _build_update_zip(lbl_dir, classes, fmap, subset, zip_path)
            _set_job(state="uploading",
                     message=f"updating annotations on task {task_id}…")
            with _cvat_client() as client:
                task = client.tasks.retrieve(int(task_id))
                task.remove_annotations()                 # clean replace
                _cvat_import_annotations(client, int(task_id), zip_path)
            _set_job(state="done", running=False, task_id=int(task_id),
                     task_url=f"{CVAT_URL}/tasks/{task_id}",
                     message=f"updated task {task_id} ✓")
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


@app.route("/api/cvat/projects")
def api_cvat_projects():
    """List CVAT projects (id + name) in the configured org, for the dropdown."""
    try:
        with _cvat_client() as client:
            projects = [{"id": p.id, "name": p.name} for p in client.projects.list()]
        projects.sort(key=lambda p: p["id"])
        return jsonify({"projects": projects, "org": CVAT_ORG, "url": CVAT_URL})
    except Exception as e:
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
        return jsonify({"project_id": int(pid), "classes": names})
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


def _cvat_list_tasks(project_id):
    with _cvat_client() as client:
        out, page = [], 1
        while True:
            data, _ = client.api_client.tasks_api.list(
                project_id=int(project_id), page=page, page_size=100)
            for t in data.results:
                out.append({"id": t.id, "name": t.name, "size": getattr(t, "size", None)})
            if not getattr(data, "next", None):
                break
            page += 1
        return out


@app.route("/api/cvat/tasks")
def api_cvat_tasks():
    pid = request.args.get("project_id")
    if not pid:
        return jsonify({"error": "no project id"}), 400
    try:
        tasks = _cvat_list_tasks(pid)
        tasks.sort(key=lambda t: t["id"])
        return jsonify({"tasks": tasks})
    except Exception as e:
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
                with open(os.path.join(root, "obj.names")) as fh:
                    names = [ln.strip() for ln in fh if ln.strip()]
                break
        if names:
            with open(os.path.join(out_dir, "labels.txt"), "w") as fh:
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


def _cvat_export_task(session, task_id, zip_path):
    """Export a task as 'YOLO 1.1' with images and download the zip."""
    import time
    # location=local is required for app.cvat.ai to populate result_url
    r = session.post(f"{CVAT_URL}/api/tasks/{task_id}/dataset/export",
                     params={"format": "YOLO 1.1", "save_images": "true",
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


def _do_cvat_import(task_id):
    import tempfile
    import shutil
    zip_path = None
    try:
        _set_imp(state="exporting", message="connecting to CVAT…")
        session = _cvat_session()
        info = session.get(f"{CVAT_URL}/api/tasks/{task_id}").json()
        tname = info.get("name") or f"task_{task_id}"
        fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cvat_task_")
        os.close(fd)
        _set_imp(message=f"exporting '{tname}' (images + annotations)…")
        _cvat_export_task(session, task_id, zip_path)
        out_dir = os.path.join(IMPORTS_DIR, f"{task_id}_{_safe_name(tname)}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        _set_imp(message="extracting…")
        n, names, frames, subset = _extract_yolo_export(zip_path, out_dir)
        # remember which CVAT task this folder came from (plus the frame paths /
        # subset) so uploading can update the SAME task's annotations.
        try:
            with open(os.path.join(out_dir, ".cvat_task.json"), "w") as fh:
                json.dump({"task_id": int(task_id), "task_name": tname,
                           "project_id": info.get("project_id"),
                           "frames": frames, "subset": subset}, fh)
        except OSError:
            pass
        _set_imp(running=False, state="done", path=out_dir, count=n,
                 message=f"imported {n} images, {len(names)} classes ✓")
    except Exception as e:
        _set_imp(running=False, state="error", error=str(e),
                 message=f"import failed: {e}")
    finally:
        if zip_path and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


@app.route("/api/cvat/import", methods=["POST"])
def api_cvat_import():
    data = request.get_json(force=True)
    task_id = data.get("task_id")
    if not task_id:
        return jsonify({"error": "no task selected"}), 400
    if not (CVAT_URL and CVAT_USER and CVAT_PASS):
        return jsonify({"error": "CVAT credentials missing in .env"}), 400
    with _imp_lock:
        if _imp_job["running"]:
            return jsonify({"error": "an import is already running"}), 409
        _imp_job.update(running=True, state="starting", message="starting…",
                        path=None, count=0, error=None)
    threading.Thread(target=_do_cvat_import, args=(task_id,), daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/cvat/importstatus")
def api_cvat_importstatus():
    with _imp_lock:
        return jsonify(dict(_imp_job))


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


def _do_autoannotate_all(model_path, images, img_dir, lbl_dir, classes, conf,
                         mode, mapping):
    global CLASSES
    try:
        model = _load_model(model_path)
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        # if the folder has no classes, adopt the model's and persist labels.txt
        if not classes:
            classes = [names[k] for k in sorted(names)]
            data_dir = os.path.dirname(img_dir)
            try:
                with open(os.path.join(data_dir, "labels.txt"), "w") as fh:
                    fh.write("\n".join(classes) + "\n")
                if data_dir == DATA:
                    CLASSES = list(classes)
            except OSError:
                pass
        # mapping: model class index -> here class index. If absent, fall back to
        # matching each detection's class name to the class list.
        use_map = bool(mapping)
        name_to_idx = {str(n).strip().lower(): i for i, n in enumerate(classes)}
        written = skipped = unmapped = added = 0
        for i, img in enumerate(images):
            _set_aa(done=i, message=f"annotating {i+1}/{len(images)}…")
            stem = os.path.splitext(img)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            has_existing = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
            if mode == "skip" and has_existing:
                skipped += 1
                continue
            try:
                dets, _ = _infer(model_path, os.path.join(img_dir, img), conf)
            except Exception:
                continue
            new_boxes = []
            for d in dets:
                if use_map:
                    ci = mapping.get(d.get("cls_model"))
                else:
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


@app.route("/api/autoannotate_all", methods=["POST"])
def api_autoannotate_all():
    data = request.get_json(force=True)
    model_path = _safe_model_path(data.get("model", ""))
    if not model_path:
        return jsonify({"error": "invalid model"}), 400
    if not IMAGES:
        return jsonify({"error": "no images in the current folder"}), 400
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
    raw_map = data.get("mapping") or {}
    mapping = {}
    if isinstance(raw_map, dict):
        for k, v in raw_map.items():
            try:
                mapping[int(k)] = int(v)
            except (TypeError, ValueError):
                pass
    with _aa_lock:
        if _aa_job["running"]:
            return jsonify({"error": "an auto-annotation run is already going"}), 409
        _aa_job.update(running=True, state="starting", done=0, total=len(IMAGES),
                       written=0, skipped=0, unmapped=0,
                       message="loading model…", error=None)
    args = (model_path, list(IMAGES), IMG_DIR, LBL_DIR, list(classes), conf, mode, mapping)
    threading.Thread(target=_do_autoannotate_all, args=args, daemon=True).start()
    return jsonify({"started": True, "count": len(IMAGES)})


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
<style>
  html,body{margin:0;height:100%;background:#1e1e1e;color:#ddd;
            font-family:system-ui,Arial,sans-serif;}
  body{display:flex;flex-direction:column;}
  /* top navigation bar with the image scrubber */
  #topnav{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#2b2b2b;
          border-bottom:1px solid #000;flex:none;}
  #topnav button{background:#3a3a3a;color:#eee;border:1px solid #555;padding:6px 12px;
          border-radius:4px;cursor:pointer;font-size:13px;white-space:nowrap;}
  #topnav button:hover{background:#4a4a4a;}
  #topnav input[type=number]{width:72px;padding:5px;background:#1b1b1b;color:#eee;
          border:1px solid #555;border-radius:4px;}
  #scrub{flex:1;min-width:120px;cursor:pointer;accent-color:#49a05f;height:6px;}
  #navpos{font-family:monospace;font-size:13px;color:#bbb;white-space:nowrap;
          min-width:92px;text-align:center;}
  #meta{display:flex;gap:16px;align-items:center;padding:4px 12px;
        background:#242424;border-bottom:1px solid #000;flex:none;}
  #status{font-size:13px;color:#9c9;}
  #name{font-size:13px;color:#aaa;}
  #activeclass{font-size:13px;color:#ddd;display:flex;align-items:center;gap:6px;}
  #activeclass .sw{width:13px;height:13px;border-radius:3px;display:inline-block;
                   border:1px solid #000;}
  #main{flex:1;display:flex;min-height:0;}
  #wrap{position:relative;flex:1;min-width:0;overflow:auto;display:flex;}
  canvas#cv{background:#000;box-shadow:0 0 20px #000;margin:auto;}
  .dirty{color:#fc6 !important;}
  kbd{background:#444;border-radius:3px;padding:1px 5px;font-size:11px;}
  /* ---- left control sidebar ---- */
  #left{width:230px;flex:none;overflow-y:auto;background:#242424;
        border-right:1px solid #000;}
  .lp-sec{padding:10px 12px;border-bottom:1px solid #191919;}
  .lp-sec h4{margin:0 0 8px;font-size:10px;letter-spacing:.6px;color:#8aa;
             text-transform:uppercase;}
  .tick{display:inline-flex;align-items:center;gap:6px;cursor:pointer;}
  .tick input{accent-color:#49a05f;width:14px;height:14px;margin:0;cursor:pointer;}
  .lp-sec input[type=text],.lp-sec input[type=number]{
     width:100%;box-sizing:border-box;padding:6px;background:#1b1b1b;color:#eee;
     border:1px solid #555;border-radius:5px;font-size:13px;margin-bottom:6px;}
  .lp-row{display:flex;gap:6px;margin-bottom:6px;}
  .lp-row:last-child{margin-bottom:0;}
  .lp-row input[type=number]{width:auto;flex:1;min-width:0;margin-bottom:0;}
  .lp-row button{flex:none;}
  .lp-row button.grow{flex:1;min-width:0;}
  .lp-sec button{background:#3a3a3a;color:#eee;border:1px solid #555;padding:6px 10px;
     border-radius:5px;cursor:pointer;font-size:13px;white-space:nowrap;}
  .lp-sec button:hover{background:#474747;}
  .lp-sec button.wide{display:block;width:100%;margin-bottom:6px;box-sizing:border-box;}
  .lp-sec button.wide:last-child{margin-bottom:0;}
  .lp-sec button.grow{flex:1;}
  .lp-sec button.danger{background:#5a2a2a;border-color:#733;}
  .lp-sec button.danger:hover{background:#7a3030;}
  .lp-sec button.ok{background:#2a4a2a;border-color:#3a6a3a;}
  .lp-sec button.ok:hover{background:#356a35;}
  .lp-sec select{width:100%;box-sizing:border-box;padding:6px;background:#1b1b1b;
     color:#eee;border:1px solid #555;border-radius:5px;font-size:14px;}
  /* per-class visibility rows */
  .collapse-h{cursor:pointer;user-select:none;}
  .collapse-h #viscaret{font-size:9px;color:#8aa;display:inline-block;width:10px;}
  #visiblelist{display:flex;flex-direction:column;gap:2px;max-height:230px;overflow:auto;}
  .vis-row{display:flex;align-items:center;gap:7px;padding:3px 4px;cursor:pointer;
           font-size:12px;border-radius:4px;}
  .vis-row:hover{background:#2e2e2e;}
  .vis-row input{accent-color:#49a05f;width:14px;height:14px;margin:0;cursor:pointer;flex:none;}
  .vis-row .sw{width:12px;height:12px;border-radius:3px;border:1px solid #000;flex:none;}
  .vis-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .lp-toggles{display:flex;flex-direction:column;gap:6px;}
  .toggle{display:inline-flex;align-items:center;gap:6px;padding:6px 11px;
          border-radius:16px;background:#333;border:1px solid #555;cursor:pointer;
          font-size:12px;color:#ccc;user-select:none;white-space:nowrap;}
  .toggle:hover{background:#404040;}
  .toggle input{appearance:none;-webkit-appearance:none;width:0;height:0;margin:0;}
  .toggle:has(input:checked){background:#2f6b3d;border-color:#49a05f;color:#eafff0;}
  .toggle:has(input:checked)::before{content:'\25CF';color:#9af2b5;font-size:9px;}
  .toggle::before{content:'\25CB';color:#888;font-size:9px;}
  /* right boxes panel */
  #panel{width:280px;flex:none;overflow:auto;background:#212121;
         border-left:1px solid #000;font-size:13px;}
  #panel h3{margin:0;padding:9px 12px;background:#2c2c2c;font-size:11px;
            color:#9aa;text-transform:uppercase;letter-spacing:.6px;
            border-bottom:1px solid #000;}
  .row{display:flex;align-items:center;gap:6px;padding:6px 10px;
       border-bottom:1px solid #1a1a1a;cursor:pointer;}
  .row:hover{background:#2e2e2e;}
  .selrow{background:#3a3320;}
  .row .ix{font-family:monospace;width:30px;flex:none;}
  .row select{flex:1;min-width:0;padding:3px;background:#1d1d1d;color:#eee;
             border:1px solid #555;border-radius:3px;font-size:12px;}
  .row .del{margin-left:auto;background:#5a2a2a;color:#eee;border:1px solid #733;
            border-radius:3px;cursor:pointer;padding:2px 8px;flex:none;}
  .row .del:hover{background:#7a3030;}
  .sec{padding:6px 10px;background:#2f2f2f;color:#cc8;font-size:11px;
       text-transform:uppercase;letter-spacing:.5px;}
  #help{font-size:11px;color:#888;padding:4px 12px;background:#252525;flex:none;}
  /* radial class picker overlay (covers viewport, never eats mouse events) */
  #radial{position:fixed;inset:0;z-index:50;display:none;pointer-events:none;}
  #cvatstatus{font-size:11px;color:#9ab;margin-top:6px;word-break:break-word;line-height:1.4;}
  /* CVAT project dropdown should fill the row (beat the 52px filter-select rule) */
  .lp-row #cvatproj{flex:1;width:auto;min-width:0;}
  #cvatproj:disabled{opacity:.75;}
  /* minimal icon-only lock: no button chrome, colour reflects state */
  .lp-sec .lockbtn{flex:none;width:30px;padding:0 2px;background:none;border:none;
                 color:#7a7a7a;cursor:pointer;display:flex;align-items:center;justify-content:center;}
  .lp-sec .lockbtn:hover{background:none;color:#aaa;}
  .lp-sec .lockbtn.on{color:#49a05f;}
  /* automatic-annotation modal + progress */
  .modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:100;
            display:none;align-items:center;justify-content:center;}
  .modal{background:#262626;border:1px solid #000;border-radius:10px;width:340px;
         max-width:90vw;box-shadow:0 12px 48px #000;}
  .modal-h{padding:12px 16px;font-size:14px;font-weight:600;border-bottom:1px solid #000;}
  .modal-body{padding:14px 16px;display:flex;flex-direction:column;gap:6px;}
  .modal-body label{font-size:10px;color:#9aa;text-transform:uppercase;letter-spacing:.5px;}
  #aacfg{display:flex;flex-direction:column;gap:6px;}
  .modal select,.modal input[type=number]{padding:7px;background:#1b1b1b;color:#eee;
         border:1px solid #555;border-radius:5px;font-size:13px;width:100%;box-sizing:border-box;}
  .modal-f{padding:12px 16px;border-top:1px solid #000;display:flex;
           justify-content:flex-end;gap:8px;}
  .modal-f button{padding:7px 16px;border-radius:5px;border:1px solid #555;
                  background:#3a3a3a;color:#eee;cursor:pointer;font-size:13px;}
  .modal-f button.ok{background:#2a4a2a;border-color:#3a6a3a;}
  .modal-f button:disabled{opacity:.5;cursor:default;}
  .bar{height:12px;background:#333;border-radius:6px;overflow:hidden;margin:6px 0 10px;}
  .bar>div{height:100%;width:0;background:#49a05f;transition:width .25s;}
  /* indeterminate (CVAT upload has no % — show a moving stripe) */
  .bar.indet>div{width:35%;transition:none;animation:indet 1.1s ease-in-out infinite;}
  @keyframes indet{0%{margin-left:-35%}100%{margin-left:100%}}
  #aaprogtext,#cvprogtext,#impprogtext{font-size:12px;color:#cde;}
  .aamsg{font-size:11px;color:#e9a;min-height:14px;}
  .maphint{font-size:11px;color:#9aa;margin-bottom:8px;line-height:1.4;}
  #aamaplist{display:flex;flex-direction:column;gap:6px;max-height:320px;overflow:auto;}
  .maprow{display:flex;align-items:center;gap:8px;}
  .maprow .mc{flex:0 0 42%;font-size:12px;color:#cfe;overflow:hidden;
              text-overflow:ellipsis;white-space:nowrap;}
  .maprow .arr{color:#778;}
  .maprow select{flex:1;min-width:0;padding:5px;background:#1b1b1b;color:#eee;
                 border:1px solid #555;border-radius:5px;font-size:12px;}
  .cvtarget{font-size:13px;color:#cde;background:#1b1b1b;border:1px solid #555;
            border-radius:5px;padding:7px;word-break:break-word;}
</style>
</head>
<body>
<div id="topnav">
  <button onclick="go(-1)" title="prev (A)">&#9664; Prev</button>
  <button onclick="go(1)" title="next (D)">Next &#9654;</button>
  <input id="scrub" type="range" min="0" max="0" value="0"
         oninput="scrubTo(this.value)" onchange="scrubTo(this.value)"
         title="drag to scrub through all images">
  <span id="navpos">0 / 0</span>
  <input id="jump" type="number" min="1" placeholder="#"
         onkeydown="if(event.key==='Enter')jump()">
  <button onclick="jump()">Go</button>
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
    </div>
    <div class="lp-sec">
      <h4 class="collapse-h" onclick="toggleVisSec()"><span id="viscaret">&#9654;</span> Visible labels</h4>
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
    <div class="lp-sec">
      <h4>CVAT project</h4>
      <div class="lp-row">
        <select id="cvatproj" onchange="onCvatProjPick()"><option value="">— loading projects… —</option></select>
        <button id="cvatlock" class="lockbtn" title="lock project" onclick="toggleCvatLock()"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-1.5"/></svg></button>
      </div>
      <button class="wide" onclick="loadCvatProjects()">Refresh list</button>
      <button class="wide" onclick="openImpModal()">Import from CVAT…</button>
      <button class="wide ok" onclick="openCvModal()">Upload to CVAT…</button>
      <div id="cvatstatus"></div>
    </div>
  </div>

  <div id="wrap"><canvas id="cv"></canvas></div>

  <div id="panel">
    <div class="lp-sec">
      <h4>Tools</h4>
      <div class="lp-toggles">
        <label class="toggle"><input type="checkbox" id="autosave" checked> &#10515; autosave</label>
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
        <label>Model</label>
        <select id="aamodel"><option value="">— loading models… —</option></select>
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
          <div id="cvtarget" class="cvtarget">— select a project in the sidebar —</div>
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

<div id="impmodal" class="modal-bg">
  <div class="modal">
    <div class="modal-h">Import from CVAT</div>
    <div class="modal-body">
      <div id="impcfg">
        <label>Project</label>
        <select id="impproj" onchange="loadImpTasks()"><option value="">— select project —</option></select>
        <label>Task</label>
        <select id="imptask"><option value="">— select a project first —</option></select>
      </div>
      <div id="impprog" style="display:none;">
        <div class="bar indet" id="impbarwrap"><div id="impbar"></div></div>
        <div id="impprogtext"></div>
      </div>
      <div id="impmsg" class="aamsg"></div>
    </div>
    <div class="modal-f">
      <button id="impcancel" onclick="closeImpModal()">Cancel</button>
      <button id="imprun" class="ok" onclick="runCvatImport()">Import</button>
    </div>
  </div>
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
function classColor(i){
  if(i<0) return '#888';
  const hue = (i*47) % 360;
  return 'hsl('+hue+',70%,55%)';
}
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
  updateNav();
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
  document.getElementById('viscaret').innerHTML = open ? '&#9660;' : '&#9654;';  // ▼ / ▶
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
  document.getElementById('classsel').value=String(i);
  const ac=document.getElementById('activeclass');
  ac.innerHTML='class: <span class="sw" style="background:'+classColor(i)+'"></span> '
    +'<b>'+escapeHtml(className(i))+'</b> ('+i+')';
  // NB: this only sets the class for NEW boxes. The selected box's class is
  // changed only via the right-panel dropdown (setCls).
}
function escapeHtml(s){ return String(s).replace(/[&<>"]/g,c=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

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

async function load(i){
  if(i<0||i>=count) return;
  const r = await fetch('/api/item/'+i+'?t='+Date.now()).then(r=>r.json());
  idx=r.idx; name=r.name; count=r.count; origBoxes=r.boxes;
  // existing labels are always loaded as editable boxes
  boxes = origBoxes.map(b=>({...b}));
  touched=false; sel=-1; markDirty(false);
  updateName();
  document.getElementById('jump').value = idx+1;
  img = new Image();
  img.onload = ()=>{ fit(); draw(); };
  img.src = '/api/image/'+i+'?t='+Date.now();
  setStatus('loaded');
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
    html+='<div style="color:#777;padding:8px 10px;">none &mdash; draw a box</div>';
  } else if(!shownN){
    html+='<div style="color:#777;padding:8px 10px;">all classes hidden</div>';
  } else {
    boxes.forEach((b,i)=>{
      if(hiddenClasses.has(b.cls)) return;
      html+='<div class="row'+(i===sel?' selrow':'')+'" onclick="selectBox('+i+')">'
        +'<span class="ix" style="color:'+classColor(b.cls)+'">#'+(i+1)+'</span>'
        +'<select onclick="event.stopPropagation()" onchange="setCls('+i+',this.value)">'+opts(b.cls)+'</select>'
        +'<button class="del" onclick="event.stopPropagation();removeBox('+i+')">✕</button>'
        +'</div>';
    });
  }
  list.innerHTML=html;
}

function selectBox(i){ sel=i; draw(); }
function setCls(i, v){
  boxes[i].cls = parseInt(v,10) || 0;
  touched=true; markDirty(true); draw(); maybeAutosave();
}
function removeBox(i){
  boxes.splice(i,1);
  if(sel===i) sel=-1; else if(sel>i) sel--;
  touched=true; markDirty(true); draw(); maybeAutosave();
}

// delete the current image (and its label) from disk, then advance
async function deleteImage(){
  if(!count){ setStatus('no image to delete'); return; }
  if(!confirm('Delete image "'+name+'" and its label from disk?\\nThis cannot be undone.')) return;
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
        touched=true; markDirty(true); draw(); maybeAutosave();
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
  activeClass=i; document.getElementById('classsel').value=String(i);
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
  maybeAutosave();
});

function maybeAutosave(){
  if(document.getElementById('autosave').checked && touched) save();
}
function delSel(){
  if(sel<0){ setStatus('no box selected'); return; }
  boxes.splice(sel,1); sel=-1; touched=true; markDirty(true); draw(); maybeAutosave();
}
function clearAll(){
  if(!boxes.length) return;
  if(!confirm('Delete ALL boxes on this image?')) return;
  boxes=[]; sel=-1; touched=true; markDirty(true); draw(); maybeAutosave();
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
  buildClassUI();
  setStatus('loaded '+r.count+' images, '+classes.length+' classes from '+r.path);
  if(count) load(0); else { name=''; updateName(); }
}

// ---- automatic annotation (modal + progress) ----
function setAaMsg(t){ const e=document.getElementById('aamsg'); if(e) e.textContent=t||''; }
function aaProgText(t){ const e=document.getElementById('aaprogtext'); if(e) e.textContent=t||''; }
function setBar(pct){ const e=document.getElementById('aabar');
  if(e) e.style.width=Math.max(0,Math.min(100,pct))+'%'; }
async function loadModels(){
  const sel=document.getElementById('aamodel');
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/models?t='+Date.now()).then(r=>r.json());
    if(!r.models.length){ sel.innerHTML='<option value="">no .pt models found</option>';
      setAaMsg('no models in '+(r.dir||'models/')); return; }
    sel.innerHTML='<option value="">— select model —</option>'
      +r.models.map(m=>'<option value="'+escapeHtml(m.path)+'">'+escapeHtml(m.name)
      +'</option>').join('');
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; setAaMsg('model list failed'); }
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
let aaModelClasses=[];
async function aaNext(){
  const model=document.getElementById('aamodel').value;
  if(!model){ setAaMsg('select a model'); return; }
  setAaMsg('loading model classes…');
  try{
    const r=await fetch('/api/modelclasses?model='+encodeURIComponent(model)
      +'&t='+Date.now()).then(r=>r.json());
    if(r.error){ setAaMsg('error: '+r.error); return; }
    aaModelClasses=r.classes||[];
    if(!aaModelClasses.length){ setAaMsg('model exposes no classes'); return; }
    buildAaMap(); setAaMsg(''); aaShowMap();
  }catch(e){ setAaMsg('failed to load model classes'); }
}
// build the model-class -> here-class mapping rows (auto-match by name)
function buildAaMap(){
  const host=document.getElementById('aamaplist');
  const names = classes.length ? classes : [];
  const lower={}; names.forEach((n,i)=>{ lower[String(n).trim().toLowerCase()]=i; });
  const opts=(selIdx)=>'<option value="">— skip —</option>'
    + names.map((n,i)=>'<option value="'+i+'"'+(i===selIdx?' selected':'')+'>'
        +i+': '+escapeHtml(n)+'</option>').join('');
  host.innerHTML = aaModelClasses.map((mn,mi)=>{
    const m=lower[String(mn).trim().toLowerCase()];
    return '<div class="maprow"><span class="mc" title="'+escapeHtml(mn)+'">'+mi+': '
      +escapeHtml(mn)+'</span><span class="arr">→</span>'
      +'<select data-mi="'+mi+'">'+opts(m===undefined?-1:m)+'</select></div>';
  }).join('');
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
  const model=document.getElementById('aamodel').value;
  const conf=parseFloat(document.getElementById('aaconf').value)||0.25;
  const mode=document.getElementById('aamode').value;
  // gather the model-class -> here-class mapping from the dropdowns
  const mapping={};
  document.querySelectorAll('#aamaplist select').forEach(s=>{
    if(s.value!=='') mapping[s.getAttribute('data-mi')]=parseInt(s.value,10);
  });
  if(!Object.keys(mapping).length){ setAaMsg('map at least one class (or all are set to skip)'); return; }
  setAaMsg(''); showAaProgress();
  setBar(0); aaProgText('starting… ('+count+' images)');
  try{
    const r=await fetch('/api/autoannotate_all',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model, conf, mode, classes, mapping})}).then(r=>r.json());
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
async function loadCvatProjects(){
  const sel=document.getElementById('cvatproj');
  const prev=sel.value;                       // keep current pick across a refresh
  sel.innerHTML='<option value="">loading…</option>';
  setCvatStatus('fetching projects…');
  try{
    const r=await fetch('/api/cvat/projects?t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>';
      setCvatStatus('error: '+r.error); return; }
    if(!r.projects.length){ sel.innerHTML='<option value="">no projects</option>';
      setCvatStatus('no projects in org "'+(r.org||'')+'"'); return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '
      +escapeHtml(p.name)+'</option>').join('');
    if(prev) sel.value=prev;                   // restore selection if still present
    setCvatStatus(r.projects.length+' projects (org "'+(r.org||'')+'")'
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
    classes=r.classes; activeClass=0; buildClassUI(); draw();   // recolour with new classes
    setCvatStatus('using '+classes.length+' classes from CVAT project '+pid);
  }catch(e){ setCvatStatus('request failed'); }
}
function onCvatProjPick(){
  if(document.getElementById('cvatproj').value) fetchCvatClasses();
}
// ---- CVAT upload modal + progress ----
let cvatPoll=null;
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
// updating an imported task hides the "create new" fields
function cvUpdateToggle(){
  const upd = linkedTask && document.getElementById('cvupdate').checked;
  document.getElementById('cvnewwrap').style.display = upd ? 'none' : 'block';
  updateCvRunState();
}
function updateCvRunState(){
  const run=document.getElementById('cvrun');
  if(linkedTask && document.getElementById('cvupdate').checked){ run.disabled=false; cvMsg(''); return; }
  const sel=document.getElementById('cvatproj');
  if(sel.value){ run.disabled=false; cvMsg(''); }
  else { run.disabled=true; cvMsg('pick a CVAT project on the left first'); }
}
async function openCvModal(){
  document.getElementById('cvmodal').style.display='flex';
  cvMsg(''); showCvConfig();
  // linked-task (imported folder) option
  const lw=document.getElementById('cvlinkwrap');
  if(linkedTask){
    lw.style.display='block';
    document.getElementById('cvlinkedname').textContent='#'+linkedTask.task_id
      +(linkedTask.task_name?(' ('+linkedTask.task_name+')'):'');
    document.getElementById('cvupdate').checked=true;
  } else lw.style.display='none';
  // reflect the project chosen in the sidebar (for create-new)
  const sel=document.getElementById('cvatproj');
  const tgt=document.getElementById('cvtarget');
  tgt.textContent = sel.value ? sel.options[sel.selectedIndex].text
                              : '— select a project in the sidebar —';
  cvUpdateToggle();
  // if an upload is already running, jump straight to its progress
  try{
    const s=await fetch('/api/cvat/uploadstatus?t='+Date.now()).then(r=>r.json());
    if(s.running){ showCvProgress(); cvIndet(true); pollCvatUpload(); }
  }catch(e){}
}
function closeCvModal(){ document.getElementById('cvmodal').style.display='none'; }
async function runCvatUpload(){
  const updating = linkedTask && document.getElementById('cvupdate').checked;
  let body;
  if(updating){
    body={task_id:linkedTask.task_id, classes};
  } else {
    const pid=document.getElementById('cvatproj').value;
    const tname=document.getElementById('cvattask').value.trim();
    if(!pid){ cvMsg('select a project in the sidebar'); return; }
    if(!tname){ cvMsg('enter a task name'); return; }
    body={project_id:pid, task_name:tname, classes};
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
      if(s.error) cvProgText('failed: '+s.error);
      else if(s.task_id) cvProgText('done ✓ — task '+s.task_id
        +(s.task_url?' ('+s.task_url+')':''));
    }
  }, 1500);
}

// ---- CVAT import modal + progress ----
let impPoll=null;
function impMsg(t){ const e=document.getElementById('impmsg'); if(e) e.textContent=t||''; }
function impProgText(t){ const e=document.getElementById('impprogtext'); if(e) e.textContent=t||''; }
function impIndet(on){
  const w=document.getElementById('impbarwrap'), b=document.getElementById('impbar');
  if(on){ w.classList.add('indet'); b.style.width=''; b.style.marginLeft=''; }
  else { w.classList.remove('indet'); b.style.marginLeft='0'; b.style.width='100%'; }
}
function showImpConfig(){
  document.getElementById('impcfg').style.display='block';
  document.getElementById('impprog').style.display='none';
  const run=document.getElementById('imprun'); run.style.display=''; run.disabled=false;
  document.getElementById('impcancel').textContent='Cancel';
  impProgText('');
}
function showImpProgress(){
  document.getElementById('impcfg').style.display='none';
  document.getElementById('impprog').style.display='block';
  document.getElementById('imprun').style.display='none';
  document.getElementById('impcancel').textContent='Close';
}
async function openImpModal(){
  document.getElementById('impmodal').style.display='flex';
  impMsg(''); showImpConfig();
  document.getElementById('imptask').innerHTML='<option value="">— select a project first —</option>';
  loadImpProjects();
  // if an import is already running, jump to its progress
  try{
    const s=await fetch('/api/cvat/importstatus?t='+Date.now()).then(r=>r.json());
    if(s.running){ showImpProgress(); impIndet(true); pollCvatImport(); }
  }catch(e){}
}
function closeImpModal(){ document.getElementById('impmodal').style.display='none'; }
async function loadImpProjects(){
  const sel=document.getElementById('impproj');
  sel.innerHTML='<option value="">loading…</option>';
  try{
    const r=await fetch('/api/cvat/projects?t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; impMsg('error: '+r.error); return; }
    sel.innerHTML='<option value="">— select project —</option>'
      +r.projects.map(p=>'<option value="'+p.id+'">'+p.id+' — '+escapeHtml(p.name)+'</option>').join('');
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; impMsg('project list failed'); }
}
async function loadImpTasks(){
  const pid=document.getElementById('impproj').value;
  const sel=document.getElementById('imptask');
  if(!pid){ sel.innerHTML='<option value="">— select a project first —</option>'; return; }
  sel.innerHTML='<option value="">loading tasks…</option>';
  try{
    const r=await fetch('/api/cvat/tasks?project_id='+encodeURIComponent(pid)+'&t='+Date.now()).then(r=>r.json());
    if(r.error){ sel.innerHTML='<option value="">— error —</option>'; impMsg('error: '+r.error); return; }
    if(!r.tasks.length){ sel.innerHTML='<option value="">no tasks in this project</option>'; return; }
    sel.innerHTML='<option value="">— select task —</option>'
      +r.tasks.map(t=>'<option value="'+t.id+'">'+t.id+' — '+escapeHtml(t.name)
        +(t.size!=null?(' ('+t.size+' imgs)'):'')+'</option>').join('');
  }catch(e){ sel.innerHTML='<option value="">— error —</option>'; impMsg('task list failed'); }
}
async function runCvatImport(){
  const task=document.getElementById('imptask').value;
  if(!task){ impMsg('select a task'); return; }
  impMsg(''); showImpProgress(); impIndet(true);
  impProgText('starting import…');
  try{
    const r=await fetch('/api/cvat/import',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:task})}).then(r=>r.json());
    if(r.error){ impProgText('error: '+r.error); impIndet(false); return; }
    pollCvatImport();
  }catch(e){ impProgText('request failed'); impIndet(false); }
}
function pollCvatImport(){
  clearInterval(impPoll);
  impPoll=setInterval(async()=>{
    let s; try{ s=await fetch('/api/cvat/importstatus?t='+Date.now()).then(r=>r.json()); }
    catch(e){ return; }
    impProgText(s.message||s.state||'');
    if(!s.running){
      clearInterval(impPoll);
      impIndet(false);
      document.getElementById('impcancel').textContent='Done';
      if(s.error){ impProgText('failed: '+s.error); return; }
      if(s.path){
        impProgText('imported '+s.count+' images ✓ — opening folder…');
        document.getElementById('folder').value=s.path;
        loadFolder();                                  // switch the app to the imported folder
      }
    }
  }, 1200);
}

async function saveIfDirty(){
  if(touched){
    if(document.getElementById('autosave').checked) await save();
    else if(dirty && confirm('Unsaved changes — save before moving?')) await save();
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
  if(e.target.tagName==='INPUT' || e.target.tagName==='SELECT' || e.target.tagName==='TEXTAREA') return;
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
  document.getElementById('folder').value = m.path || '';
  buildClassUI();
  updateNav();
  loadCvatProjects();          // populate the CVAT project dropdown in the background
  if(!count){ setStatus('no images in '+(m.path||'')+' — set a folder above'); return; }
  load(0);
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
