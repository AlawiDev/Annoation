# main.py
import os
import uuid
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ultralytics YOLO (may require installation of ultralytics package)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ------------------------------
# Path configuration (YOUR layout)
# ------------------------------
BACKEND_DIR = Path(__file__).resolve().parent                 # ANNOATION/annotation-backend
FRONTEND_DIR = BACKEND_DIR.parent / "annotation-frontend"     # ANNOATION/annotation-frontend

UPLOADS_DIR = BACKEND_DIR / "uploads"                         # uploaded videos (backend)
ANNOTATIONS_DIR = BACKEND_DIR / "annotations"                 # annotation JSONs (backend)
PERSONS_DB = BACKEND_DIR / "persons"                          # final confirmed datasets (backend)
TEMP_ROOT = FRONTEND_DIR / "temp"                             # temporary session crops (frontend)
STATIC_DIR = FRONTEND_DIR / "static"                          # frontend static (css/js)
TEMPLATES_DIR = FRONTEND_DIR / "templates"                    # frontend templates

# Ensure folders exist
for p in (UPLOADS_DIR, ANNOTATIONS_DIR, PERSONS_DB, TEMP_ROOT, STATIC_DIR, TEMPLATES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------
# YOLO model
# ------------------------------
MODEL_WEIGHTS = BACKEND_DIR / "yolov8n.pt"   # ensure file exists
if YOLO is None:
    print("WARNING: ultralytics package not available. Install `pip install ultralytics` to enable detection.")
    model = None
else:
    if not MODEL_WEIGHTS.exists():
        print(f"WARNING: Model weights not found at {MODEL_WEIGHTS}. Place yolov8n.pt here for detection.")
    model = YOLO(str(MODEL_WEIGHTS)) if MODEL_WEIGHTS.exists() else None

# ------------------------------
# FastAPI app + mounts
# ------------------------------
app = FastAPI(title="Human-in-the-loop Annotation")

# Serve frontend static, temp session files, and final persons DB
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/temp", StaticFiles(directory=str(TEMP_ROOT)), name="temp")
app.mount("/persons", StaticFiles(directory=str(PERSONS_DB)), name="persons")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=JSONResponse)
def root():
    return {"message": "Annotation backend running", "ui": "/ui"}


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    """Renders the frontend index.html (Jinja2 template)"""
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------
# Helpers
# ------------------------------
def safe_temp_session_path(session_id: str) -> Path:
    path = (TEMP_ROOT / session_id).resolve()
    if not str(path).startswith(str(TEMP_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid session path")
    return path


def list_session_persons(session_id: str) -> Dict[str, List[str]]:
    """Return mapping person_folder -> list of web URL paths (/temp/...) for a given session"""
    session_folder = TEMP_ROOT / session_id
    if not session_folder.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    out: Dict[str, List[str]] = {}
    for person_folder in sorted(session_folder.iterdir()):
        if person_folder.is_dir():
            images = []
            for f in sorted(person_folder.iterdir()):
                if f.is_file():
                    images.append(f"/temp/{session_id}/{person_folder.name}/{f.name}")
            out[person_folder.name] = images
    return out


# ------------------------------
# Upload & process video (creates temp session folder with person_x subfolders)
# ------------------------------
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """
    - Save uploaded video to backend/uploads/<session_id>.mp4
    - Run detection and write crops to frontend/temp/<session_id>/person_<id>/
    - Return JSON with session_id and persons mapping
    """
    # Save uploaded video
    session_id = uuid.uuid4().hex
    upload_path = UPLOADS_DIR / f"{session_id}.mp4"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # Make temp session folder
    session_temp = TEMP_ROOT / session_id
    session_temp.mkdir(parents=True, exist_ok=True)

    # Basic detection parameters (tweak as needed)
    CONF_THRES = 0.35
    IOU_THRES = 0.5
    MIN_BOX_AREA = 25 * 25  # filter tiny boxes
    SAMPLE_EVERY_N_FRAMES = 1  # process every frame (set >1 to skip frames)

    # If YOLO model not available, return an empty session structure (but create temp folder)
    if model is None:
        return JSONResponse({"session_id": session_id, "persons": {}})

    # Try to run tracker if available; fallback to per-frame detection if tracker not supported
    use_tracking = True
    try:
        # try a small track run to check availability (no heavy cost)
        # We'll use model.track streaming directly for the whole video below if supported.
        _ = getattr(model, "track", None)
        if _ is None:
            use_tracking = False
    except Exception:
        use_tracking = False

    person_map: Dict[str, List[str]] = {}

    # Helper to save crop and map to web path
    def save_crop_and_map(person_key: str, crop_img, fname: str):
        folder = session_temp / person_key
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / fname
        cv2.imwrite(str(dst), crop_img)
        web = f"/temp/{session_id}/{person_key}/{fname}"
        person_map.setdefault(person_key, []).append(web)

    try:
        if use_tracking:
            # Use YOLO's track (ByteTrack) - keeps consistent track IDs across frames if supported
            # IMPORTANT: depending on ultralytics version you may need tracker config available.
            stream = model.track(source=str(upload_path),
                                 stream=True,
                                 conf=CONF_THRES,
                                 iou=IOU_THRES,
                                 classes=[0],
                                 tracker="bytetrack.yaml",
                                 persist=True,
                                 verbose=False)
            frame_idx = 0
            for result in stream:
                try:
                    frame = getattr(result, "orig_img", None)
                    if frame is None:
                        frame_idx += 1
                        continue
                    boxes = getattr(result, "boxes", None)
                    if boxes is None:
                        frame_idx += 1
                        continue

                    # extract ids and xyxy safely
                    ids = getattr(boxes, "id", None)
                    xyxy = getattr(boxes, "xyxy", None)

                    # convert to list/ndarray where possible
                    ids_list = None
                    try:
                        ids_list = ids.int().cpu().numpy().tolist() if ids is not None else []
                    except Exception:
                        # maybe it's a list already
                        ids_list = list(ids) if ids is not None else []

                    try:
                        xyxy_np = xyxy.cpu().numpy() if xyxy is not None else None
                    except Exception:
                        xyxy_np = xyxy

                    if xyxy_np is None:
                        frame_idx += 1
                        continue

                    h, w = frame.shape[:2]
                    for i, tid in enumerate(ids_list):
                        try:
                            coords = xyxy_np[i]
                            x1, y1, x2, y2 = map(int, coords.tolist())
                        except Exception:
                            continue
                        # clamp and area filter
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(0, min(x2, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(0, min(y2, h - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        person_key = f"person_{int(tid)}"
                        fname = f"f{frame_idx}_id{int(tid)}.jpg"
                        save_crop_and_map(person_key, crop, fname)
                    frame_idx += 1
                except Exception:
                    # ignore frame-level issues but continue processing
                    frame_idx += 1
                    continue

        else:
            # fallback: per-frame detection without tracking (assign new person ids incrementally)
            cap = cv2.VideoCapture(str(upload_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
            frame_idx = 0
            next_pid = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
                    # write one frame to disk for YOLO convenience
                    tmp_frame_path = session_temp / f"frame_{frame_idx}.jpg"
                    cv2.imwrite(str(tmp_frame_path), frame)
                    results = model(str(tmp_frame_path), conf=CONF_THRES)
                    if len(results) > 0:
                        boxes = getattr(results[0], "boxes", None)
                        if boxes:
                            for r in boxes:
                                try:
                                    if int(r.cls[0].item()) != 0:
                                        continue
                                    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                                except Exception:
                                    continue
                                h, w = frame.shape[:2]
                                x1 = max(0, min(x1, w - 1))
                                x2 = max(0, min(x2, w - 1))
                                y1 = max(0, min(y1, h - 1))
                                y2 = max(0, min(y2, h - 1))
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                                    continue
                                crop = frame[y1:y2, x1:x2]
                                person_key = f"person_{next_pid}"
                                fname = f"f{frame_idx}_p{next_pid}.jpg"
                                save_crop_and_map(person_key, crop, fname)
                                next_pid += 1
                frame_idx += 1
            cap.release()

        # Save session annotation.json inside temp folder
        ann_file = session_temp / "annotation.json"
        with open(ann_file, "w") as f:
            json.dump(person_map, f, indent=2)

        # Also write a backend-side JSON summary (in annotations folder) for record
        summary_path = ANNOTATIONS_DIR / f"{session_id}_summary.json"
        with open(summary_path, "w") as sf:
            json.dump({"session_id": session_id, "persons": person_map}, sf, indent=2)

        return JSONResponse({"session_id": session_id, "persons": person_map})

    except Exception as e:
        # cleanup partial temp on error
        try:
            if session_temp.exists():
                shutil.rmtree(session_temp, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}") from e


# ------------------------------
# Session helpers + human-in-loop endpoints
# ------------------------------
@app.get("/session/{session_id}/persons/")
def get_session_persons(session_id: str):
    return JSONResponse({"session_id": session_id, "persons": list_session_persons(session_id)})


@app.get("/sessions/")
def list_sessions():
    sessions = [p.name for p in TEMP_ROOT.iterdir() if p.is_dir()]
    return JSONResponse({"sessions": sessions})


@app.post("/add_person/")
def add_person(payload: Dict = Body(...)):
    session_id = payload.get("session_id")
    name = payload.get("name", None)
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    session_dir = TEMP_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    # choose name
    if name:
        new_name = name.strip().replace(" ", "_")
    else:
        # auto next index
        existing = [p.name for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("person_")]
        idx = 0
        for e in existing:
            try:
                n = int(e.split("_")[1])
                idx = max(idx, n + 1)
            except Exception:
                pass
        new_name = f"person_{idx}"
    new_folder = session_dir / new_name
    if new_folder.exists():
        raise HTTPException(status_code=400, detail="Person folder exists")
    new_folder.mkdir(parents=True, exist_ok=False)
    return JSONResponse({"person": new_name})


@app.post("/move_image/")
def move_image(payload: Dict = Body(...)):
    src = payload.get("src")
    dest_person = payload.get("dest_person")
    if not src or not dest_person:
        raise HTTPException(status_code=400, detail="src and dest_person required")
    if not src.startswith("/temp/"):
        raise HTTPException(status_code=400, detail="Only /temp/ paths supported")
    parts = src.split("/")
    if len(parts) < 5:
        raise HTTPException(status_code=400, detail="Invalid src path")
    session_id = parts[2]
    filename = parts[-1]
    src_fs = TEMP_ROOT / session_id / parts[3] / filename
    if not src_fs.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    dest_folder_fs = TEMP_ROOT / session_id / dest_person
    dest_folder_fs.mkdir(parents=True, exist_ok=True)
    dest_fs = dest_folder_fs / filename
    if dest_fs.exists():
        base, ext = os.path.splitext(filename)
        dest_fs = dest_folder_fs / f"{base}_{uuid.uuid4().hex[:6]}{ext}"
    shutil.move(str(src_fs), str(dest_fs))
    new_web = f"/temp/{session_id}/{dest_person}/{dest_fs.name}"
    return JSONResponse({"moved_to": new_web})


@app.post("/delete_image/")
def delete_image(payload: Dict = Body(...)):
    path = payload.get("path")
    if not path or not path.startswith("/temp/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    parts = path.split("/")
    if len(parts) < 5:
        raise HTTPException(status_code=400, detail="Invalid path")
    session_id = parts[2]
    filename = parts[-1]
    fs = TEMP_ROOT / session_id / parts[3] / filename
    if not fs.exists():
        raise HTTPException(status_code=404, detail="File not found")
    fs.unlink()
    return JSONResponse({"deleted": path})


@app.post("/rename_person/")
def rename_person(payload: Dict = Body(...)):
    session_id = payload.get("session_id")
    old_name = payload.get("old_name")
    new_name = payload.get("new_name")
    if not (session_id and old_name and new_name):
        raise HTTPException(status_code=400, detail="Missing parameters")
    session_dir = TEMP_ROOT / session_id
    old_fs = session_dir / old_name
    new_fs = session_dir / new_name
    if not old_fs.exists():
        raise HTTPException(status_code=404, detail="Old person not found")
    if new_fs.exists():
        raise HTTPException(status_code=400, detail="Target name exists")
    old_fs.rename(new_fs)
    return JSONResponse({"from": old_name, "to": new_name})


@app.post("/confirm_session/")
def confirm_session(payload: Dict = Body(...)):
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    src = TEMP_ROOT / session_id
    if not src.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    # find next db index inside BACKEND persons folder
    PERSONS_DB.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in PERSONS_DB.iterdir() if d.is_dir()]
    n = 1
    for d in existing:
        if d.startswith("database_"):
            try:
                m = int(d.split("_")[1])
                n = max(n, m + 1)
            except Exception:
                pass
    dest = PERSONS_DB / f"database_{n}"
    shutil.move(str(src), str(dest))
    return JSONResponse({"moved_to": f"/persons/{dest.name}"})
