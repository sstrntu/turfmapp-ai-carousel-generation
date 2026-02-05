from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from env import load_env
from image_search import download_image
from pipeline import generate_carousel

load_env()

PROJECT_DIR = Path(__file__).resolve().parent

app = FastAPI(title="IG Carousel")
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))

RUNS: dict[str, dict[str, Any]] = {}
RUNS_DIR = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def _run_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "run.json"


def _load_run(run_id: str) -> dict[str, Any] | None:
    path = _run_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_run(run_id: str, data: dict[str, Any]) -> None:
    try:
        path = _run_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to save run {run_id}: {e}")
        raise


def _log(run_id: str, message: str) -> None:
    run = RUNS.get(run_id)
    if not run:
        run = _load_run(run_id)
        if not run:
            return
        RUNS[run_id] = run
    run["logs"].append(message)
    _save_run(run_id, run)


def _set_status(run_id: str, status: str) -> None:
    run = RUNS.get(run_id)
    if not run:
        run = _load_run(run_id)
        if not run:
            return
        RUNS[run_id] = run
    run["status"] = status
    _save_run(run_id, run)


# Load existing runs on server startup
for run_dir in RUNS_DIR.iterdir():
    if run_dir.is_dir():
        run = _load_run(run_dir.name)
        if run:
            RUNS[run_dir.name] = run
            print(f"Loaded existing run: {run_dir.name} (status: {run.get('status', 'unknown')})")


class GenerateRequest(BaseModel):
    intent: str
    allow_external: bool = False
    image_urls: list[str] = []


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "intent": "",
            "allow_external": True,
            "error": None,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    intent: str = Form(...),
    allow_external: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    if not intent.strip():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "intent": intent, "allow_external": True, "error": "Enter an intent."},
        )

    if not files:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "intent": intent, "allow_external": True, "error": "Upload at least one image."},
        )

    allow = allow_external is not None

    run_id = os.urandom(4).hex()
    run_tmp = RUNS_DIR / run_id
    run_tmp.mkdir(parents=True, exist_ok=True)

    uploads: list[str] = []
    for f in files:
        name = Path(f.filename or "upload.jpg").name
        path = run_tmp / name
        data = await f.read()
        path.write_bytes(data)
        uploads.append(str(path))

    RUNS[run_id] = {
        "status": "queued",
        "logs": ["Queued for processing"],
        "result": None,
        "intent": intent,
        "allow_external": allow,
        "output_dir": str(run_tmp),
    }
    _save_run(run_id, RUNS[run_id])

    def _worker():
        try:
            _set_status(run_id, "running")
            _log(run_id, "Starting pipeline")

            # Check if cancelled before starting
            run = RUNS.get(run_id) or _load_run(run_id)
            if run and run["status"] == "cancelled":
                _log(run_id, "Cancelled before processing started")
                return

            # Phase 1: Generate slide plan only (no rendering yet)
            from pipeline import generate_slide_plan

            plan_result = generate_slide_plan(
                intent=intent,
                uploads=uploads,
                allow_external=allow,
                workdir=str(RUNS_DIR),
                run_id=run_id,
                log=lambda m: _log(run_id, m),
            )

            # Save plan and wait for approval
            RUNS[run_id]["plan"] = plan_result
            _save_run(run_id, RUNS[run_id])
            _set_status(run_id, "awaiting_approval")
            _log(run_id, "✓ Slide plan generated - awaiting your approval")

        except Exception as e:
            # Check if it was cancelled during error
            run = RUNS.get(run_id) or _load_run(run_id)
            if run and run["status"] != "cancelled":
                _set_status(run_id, "error")
                _log(run_id, f"Error: {e}")

    threading.Thread(target=_worker, daemon=True).start()
    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_status_page(request: Request, run_id: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # If status is awaiting_approval, show preview page instead
    if run["status"] == "awaiting_approval":
        return RedirectResponse(url=f"/run/{run_id}/preview", status_code=303)

    return templates.TemplateResponse(
        "run.html",
        {
            "request": request,
            "run_id": run_id,
            "status": run["status"],
            "logs": run["logs"],
            "result": run.get("result"),
            "intent": run.get("intent", ""),
            "allow_external": run.get("allow_external", False),
        },
    )


@app.get("/run/{run_id}/preview", response_class=HTMLResponse)
def preview_plan_page(request: Request, run_id: str):
    """Show slide plan preview for user approval"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    plan = run.get("plan")
    if not plan:
        raise HTTPException(status_code=404, detail="plan not found")

    config = plan["config"]
    slides_data = config.get("slides", [])
    photos_dir = Path(config.get("photos_dir", ""))

    # Format slides for template with image URLs
    slides = []
    for slide in slides_data:
        # Check both old format (image object) and new format (photo field)
        image_data = slide.get("image", {})
        photo_filename = slide.get("photo", "") or image_data.get("filename", "")

        # Determine if uploaded or external
        image_source = image_data.get("source", "upload") if image_data else "upload"
        image_query = image_data.get("query", "")

        # Generate image URL for uploaded images
        image_url = None
        if image_source == "upload" and photo_filename:
            if photos_dir.exists():
                # Use photos subdirectory path
                image_url = f"/run/{run_id}/photo/{photo_filename}"

        # Get image reasoning - check both root level and image object
        image_reasoning = slide.get("image_reasoning") or image_data.get("reasoning", "No reasoning provided")

        slides.append({
            "kicker": slide.get("kicker"),
            "title": slide.get("title", ""),
            "body": slide.get("body"),
            "image_source": image_source,
            "image_filename": photo_filename,
            "image_query": image_query,
            "image_url": image_url,
            "image_reasoning": image_reasoning,
        })

    # Check if first slide is cover (no kicker)
    has_cover = len(slides) > 0 and not slides[0].get("kicker")
    content_count = len(slides) - 1 if has_cover else len(slides)

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "run_id": run_id,
            "slides": slides,
            "slide_count": len(slides),
            "has_cover": has_cover,
            "content_count": content_count,
        },
    )


@app.post("/api/generate")
def generate_api(payload: GenerateRequest):
    if not payload.intent.strip():
        raise HTTPException(status_code=400, detail="intent is required")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ig-carousel-url-"))
    uploads: list[str] = []
    try:
        for i, url in enumerate(payload.image_urls):
            ext = os.path.splitext(url)[-1]
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            out = tmp_dir / f"url_{i+1}{ext}"
            if download_image(url, str(out)):
                uploads.append(str(out))

        if not uploads:
            raise HTTPException(status_code=400, detail="No valid images downloaded from image_urls")

        result = generate_carousel(intent=payload.intent, uploads=uploads, allow_external=payload.allow_external)
        RUNS[result["run_id"]] = result

        return JSONResponse(
            {
                "run_id": result["run_id"],
                "outputs": result["outputs"],
                "download_url": f"/download/{result['run_id']}",
            }
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/generate_uploads")
async def generate_api_uploads(
    intent: str = Form(...),
    allow_external: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    allow = allow_external is not None
    tmp_dir = Path(tempfile.mkdtemp(prefix="ig-carousel-upload-"))
    uploads: list[str] = []
    try:
        for f in files:
            name = Path(f.filename or "upload.jpg").name
            path = tmp_dir / name
            data = await f.read()
            path.write_bytes(data)
            uploads.append(str(path))

        result = generate_carousel(intent=intent, uploads=uploads, allow_external=allow)
        RUNS[result["run_id"]] = result

        return JSONResponse(
            {
                "run_id": result["run_id"],
                "outputs": result["outputs"],
                "download_url": f"/download/{result['run_id']}",
            }
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/runs/{run_id}/{filename}")
def get_run_file(run_id: str, filename: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # Check if result exists with proper output_dir
    result = run.get("result")
    if result and "output_dir" in result:
        output_dir = Path(result["output_dir"])
    else:
        # Fallback: try output subdirectory in run directory
        run_dir = Path(run["output_dir"]) if "output_dir" in run else RUNS_DIR / run_id
        output_dir = run_dir / "output"

    path = output_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(path))


@app.get("/run/{run_id}/photo/{filename}")
def get_run_photo(run_id: str, filename: str):
    """Serve photos from the run's photos directory"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # Get photos directory from plan or run directory
    plan = run.get("plan")
    if plan and "config" in plan:
        photos_dir = Path(plan["config"].get("photos_dir", ""))
    else:
        photos_dir = RUNS_DIR / run_id / "photos"

    path = photos_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="photo not found")

    return FileResponse(str(path))


@app.get("/api/status/{run_id}")
def api_status(run_id: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    result = run.get("result")
    return JSONResponse(
        {
            "status": run["status"],
            "logs": run["logs"],
            "outputs": (result or {}).get("outputs", []),
            "download_url": f"/download/{run_id}" if result else None,
        }
    )


@app.get("/download/{run_id}")
def download_zip(run_id: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # Get the correct output directory
    result = run.get("result")
    if result and "output_dir" in result:
        output_dir = Path(result["output_dir"])
    else:
        # Fallback: try output subdirectory in run directory
        run_dir = Path(run["output_dir"]) if "output_dir" in run else RUNS_DIR / run_id
        output_dir = run_dir / "output"

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="output directory not found")

    zip_path = output_dir / f"carousel_{run_id}.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(output_dir.glob("*.jpg")):
                zf.write(p, arcname=p.name)

    return FileResponse(str(zip_path), filename=zip_path.name)


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    if run["status"] in ("done", "error"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel run with status: {run['status']}")

    _set_status(run_id, "cancelled")
    _log(run_id, "Run cancelled by user")

    return JSONResponse({"message": "Run cancelled", "run_id": run_id, "status": "cancelled"})


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # Remove from memory
    if run_id in RUNS:
        del RUNS[run_id]

    # Delete directory
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    return JSONResponse({"message": "Run deleted", "run_id": run_id})


class ApproveRequest(BaseModel):
    edits: list[dict[str, Any]] | None = None


@app.post("/api/approve-plan/{run_id}")
def approve_plan(run_id: str, payload: ApproveRequest = None):
    """User approves the slide plan - continue with rendering"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    if run["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Cannot approve - status is {run['status']}")

    plan = run.get("plan")
    if not plan:
        raise HTTPException(status_code=400, detail="No plan found to approve")

    # Apply text edits if provided
    if payload and payload.edits:
        config = plan["config"]
        slides = config.get("slides", [])

        for edit in payload.edits:
            idx = edit.get("index")
            if idx is not None and 0 <= idx < len(slides):
                if "kicker" in edit:
                    kicker = edit["kicker"]
                    slides[idx]["kicker"] = kicker if kicker and kicker != "Click to add kicker..." else None
                if "title" in edit:
                    slides[idx]["title"] = edit["title"]
                if "body" in edit:
                    body = edit["body"]
                    slides[idx]["body"] = body if body and body != "Click to add body text..." else None

        # Save updated plan
        RUNS[run_id]["plan"] = plan
        _save_run(run_id, RUNS[run_id])
        _log(run_id, "✓ Text edits applied")

    # Continue with rendering in background
    def _render_worker():
        try:
            _set_status(run_id, "rendering")
            _log(run_id, "✓ Plan approved - starting graphics rendering")

            from pipeline import render_from_plan

            result = render_from_plan(
                plan=plan,
                workdir=str(RUNS_DIR),
                run_id=run_id,
                log=lambda m: _log(run_id, m),
            )

            # Check if cancelled
            run = RUNS.get(run_id) or _load_run(run_id)
            if run and run["status"] == "cancelled":
                _log(run_id, "Cancelled during rendering")
                return

            RUNS[run_id]["result"] = result
            _save_run(run_id, RUNS[run_id])
            _set_status(run_id, "done")
            _log(run_id, "✓ Rendering complete")

        except Exception as e:
            run = RUNS.get(run_id) or _load_run(run_id)
            if run and run["status"] != "cancelled":
                _set_status(run_id, "error")
                _log(run_id, f"Error during rendering: {e}")

    threading.Thread(target=_render_worker, daemon=True).start()

    return JSONResponse({"message": "Plan approved, rendering started", "run_id": run_id})


class UpdateSlideRequest(BaseModel):
    positions: list[dict[str, Any]]
    text_updates: dict[str, str] | None = None  # For editing text content


@app.get("/api/slide-config/{run_id}/{slide_num}")
def get_slide_config(run_id: str, slide_num: int):
    """Get text element positions for a specific slide"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    result = run.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="result not found")

    # Load the config file
    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config not found")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    # Find the slide (1-indexed)
    slides = config.get("slides", [])
    if slide_num < 1 or slide_num > len(slides):
        raise HTTPException(status_code=404, detail="slide not found")

    slide = slides[slide_num - 1]

    # Extract text positions
    # The positions are stored in the centering field [x, y] normalized to 0-1
    # Convert to pixel positions (assuming 1080x1350 canvas)
    width = 1080
    height = 1350
    centering = slide.get("centering", [0.5, 0.35])

    response = {
        "title": slide.get("title", ""),
        "title_pos": {
            "x": int(centering[0] * width),
            "y": int(centering[1] * height)
        }
    }

    if slide.get("kicker"):
        response["kicker"] = slide["kicker"]
        # Kicker is above title
        response["kicker_pos"] = {
            "x": int(centering[0] * width),
            "y": int(centering[1] * height - 50)  # Approximate offset
        }

    if slide.get("body"):
        response["body"] = slide["body"]
        # Body is below title
        response["body_pos"] = {
            "x": int(centering[0] * width),
            "y": int(centering[1] * height + 100)  # Approximate offset
        }

    return JSONResponse(response)


@app.post("/api/update-slide/{run_id}/{slide_num}")
def update_slide(run_id: str, slide_num: int, payload: UpdateSlideRequest):
    """Update text positions for a slide and re-render it"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    result = run.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="result not found")

    # Load the config file
    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config not found")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    # Find the slide
    slides = config.get("slides", [])
    if slide_num < 1 or slide_num > len(slides):
        raise HTTPException(status_code=404, detail="slide not found")

    # Update text content if provided
    if payload.text_updates:
        for text_type, new_text in payload.text_updates.items():
            if text_type in ["kicker", "title", "body"]:
                if new_text.strip():
                    slides[slide_num - 1][text_type] = new_text
                elif text_type in slides[slide_num - 1]:
                    # Remove empty text fields
                    del slides[slide_num - 1][text_type]

    # Update positions
    # Convert pixel positions back to normalized 0-1 coordinates
    width = 1080
    height = 1350

    for pos in payload.positions:
        pos_type = pos.get("type")
        x = pos.get("x", 0)
        y = pos.get("y", 0)

        if pos_type == "title":
            # Update centering based on title position
            slides[slide_num - 1]["centering"] = [x / width, y / height]

    # Save updated config
    try:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # Re-render the slide
    try:
        import subprocess

        slide_data = slides[slide_num - 1]
        output_dir = Path(result["output_dir"])
        output_filename = f"Artboard {slide_num}.jpg"
        output_path = output_dir / output_filename

        # Get photos directory from config
        photos_dir = config.get("photos_dir", str(run_dir / "photos"))

        # Create a temp config file for rendering this slide
        temp_config = {
            "photos_dir": photos_dir,  # Directory where original images are
            "output_dir": str(output_dir),  # Where to save output
            "logo_path": config.get("logo_path", "/Users/sirasasitorn/.openclaw/workspace/assets/logos/Logo JL.png"),
            "slides": [{
                "photo": slide_data.get("photo", ""),
                "kicker": slide_data.get("kicker"),
                "title": slide_data.get("title", ""),
                "body": slide_data.get("body"),
                "centering": slide_data.get("centering", [0.5, 0.35])
            }]
        }

        temp_config_path = output_dir / f"temp_config_{slide_num}.json"
        temp_config_path.write_text(json.dumps(temp_config, indent=2), encoding="utf-8")

        # Set environment variable for renderer
        env = os.environ.copy()
        env["IG_CAROUSEL_CONFIG"] = str(temp_config_path)

        # Run the renderer
        render_script = PROJECT_DIR / "ig-carousel-skia.py"
        result = subprocess.run(
            ["python3", str(render_script)],
            env=env,
            check=True,
            capture_output=True,
            text=True
        )

        # Rename output to match expected filename
        rendered_output = output_dir / "Artboard 1.jpg"
        if rendered_output.exists():
            rendered_output.rename(output_path)

        # Clean up temp file
        temp_config_path.unlink(missing_ok=True)

        return JSONResponse({"message": "Slide updated successfully", "slide_num": slide_num})

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Failed to re-render slide: {e}\n{traceback.format_exc()}")
