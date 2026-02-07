from __future__ import annotations

import json
import io
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image, ImageOps

from env import load_env
from image_search import download_image
from pipeline import generate_carousel, _validate_downloaded_image

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
    """Show slide plan preview for user approval (or re-editing after generation)"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    plan = run.get("plan")
    if not plan:
        raise HTTPException(status_code=404, detail="plan not found")

    # Allow preview in both awaiting_approval and done states
    is_rerender = run["status"] == "done"

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

        # Get image description and reasoning - check both root level and image object
        image_description = slide.get("image_description") or image_data.get("description", "No description provided")
        image_reasoning = slide.get("image_reasoning") or image_data.get("reasoning", "No reasoning provided")

        slides.append({
            "kicker": slide.get("kicker"),
            "title": slide.get("title", ""),
            "body": slide.get("body"),
            "image_source": image_source,
            "image_filename": photo_filename,
            "image_query": image_query,
            "image_url": image_url,
            "image_description": image_description,
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
            "is_rerender": is_rerender,
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
    return FileResponse(
        str(path),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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

    return FileResponse(
        str(path),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/editor-bg/{run_id}/{slide_num}")
def get_editor_background(run_id: str, slide_num: int):
    """Serve a 1080x1350 cropped image using the same fit/centering as renderer."""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "run_config.json"
    config = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    if config is None:
        plan = run.get("plan")
        if not plan or "config" not in plan:
            raise HTTPException(status_code=404, detail="config not found")
        config = plan["config"]

    slides = config.get("slides", [])
    if slide_num < 1 or slide_num > len(slides):
        raise HTTPException(status_code=404, detail="slide not found")

    slide = slides[slide_num - 1]
    photo_filename = slide.get("photo")
    if not photo_filename:
        raise HTTPException(status_code=404, detail="slide photo not found")

    photos_dir = Path(config.get("photos_dir", ""))
    photo_path = photos_dir / photo_filename
    if not photo_path.exists():
        raise HTTPException(status_code=404, detail="photo not found")

    centering = slide.get("centering", [0.5, 0.35])
    try:
        cx = float(centering[0])
        cy = float(centering[1])
    except Exception:
        cx, cy = 0.5, 0.35

    try:
        with Image.open(photo_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            fitted = ImageOps.fit(img, (1080, 1350), method=Image.LANCZOS, centering=(cx, cy))
            buf = io.BytesIO()
            fitted.save(buf, format="JPEG", quality=95)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render editor background: {e}")

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
    """User approves the slide plan - continue with rendering (or re-rendering if already done)"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    # Allow approval from awaiting_approval (initial) or done (re-rendering)
    if run["status"] not in ("awaiting_approval", "done"):
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

    # Check if this is a re-render or initial render
    is_rerender = run["status"] == "done"

    # Continue with rendering in background
    def _render_worker():
        try:
            _set_status(run_id, "rendering")
            if is_rerender:
                _log(run_id, "✓ Re-rendering graphics with updated text")
            else:
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


class ChangeImageRequest(BaseModel):
    slide_index: int
    source: str  # "upload" or "external"
    filename: str | None = None  # For upload source - existing file in photos_dir
    query: str | None = None  # For external source - search query


@app.get("/api/available-images/{run_id}")
def get_available_images(run_id: str):
    """Get list of available uploaded images for a run"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    plan = run.get("plan")
    if not plan:
        raise HTTPException(status_code=404, detail="plan not found")

    # Get photos directory
    photos_dir = Path(plan["config"].get("photos_dir", ""))
    if not photos_dir.exists():
        return JSONResponse({"images": []})

    # List all image files
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for img_path in photos_dir.glob(ext):
            images.append({
                "filename": img_path.name,
                "url": f"/run/{run_id}/photo/{img_path.name}"
            })

    return JSONResponse({"images": sorted(images, key=lambda x: x["filename"])})


@app.post("/api/change-image/{run_id}")
def change_slide_image(run_id: str, payload: ChangeImageRequest):
    """Change the image for a specific slide"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    plan = run.get("plan")
    if not plan:
        raise HTTPException(status_code=404, detail="plan not found")

    config = plan["config"]
    slides = config.get("slides", [])

    if payload.slide_index < 0 or payload.slide_index >= len(slides):
        raise HTTPException(status_code=400, detail="Invalid slide index")

    slide = slides[payload.slide_index]
    photos_dir = Path(config.get("photos_dir", ""))

    if payload.source == "upload":
        # Use an existing uploaded image
        if not payload.filename:
            raise HTTPException(status_code=400, detail="Filename required for upload source")

        # Verify file exists
        img_path = photos_dir / payload.filename
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {payload.filename}")

        # Update slide
        slide["photo"] = payload.filename
        slide["image"] = {
            "source": "upload",
            "filename": payload.filename,
            "description": f"User-selected image: {payload.filename}",
            "reasoning": "Manually selected by user"
        }

    elif payload.source == "external":
        # Search and download a new image
        if not payload.query:
            raise HTTPException(status_code=400, detail="Query required for external source")

        try:
            from image_search import search_images, download_image

            # Search for images - returns list of URL strings
            urls = search_images(payload.query, max_results=8)
            if not urls:
                raise HTTPException(status_code=404, detail="No images found for query")

            # Try downloading each URL until one succeeds validation
            downloaded_path = None
            for url in urls:
                if not url:
                    continue

                # Generate filename
                filename = f"external_{payload.slide_index + 1}_{os.urandom(4).hex()}.jpg"
                out_path = photos_dir / filename

                # Download
                if download_image(url, str(out_path)):
                    # Validate (min 800x800)
                    if _validate_downloaded_image(str(out_path)):
                        downloaded_path = out_path
                        break
                    else:
                        out_path.unlink(missing_ok=True)

            if not downloaded_path:
                raise HTTPException(status_code=500, detail="Could not find image large enough (min 800x800). Try a different search query.")

            # Update slide
            slide["photo"] = downloaded_path.name
            slide["image"] = {
                "source": "external",
                "filename": downloaded_path.name,
                "query": payload.query,
                "description": f"External image for: {payload.query}",
                "reasoning": "Downloaded via user search"
            }

        except HTTPException:
            raise
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Image search not available: {e}")
        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=f"Failed to fetch image: {e}")

    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use 'upload' or 'external'")

    # Save updated plan
    RUNS[run_id]["plan"] = plan
    _save_run(run_id, RUNS[run_id])

    # Return updated slide info
    image_url = f"/run/{run_id}/photo/{slide['photo']}" if slide.get("photo") else None

    return JSONResponse({
        "message": "Image changed successfully",
        "slide_index": payload.slide_index,
        "filename": slide.get("photo"),
        "image_url": image_url,
        "source": payload.source
    })


class UpdateSlideRequest(BaseModel):
    positions: list[dict[str, Any]]
    text_updates: dict[str, str] | None = None  # For editing text content
    size_updates: dict[str, float] | None = None  # Per-text size scale (kicker/title/body)


@app.get("/api/slide-config/{run_id}/{slide_num}")
def get_slide_config(run_id: str, slide_num: int):
    """Get text element positions for a specific slide"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "run_config.json"
    config = None

    # Read render config if available (post-render editing).
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    # Fallback to draft plan config (pre-render editing).
    if config is None:
        plan = run.get("plan")
        if not plan or "config" not in plan:
            raise HTTPException(status_code=404, detail="config not found")
        config = plan["config"]

    # Find the slide (1-indexed)
    slides = config.get("slides", [])
    if slide_num < 1 or slide_num > len(slides):
        raise HTTPException(status_code=404, detail="slide not found")

    slide = slides[slide_num - 1]
    photo_filename = slide.get("photo")
    photo_url = f"/run/{run_id}/photo/{photo_filename}" if photo_filename else None

    # Extract text positions
    # Primary source: manual_positions (per-text, normalized 0-1)
    # Fallback: centering plus estimated offsets
    # Convert to pixel positions (assuming 1080x1350 canvas)
    width = 1080
    height = 1350
    centering = slide.get("centering", [0.5, 0.35])
    manual_positions = slide.get("manual_positions") or {}
    manual_scales = slide.get("manual_scales") or {}

    def _manual_pos(text_type: str) -> tuple[float, float] | None:
        pos = manual_positions.get(text_type)
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return None
        try:
            x = float(pos[0])
            y = float(pos[1])
        except Exception:
            return None
        # Treat small-magnitude values as normalized coordinates.
        # This tolerates slight overflow from dragging near the edges.
        if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
            return x * width, y * height
        return x, y

    # Calculate positions - centering is the center point of the text block
    center_x = float(centering[0]) * width
    center_y = float(centering[1]) * height

    # Estimate offsets based on typical text sizes
    # Kicker: ~24px font, positioned above title
    # Title: ~72px font (main element)
    # Body: ~32px font, positioned below title
    has_kicker = bool(slide.get("kicker"))
    has_body = bool(slide.get("body"))

    # Title is at the centering point
    title_y = center_y

    # Adjust title position if there are other elements
    if has_kicker and has_body:
        # Text block has all three - title is in middle
        kicker_offset = -80  # Above title
        body_offset = 100    # Below title
    elif has_kicker:
        # Just kicker and title
        kicker_offset = -60
        body_offset = 0
    elif has_body:
        # Just title and body
        kicker_offset = 0
        body_offset = 80
    else:
        # Just title
        kicker_offset = 0
        body_offset = 0

    manual_title = _manual_pos("title")
    title_x = manual_title[0] if manual_title else center_x
    title_y = manual_title[1] if manual_title else center_y

    response = {
        "title": slide.get("title", ""),
        "photo_url": photo_url,
        "editor_bg_url": f"/api/editor-bg/{run_id}/{slide_num}",
        "centering": centering,
        "scales": {
            "kicker": float(manual_scales.get("kicker", 1.0)),
            "title": float(manual_scales.get("title", 1.0)),
            "body": float(manual_scales.get("body", 1.0)),
        },
        "title_pos": {
            "x": title_x,
            "y": title_y
        }
    }

    if has_kicker:
        manual_kicker = _manual_pos("kicker")
        response["kicker"] = slide["kicker"]
        response["kicker_pos"] = {
            "x": manual_kicker[0] if manual_kicker else title_x,
            "y": manual_kicker[1] if manual_kicker else title_y + kicker_offset
        }

    if has_body:
        manual_body = _manual_pos("body")
        response["body"] = slide["body"]
        response["body_pos"] = {
            "x": manual_body[0] if manual_body else title_x,
            "y": manual_body[1] if manual_body else title_y + body_offset
        }

    return JSONResponse(response)


@app.post("/api/update-slide/{run_id}/{slide_num}")
def update_slide(run_id: str, slide_num: int, payload: UpdateSlideRequest):
    """Update text positions for a slide and re-render it"""
    run = RUNS.get(run_id) or _load_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "run_config.json"
    has_render_config = config_path.exists()
    result = run.get("result")

    # Use render config when present; otherwise update draft plan config.
    if has_render_config:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")
    else:
        plan = run.get("plan")
        if not plan or "config" not in plan:
            raise HTTPException(status_code=404, detail="config not found")
        config = plan["config"]

    # Find the slide
    slides = config.get("slides", [])
    if slide_num < 1 or slide_num > len(slides):
        raise HTTPException(status_code=404, detail="slide not found")

    slide_data = slides[slide_num - 1]

    # Update text content if provided
    if payload.text_updates:
        for text_type, new_text in payload.text_updates.items():
            if text_type in ["kicker", "title", "body"]:
                if new_text.strip():
                    slide_data[text_type] = new_text
                elif text_type in slide_data:
                    # Remove empty text fields
                    del slide_data[text_type]

    # Update positions
    # - Persist per-text manual_positions for independent editing
    # - Keep title as centering for backward compatibility
    # Convert pixel positions back to normalized 0-1 coordinates
    width = 1080
    height = 1350
    manual_positions = slide_data.get("manual_positions")
    if not isinstance(manual_positions, dict):
        manual_positions = {}
    manual_scales = slide_data.get("manual_scales")
    if not isinstance(manual_scales, dict):
        manual_scales = {}

    for pos in payload.positions:
        pos_type = pos.get("type")
        x = pos.get("x", 0)
        y = pos.get("y", 0)
        if pos_type not in ("kicker", "title", "body"):
            continue
        nx = x / width
        ny = y / height
        manual_positions[pos_type] = [nx, ny]

        if pos_type == "title":
            # Title position becomes the centering point
            slide_data["centering"] = [nx, ny]

    if manual_positions:
        slide_data["manual_positions"] = manual_positions
    elif "manual_positions" in slide_data:
        del slide_data["manual_positions"]

    if payload.size_updates:
        for text_type, scale in payload.size_updates.items():
            if text_type not in ("kicker", "title", "body"):
                continue
            try:
                manual_scales[text_type] = float(scale)
            except Exception:
                continue
    if manual_scales:
        slide_data["manual_scales"] = manual_scales
    elif "manual_scales" in slide_data:
        del slide_data["manual_scales"]

    # Save run config only when it exists (post-render mode).
    if has_render_config:
        try:
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # Keep draft plan config in sync for preview and re-render flow.
    plan = run.get("plan")
    if plan and "config" in plan:
        plan_slides = plan["config"].get("slides", [])
        if 1 <= slide_num <= len(plan_slides):
            plan_slide = plan_slides[slide_num - 1]
            plan_slide["centering"] = slide_data.get("centering", [0.5, 0.35])
            for key in ("kicker", "title", "body"):
                if key in slide_data:
                    plan_slide[key] = slide_data[key]
                elif key in plan_slide:
                    del plan_slide[key]
            if "manual_positions" in slide_data:
                plan_slide["manual_positions"] = slide_data["manual_positions"]
            elif "manual_positions" in plan_slide:
                del plan_slide["manual_positions"]
            if "manual_scales" in slide_data:
                plan_slide["manual_scales"] = slide_data["manual_scales"]
            elif "manual_scales" in plan_slide:
                del plan_slide["manual_scales"]
        run["plan"] = plan
        RUNS[run_id] = run
        _save_run(run_id, run)

    # Draft mode: no rendered outputs yet, so only persist updates.
    if not (result and has_render_config):
        return JSONResponse({"message": "Slide draft updated successfully", "slide_num": slide_num})

    # Re-render the slide
    temp_render_dir: Path | None = None
    temp_config_path: Path | None = None
    try:
        import subprocess

        output_dir = Path(result["output_dir"])
        output_filename = f"Artboard {slide_num}.jpg"
        output_path = output_dir / output_filename

        # Get paths from config
        photos_dir = config.get("photos_dir", str(run_dir / "photos"))
        temp_render_dir = run_dir / f".single_render_tmp_{slide_num}_{os.urandom(4).hex()}"
        temp_render_dir.mkdir(parents=True, exist_ok=True)

        # Create a temp config file for rendering this single slide
        # Include all necessary fields from the original slide config
        temp_config = {
            "photos_dir": photos_dir,
            "output_dir": str(temp_render_dir),
            "logo_path": config.get("logo_path", ""),
            "slides": [{
                "photo": slide_data.get("photo", ""),
                "kicker": slide_data.get("kicker"),
                "title": slide_data.get("title", ""),
                "body": slide_data.get("body"),
                "centering": slide_data.get("centering", [0.5, 0.35]),
                "manual_positions": slide_data.get("manual_positions"),
                "manual_scales": slide_data.get("manual_scales"),
                # Include override if present (align, anchor, fade)
                "override": slide_data.get("override"),
                # Include subject_region for split layout decisions
                "subject_region": slide_data.get("subject_region"),
            }]
        }

        temp_config_path = run_dir / f"temp_config_{slide_num}_{os.urandom(4).hex()}.json"
        temp_config_path.write_text(json.dumps(temp_config, indent=2), encoding="utf-8")

        # Set environment variable for renderer
        env = os.environ.copy()
        env["IG_CAROUSEL_CONFIG"] = str(temp_config_path)

        # Run the renderer
        render_script = PROJECT_DIR / "ig-carousel-skia.py"
        proc_result = subprocess.run(
            ["python3", str(render_script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )

        if proc_result.returncode != 0:
            raise Exception(f"Renderer failed: {proc_result.stderr}")

        # Move isolated render output into the target artboard path.
        rendered_output = temp_render_dir / "Artboard 1.jpg"
        if not rendered_output.exists():
            raise Exception("Renderer completed but did not produce Artboard 1.jpg")

        # Replace the target artboard atomically.
        if output_path.exists():
            output_path.unlink()
        rendered_output.replace(output_path)

        return JSONResponse({"message": "Slide updated successfully", "slide_num": slide_num})

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Rendering timed out")
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Failed to re-render slide: {e}")
    finally:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        if temp_render_dir is not None:
            shutil.rmtree(temp_render_dir, ignore_errors=True)
