import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_DIR / "ig-carousel-skia.py"
CONFIG_PATH = PROJECT_DIR / "run_config.json"

st.set_page_config(page_title="IG Carousel (Skia)", layout="wide")

st.title("IG Carousel (Skia)")
st.caption("Manual UI for generating carousels. OpenClaw chat can still run the same script.")

with st.sidebar:
    st.header("Run")
    output_dir = st.text_input(
        "Output directory",
        value=str(Path.home() / "Desktop" / "Test Project" / "Output" / "IG Carousel (Skia)"),
    )
    logo_path = st.text_input(
        "Logo path",
        value=str(Path.home() / ".openclaw" / "workspace" / "assets" / "logos" / "Logo JL.png"),
    )
    logo_pad = st.number_input("Logo pad (px)", min_value=0, max_value=200, value=60, step=1)
    logo_size = st.number_input("Logo size (px)", min_value=20, max_value=400, value=84, step=1)
    run_btn = st.button("Generate carousel", type="primary")

st.subheader("1) Upload photos")
photos = st.file_uploader(
    "Upload 1–30 images (JPG/PNG/WebP). They will be copied into a temp folder for this run.",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png", "webp"],
)

st.subheader("2) Slides (text + mapping to photos)")
st.write("Provide a JSON array of slides. Each slide can reference a photo by filename.")

default_slides = [
    {
        "photo": "(pick)",
        "kicker": "J.League Club Spotlight",
        "title": "MITO HOLLYHOCK",
        "body": "A quick intro for new fans — from Ibaraki, Japan.",
        "centering": [0.78, 0.30],
    },
    {
        "photo": "(pick)",
        "kicker": "Who are they?",
        "title": "HOME TOWN PRIDE",
        "body": "A community-first club — supported by Ibaraki, through every moment.",
        "centering": [0.52, 0.52],
    },
]

slides_json = st.text_area(
    "Slides JSON",
    value=json.dumps(default_slides, indent=2),
    height=360,
)

st.subheader("3) Optional: layout overrides")
st.write("Overrides is a dict keyed by slide number (1-based): {1: [align, anchor, fade], ...}")
overrides_json = st.text_area(
    "Overrides JSON",
    value=json.dumps({1: ["left", "mid", "bottom"], 3: ["right", "bottom", "none"]}, indent=2),
    height=180,
)

st.divider()

log_box = st.empty()


def save_uploads(files, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files:
        name = Path(f.name).name
        data = f.getvalue()
        (out_dir / name).write_bytes(data)
        names.append(name)
    return names


if run_btn:
    if not photos:
        st.error("Upload at least 1 photo.")
        st.stop()

    try:
        slides = json.loads(slides_json)
        overrides = json.loads(overrides_json) if overrides_json.strip() else {}
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    with tempfile.TemporaryDirectory(prefix="ig-carousel-") as td:
        temp_dir = Path(td)
        photos_dir = temp_dir / "photos"
        photo_names = save_uploads(photos, photos_dir)

        # Auto-assign photos for any slide with '(pick)'
        idx = 0
        for s in slides:
            if s.get("photo") in (None, "", "(pick)"):
                s["photo"] = photo_names[idx % len(photo_names)]
                idx += 1

        config = {
            "photos_dir": str(photos_dir),
            "output_dir": output_dir,
            "logo_path": logo_path,
            "spec": {"logo_pad": int(logo_pad), "logo_size": int(logo_size)},
            "slides": slides,
            "overrides": overrides,
        }

        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

        cmd = ["python3", str(SCRIPT_PATH)]
        env = os.environ.copy()
        env["IG_CAROUSEL_CONFIG"] = str(CONFIG_PATH)

        log_box.info("Running renderer…")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            st.error("Render failed")
            st.code(p.stderr or p.stdout)
        else:
            st.success("Done")
            st.code(p.stdout.strip() or "(no output)")

        # Preview outputs
        out_path = Path(output_dir)
        if out_path.exists():
            imgs = sorted([p for p in out_path.glob("*.jpg")])
            if imgs:
                st.subheader("Preview")
                cols = st.columns(5)
                for i, img_path in enumerate(imgs[:20]):
                    with cols[i % 5]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
