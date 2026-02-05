# IG Carousel (Skia)

This project currently has **two ways to run**:

1) **Streamlit web UI** (manual mode)
2) **Direct script execution** (what OpenClaw uses via chat/Telegram)

---

## 1) Streamlit Web UI

```bash
cd /Users/sirasasitorn/.openclaw/workspace/projects/ig-carousel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501

The UI writes a `run_config.json` and runs the renderer with `IG_CAROUSEL_CONFIG` set.

---

## 2) Renderer Script (OpenClaw-compatible)

Renderer:

```bash
python3 /Users/sirasasitorn/.openclaw/workspace/projects/ig-carousel/ig-carousel-skia.py
```

### Config-driven run (same as Streamlit)

```bash
export IG_CAROUSEL_CONFIG=/Users/sirasasitorn/.openclaw/workspace/projects/ig-carousel/run_config.json
python3 /Users/sirasasitorn/.openclaw/workspace/projects/ig-carousel/ig-carousel-skia.py
```

---

## Assets

Shared assets live in:

- `/Users/sirasasitorn/.openclaw/workspace/assets/fonts/`
  - `JLEAGUEKICK-BoldCondensed.otf`
  - `JLEAGUEKICK-BoldExtraCondensed.otf`

- `/Users/sirasasitorn/.openclaw/workspace/assets/logos/`
  - `Logo JL.png`
