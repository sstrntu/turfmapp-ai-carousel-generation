from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import skia
import numpy as np
from PIL import Image, ImageOps


def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    # Replace common unicode punctuation with ASCII equivalents.
    # NOTE: This reduces the chance of missing glyphs, but we also implement font fallback.
    return (
        s.replace(""", '"')
        .replace(""", '"')
        .replace("'", "'")
        .replace("'", "'")
        .replace("—", "-")  # em dash
        .replace("–", "-")  # en dash
        .replace("−", "-")  # minus sign
        .replace("‐", "-")  # hyphen
        .replace("‑", "-")  # non-breaking hyphen
        .replace("…", "...")  # ellipsis
        .replace("•", "*")  # bullet
        .replace("·", "*")  # middle dot
        .replace("×", "x")  # multiplication sign
        .replace("÷", "/")  # division sign
        .replace("°", "deg")  # degree sign
        .replace("™", "(TM)")  # trademark
        .replace("®", "(R)")  # registered
        .replace("©", "(C)")  # copyright
        .replace("\u00a0", " ")  # non-breaking space
        .replace("\u2009", " ")  # thin space
        .replace("\u200b", "")  # zero-width space
        .replace("\ufeff", "")  # zero-width no-break space
    )


@dataclass
class SlideText:
    kicker: Optional[str]
    title: str
    body: Optional[str]
    split_layout: bool = False  # If True, separate body from title

    def normalized(self) -> "SlideText":
        return SlideText(
            normalize_text(self.kicker),
            normalize_text(self.title) or "",
            normalize_text(self.body),
            self.split_layout,
        )


@dataclass
class Spec:
    W: int = 1080
    H: int = 1350
    margin_l: int = 60  # Reduced for more space
    margin_r: int = 60  # Reduced for more space
    margin_t: int = 100  # Increased to avoid covering faces
    margin_b: int = 120  # Increased to ensure text stays in safe zone
    gap: int = 14

    # Type sizes - GRAND and BOLD like examples
    kicker_size: int = 48  # Increased for better visibility
    title_size: int = 120  # MUCH LARGER for grand impact
    body_size: int = 34
    slide_num_size: int = 24

    # Gradient overlay
    grad_h_ratio: float = 0.42
    grad_alpha_bottom: int = 224  # 0-255

    # Logo (brand mark) - bigger and clearly readable
    # Note: truly "5x" larger than 160px would be ~800px (that becomes a hero badge/watermark).
    logo_size: int = 84  # 5x smaller than 420px
    logo_pad: int = 60  # fixed lockup: 60px from top + 60px from left


# -------------------- Font + Layout helpers --------------------

def cover_fit_pil(pil: Image.Image, w: int, h: int, centering=(0.5, 0.35)) -> Image.Image:
    return ImageOps.fit(pil, (w, h), method=Image.LANCZOS, centering=centering)


def skia_typefaces() -> tuple[skia.Typeface, skia.Typeface, skia.Typeface]:
    """Return (title_bold, kicker_bold, body_font).

    Prefer local brand fonts if present in workspace assets.
    Fallback to system fonts for special characters.
    """
    fonts_dir = "/Users/sirasasitorn/.openclaw/workspace/assets/fonts"
    title_path = os.path.join(fonts_dir, "JLEAGUEKICK-BoldExtraCondensed.otf")
    kicker_path = os.path.join(fonts_dir, "JLEAGUEKICK-BoldCondensed.otf")

    title_tf = skia.Typeface.MakeFromFile(title_path) if os.path.exists(title_path) else None
    kicker_tf = skia.Typeface.MakeFromFile(kicker_path) if os.path.exists(kicker_path) else None

    fm = skia.FontMgr.RefDefault()

    # Try multiple fallback fonts for best coverage of special characters
    # Priority: Helvetica Neue → Arial → Helvetica → System Default
    body_tf = None
    for font_name in ["Helvetica Neue", "Arial", "Helvetica"]:
        body_tf = fm.matchFamilyStyle(font_name, skia.FontStyle.Normal())
        if body_tf is not None:
            break

    if body_tf is None:
        body_tf = skia.Typeface.MakeDefault()

    # If JLEAGUEKICK fonts not found, use fallback
    if title_tf is None:
        for font_name in ["Helvetica Neue", "Arial", "Helvetica"]:
            title_tf = fm.matchFamilyStyle(font_name, skia.FontStyle.Bold())
            if title_tf is not None:
                break
        if title_tf is None:
            title_tf = skia.Typeface.MakeDefault()

    if kicker_tf is None:
        for font_name in ["Helvetica Neue", "Arial", "Helvetica"]:
            kicker_tf = fm.matchFamilyStyle(font_name, skia.FontStyle.Bold())
            if kicker_tf is not None:
                break
        if kicker_tf is None:
            kicker_tf = skia.Typeface.MakeDefault()

    return title_tf, kicker_tf, body_tf


def draw_string_with_fallback(
    canvas: skia.Canvas,
    text: str,
    x: float,
    y: float,
    font_primary: skia.Font,
    font_fallback: skia.Font,
    paint: skia.Paint,
    shadow_paint: skia.Paint | None = None,
    shadow_dx: float = 0.0,
    shadow_dy: float = 0.0,
) -> float:
    """Draw text using primary font, but switch to fallback for missing glyphs.

    Multi-level fallback:
    1. Primary font (JLEAGUEKICK)
    2. Fallback font (Helvetica/Arial)
    3. System default (guaranteed to have all glyphs)

    Returns the advance width.
    """
    if not text:
        return 0.0

    tfp = font_primary.getTypeface()
    tff = font_fallback.getTypeface()
    # Third-level fallback: system default font for any remaining special chars
    tf_default = skia.Typeface.MakeDefault()
    font_default = skia.Font(tf_default, font_primary.getSize())
    font_default.setEdging(font_primary.getEdging())

    def glyph_ok(tf: skia.Typeface, ch: str) -> bool:
        """Check if typeface has glyph for character."""
        try:
            glyph_id = tf.unicharToGlyph(ord(ch))
            # 0 usually indicates missing glyph, but also check if it's a valid glyph
            return glyph_id != 0
        except Exception:
            return False

    def choose_font_for_char(ch: str) -> skia.Font:
        """Choose the best font for this character."""
        if glyph_ok(tfp, ch):
            return font_primary
        elif glyph_ok(tff, ch):
            return font_fallback
        else:
            # Use system default as last resort
            return font_default

    cur_font = None
    run = ""
    advance = 0.0

    def flush(run_text: str, fnt: skia.Font, xx: float) -> float:
        if not run_text:
            return 0.0
        if shadow_paint is not None:
            canvas.drawString(run_text, xx + shadow_dx, y + shadow_dy, fnt, shadow_paint)
        canvas.drawString(run_text, xx, y, fnt, paint)
        return fnt.measureText(run_text)

    xx = x
    for ch in text:
        next_font = choose_font_for_char(ch)

        if cur_font is None:
            cur_font = next_font

        if next_font is not cur_font:
            # Flush current run with current font
            w = flush(run, cur_font, xx)
            xx += w
            advance += w
            run = ""
            cur_font = next_font

        run += ch

    # Flush remaining text
    if run:
        w = flush(run, cur_font if cur_font else font_primary, xx)
        advance += w

    return advance


def wrap_to_width(text: str, font: skia.Font, max_w: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []

    def width(s: str) -> float:
        return font.measureText(s)

    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if not cur:
            cur = [w]
            continue
        if width(trial) <= max_w:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def text_block_metrics(spec: Spec, slide: SlideText) -> tuple[float, float, dict]:
    """Return (block_w, block_h, ctx) based on wrapping with current type."""
    title_tf, kicker_tf, body_tf = skia_typefaces()
    font_k = skia.Font(kicker_tf, spec.kicker_size)
    font_t = skia.Font(title_tf, spec.title_size)
    font_b = skia.Font(body_tf, spec.body_size)
    for f in (font_k, font_t, font_b):
        f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)

    max_w = spec.W - spec.margin_l - spec.margin_r
    kicker_lines = wrap_to_width(slide.kicker, font_k, max_w) if slide.kicker else []
    title_lines = wrap_to_width(slide.title, font_t, max_w)
    body_lines = wrap_to_width(slide.body, font_b, max_w) if slide.body else []

    mk = font_k.getMetrics(); hk = mk.fDescent - mk.fAscent
    mt = font_t.getMetrics(); ht = mt.fDescent - mt.fAscent
    mb = font_b.getMetrics(); hb = mb.fDescent - mb.fAscent

    # height
    block_h = 0
    if kicker_lines:
        block_h += len(kicker_lines) * (hk + 6) + spec.gap
    block_h += len(title_lines) * (ht + 2)
    if body_lines:
        block_h += spec.gap + len(body_lines) * (hb + 8)

    # width (max line width)
    widths = []
    for line in kicker_lines:
        widths.append(font_k.measureText(line))
    for line in title_lines:
        widths.append(font_t.measureText(line))
    for line in body_lines:
        widths.append(font_b.measureText(line))
    block_w = min(max(widths) if widths else max_w, max_w)

    ctx = {
        "font_k": font_k, "font_t": font_t, "font_b": font_b,
        "kicker_lines": kicker_lines, "title_lines": title_lines, "body_lines": body_lines,
        "mk": mk, "mt": mt, "mb": mb,
        "hk": hk, "ht": ht, "hb": hb,
        "max_w": max_w,
    }
    return block_w, block_h, ctx


# -------------------- Image conversion --------------------

def pil_to_skia_image(path: str, w: int, h: int, centering=(0.5, 0.35)) -> tuple[skia.Image, Image.Image]:
    pil = Image.open(path).convert("RGBA")
    pil = cover_fit_pil(pil, w, h, centering=centering)
    img = skia.Image.frombytes(pil.tobytes(), (w, h), skia.ColorType.kRGBA_8888_ColorType)
    return img, pil


def load_logo(path: str, target_px: int) -> skia.Image:
    pil = Image.open(path).convert("RGBA")

    # If the source logo has a white *background* (common), remove only the background
    # (connected to the outer border), while keeping internal white details.
    arr = np.array(pil, dtype=np.uint8, copy=True)
    rgb = arr[..., :3]
    a = arr[..., 3].copy()

    near_white = (rgb[..., 0] > 245) & (rgb[..., 1] > 245) & (rgb[..., 2] > 245) & (a > 0)

    h, w = near_white.shape
    bg = np.zeros((h, w), dtype=bool)

    # Flood fill from borders on the near-white mask
    stack = []
    for x in range(w):
        if near_white[0, x]: stack.append((0, x))
        if near_white[h - 1, x]: stack.append((h - 1, x))
    for y in range(h):
        if near_white[y, 0]: stack.append((y, 0))
        if near_white[y, w - 1]: stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        if bg[y, x]:
            continue
        if not near_white[y, x]:
            continue
        bg[y, x] = True
        if y > 0: stack.append((y - 1, x))
        if y + 1 < h: stack.append((y + 1, x))
        if x > 0: stack.append((y, x - 1))
        if x + 1 < w: stack.append((y, x + 1))

    # Make only the detected background transparent
    a[bg] = 0
    arr[..., 3] = a
    pil = Image.fromarray(arr)

    # Crop to the non-transparent content first (removes unnecessary padding)
    alpha = pil.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        pil = pil.crop(bbox)

    pil = ImageOps.contain(pil, (target_px, target_px), method=Image.LANCZOS)

    # IMPORTANT: align to TOP-LEFT of the logo box (not centered)
    canvas = Image.new("RGBA", (target_px, target_px), (0, 0, 0, 0))
    canvas.paste(pil, (0, 0), pil)
    return skia.Image.frombytes(canvas.tobytes(), (target_px, target_px), skia.ColorType.kRGBA_8888_ColorType)


# -------------------- Frame analysis (best text placement) --------------------

def edge_energy(gray: np.ndarray) -> np.ndarray:
    """Simple Sobel magnitude (no OpenCV)."""
    # pad
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.abs(gx) + np.abs(gy)
    return mag


def score_box(pil_rgba: Image.Image, box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    crop = pil_rgba.crop((x0, y0, x1, y1)).convert("RGB")
    arr = np.asarray(crop, dtype=np.uint8)
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    mean = float(gray.mean())
    var = float(gray.var())
    edges = edge_energy(gray)
    e = float(edges.mean())

    # Prefer: low edges (clean space), low variance (less noisy), darker (better for white text)
    bright_penalty = max(0.0, (mean - 145.0) / 50.0)  # starts penalizing above ~145
    return e * 1.0 + (var ** 0.5) * 0.12 + bright_penalty * 25.0


def rect_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1-ax0)*max(0, ay1-ay0)
    area_b = max(0, bx1-bx0)*max(0, by1-by0)
    return inter / float(area_a + area_b - inter + 1e-6)


def expand_rect(r: Tuple[int,int,int,int], pad: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x0,y0,x1,y1 = r
    return (max(0, x0-pad), max(0, y0-pad), min(W, x1+pad), min(H, y1+pad))


def subject_box(spec: Spec, pil_rgba: Image.Image) -> Tuple[int,int,int,int]:
    """Approximate subject region by saliency (edges+contrast), return box in full-res coords."""
    rgb = np.asarray(pil_rgba.convert('RGB'), dtype=np.float32)
    gray = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
    # downsample for speed
    ds = 6
    g = gray[::ds, ::ds]
    e = edge_energy(g)
    # center bias (subjects tend to be near center)
    h, w = e.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w-1)/2.0, (h-1)/2.0
    dist = ((xx-cx)**2 + (yy-cy)**2)**0.5
    dist = dist / (dist.max() + 1e-6)
    score = e * (1.3 - 0.9*dist)

    # threshold top percentile
    t = np.percentile(score, 92)
    mask = score >= t
    if mask.sum() < 50:
        # fallback: central box
        return (int(spec.W*0.2), int(spec.H*0.18), int(spec.W*0.8), int(spec.H*0.82))

    ys, xs = np.where(mask)
    x0, x1 = xs.min()*ds, (xs.max()+1)*ds
    y0, y1 = ys.min()*ds, (ys.max()+1)*ds

    # expand a bit
    pad_x = int((x1-x0)*0.10)
    pad_y = int((y1-y0)*0.10)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(spec.W, x1 + pad_x)
    y1 = min(spec.H, y1 + pad_y)
    return (int(x0), int(y0), int(x1), int(y1))


def logo_boxes(spec: Spec) -> dict[str, Tuple[int,int,int,int]]:
    s = spec.logo_size
    p = spec.logo_pad
    return {
        'tl': (p, p, p+s, p+s),
        'tr': (spec.W - p - s, p, spec.W - p, p+s),
        'br': (spec.W - p - s, spec.H - p - s, spec.W - p, spec.H - p),
    }


def best_layout_for_slide(spec: Spec, pil_rgba: Image.Image, slide: SlideText, logo_corner: str = 'tl') -> tuple[str, str, str]:
    """Return (align, anchor, fade) using readability + overlap constraints.

    Logo is a fixed brand mark (locked corner) and treated as a hard exclusion zone.
    """
    subj = subject_box(spec, pil_rgba)
    # Add clearspace so text never feels cramped against the logo
    logo_box = expand_rect(logo_boxes(spec)[logo_corner], pad=24, W=spec.W, H=spec.H)

    block_w, block_h, _ = text_block_metrics(spec, slide)
    max_w = spec.W - spec.margin_l - spec.margin_r
    w = max(block_w, max_w * 0.82)
    h = block_h + 10

    def box_for(align: str, anchor: str) -> Tuple[int, int, int, int]:
        if align == "center":
            x0 = int((spec.W - w) / 2)
        elif align == "right":
            x0 = int(spec.W - spec.margin_r - w)
        else:
            x0 = int(spec.margin_l)
        x1 = int(min(spec.W - spec.margin_r, x0 + w))

        if anchor == "top":
            y0 = int(spec.margin_t)
        elif anchor == "mid":
            y0 = int((spec.H - h) * 0.50)
        else:
            y0 = int(spec.H - spec.margin_b - h)
        y1 = int(min(spec.H - spec.margin_b, y0 + h))
        return (x0, y0, x1, y1)

    # Candidates like the Example: corners + centered band
    candidates = [
        ("left", "bottom"),
        ("right", "bottom"),
        ("center", "bottom"),
        ("left", "top"),
        ("right", "top"),
        ("center", "mid"),
    ]

    best = None
    for align, anchor in candidates:
        tb = box_for(align, anchor)
        overlap_logo = rect_iou(tb, logo_box)
        overlap_subj = rect_iou(tb, subj)
        # Hard exclusion: if text box intersects logo zone at all, skip.
        if overlap_logo > 0.0:
            continue
        s = score_box(pil_rgba, tb) + overlap_subj*140.0
        if best is None or s < best[0]:
            best = (s, align, anchor, tb)

    # If everything conflicted with logo (rare), fall back to bottom-left.
    if best is None:
        align, anchor = "left", "bottom"
        tb = (spec.margin_l, spec.H - spec.margin_b - 260, spec.W - spec.margin_r, spec.H - spec.margin_b)
        s = score_box(pil_rgba, tb)
    else:
        s, align, anchor, tb = best

    if s > 40:
        fade = "top" if anchor == "top" else ("mid" if anchor == "mid" else "bottom")
    else:
        fade = "none"

    return align, anchor, fade


# -------------------- Drawing --------------------

def draw_linear_fade(canvas: skia.Canvas, spec: Spec, where: str = "bottom"):
    """Draw a black fade to improve text contrast.

    where:
      - none: no fade
      - bottom: transparent -> black towards bottom
      - top: black at top -> transparent
      - mid: subtle band centered vertically (soft edges)
    """
    if where == "none":
        return

    if where == "top":
        grad_h = int(spec.H * spec.grad_h_ratio)
        y1 = grad_h
        shader = skia.GradientShader.MakeLinear(
            points=[skia.Point(0, 0), skia.Point(0, y1)],
            colors=[skia.ColorSetARGB(spec.grad_alpha_bottom, 0, 0, 0), skia.ColorSetARGB(0, 0, 0, 0)],
            positions=[0.0, 1.0],
        )
        canvas.drawRect(skia.Rect.MakeLTRB(0, 0, spec.W, y1), skia.Paint(Shader=shader, AntiAlias=True))
        return

    if where == "mid":
        band_h = int(spec.H * 0.22)
        y0 = int((spec.H - band_h) / 2)
        y1 = y0 + band_h
        canvas.drawRect(
            skia.Rect.MakeLTRB(0, y0, spec.W, y1),
            skia.Paint(Color=skia.ColorSetARGB(int(spec.grad_alpha_bottom * 0.45), 0, 0, 0), AntiAlias=True),
        )
        return

    # bottom
    grad_h = int(spec.H * spec.grad_h_ratio)
    y0 = spec.H - grad_h
    shader = skia.GradientShader.MakeLinear(
        points=[skia.Point(0, y0), skia.Point(0, spec.H)],
        colors=[skia.ColorSetARGB(0, 0, 0, 0), skia.ColorSetARGB(spec.grad_alpha_bottom, 0, 0, 0)],
        positions=[0.0, 1.0],
    )
    canvas.drawRect(skia.Rect.MakeLTRB(0, y0, spec.W, spec.H), skia.Paint(Shader=shader, AntiAlias=True))


def draw_text_block(
    canvas: skia.Canvas,
    spec: Spec,
    slide: SlideText,
    slide_num: int,
    total: int,
    align: str = "left",
    anchor: str = "bottom",
    contrast_hint: Optional[Image.Image] = None,
):
    title_tf, kicker_tf, body_tf = skia_typefaces()

    font_k = skia.Font(kicker_tf, spec.kicker_size)
    font_t = skia.Font(title_tf, spec.title_size)
    font_b = skia.Font(body_tf, spec.body_size)
    font_n = skia.Font(body_tf, spec.slide_num_size)

    # Fallback fonts (use body typeface to cover punctuation/symbols)
    font_k_fb = skia.Font(body_tf, spec.kicker_size)
    font_t_fb = skia.Font(body_tf, spec.title_size)
    font_b_fb = skia.Font(body_tf, spec.body_size)
    for f in (font_k, font_t, font_b, font_n, font_k_fb, font_t_fb, font_b_fb):
        f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)

    max_w = spec.W - spec.margin_l - spec.margin_r
    kicker_lines = wrap_to_width(slide.kicker, font_k, max_w) if slide.kicker else []
    title_lines = wrap_to_width(slide.title, font_t, max_w)
    body_lines = wrap_to_width(slide.body, font_b, max_w) if slide.body else []

    mk = font_k.getMetrics(); hk = mk.fDescent - mk.fAscent
    mt = font_t.getMetrics(); ht = mt.fDescent - mt.fAscent
    mb = font_b.getMetrics(); hb = mb.fDescent - mb.fAscent

    # Check if we should split title and body
    split_mode = getattr(slide, 'split_layout', False) and body_lines

    if split_mode:
        # Calculate title block height (kicker + title only)
        title_block_h = 0
        if kicker_lines:
            title_block_h += len(kicker_lines) * (hk + 6) + spec.gap
        title_block_h += len(title_lines) * (ht + 2)

        # Calculate body block height
        body_block_h = len(body_lines) * (hb + 8)

        # Title block at anchor position
        if anchor == "top":
            y_top_title = spec.margin_t
            # Body at opposite end (bottom)
            y_top_body = spec.H - spec.margin_b - body_block_h
        elif anchor == "mid":
            # If mid, keep together (fallback to normal mode)
            split_mode = False
            block_h = title_block_h + spec.gap + body_block_h
            y_top = (spec.H - block_h) * 0.50
        else:  # bottom
            y_top_title = spec.H - spec.margin_b - title_block_h
            # Body at opposite end (top)
            y_top_body = spec.margin_t
    else:
        # Normal mode: all text together
        block_h = 0
        if kicker_lines:
            block_h += len(kicker_lines) * (hk + 6) + spec.gap
        block_h += len(title_lines) * (ht + 2)
        if body_lines:
            block_h += spec.gap + len(body_lines) * (hb + 8)

        if anchor == "top":
            y_top = spec.margin_t
        elif anchor == "mid":
            y_top = (spec.H - block_h) * 0.50
        else:
            y_top = spec.H - spec.margin_b - block_h

    def x_for(line: str, font: skia.Font) -> float:
        if align == "center":
            return (spec.W - font.measureText(line)) / 2
        if align == "right":
            return spec.W - spec.margin_r - font.measureText(line)
        return spec.margin_l

    white = skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(255, 255, 255, 255))
    white_soft = skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(235, 255, 255, 255))

    def local_luma(x0: float, y0: float, x1: float, y1: float) -> float:
        if contrast_hint is None:
            return 128.0
        # clamp
        x0i = max(0, min(spec.W - 1, int(x0)))
        y0i = max(0, min(spec.H - 1, int(y0)))
        x1i = max(0, min(spec.W, int(x1)))
        y1i = max(0, min(spec.H, int(y1)))
        if x1i <= x0i + 2 or y1i <= y0i + 2:
            return 128.0
        crop = contrast_hint.crop((x0i, y0i, x1i, y1i)).convert('RGB')
        arr = np.asarray(crop, dtype=np.float32)
        gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        return float(gray.mean())

    def shadow_paint_for(luma: float, kind: str = 'title') -> skia.Paint:
        # Stronger shadow on bright backgrounds
        if luma >= 165:
            a = 190 if kind == 'title' else 150
        elif luma >= 145:
            a = 155 if kind == 'title' else 120
        else:
            a = 120 if kind == 'title' else 90
        return skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(a, 0, 0, 0))

    # Set starting y position (title block)
    y = y_top_title if split_mode else y_top

    # Kicker
    for line in kicker_lines:
        x = x_for(line, font_k)
        baseline = y - mk.fAscent
        luma = local_luma(x, baseline - hk, x + font_k.measureText(line), baseline + hk)
        sh = shadow_paint_for(luma, 'body')
        # shadow
        draw_string_with_fallback(canvas, line, x, baseline + 2, font_k, font_k_fb, sh)
        # main
        draw_string_with_fallback(canvas, line, x, baseline, font_k, font_k_fb, white_soft)
        y += hk + 6
    if kicker_lines:
        y += spec.gap

    # Title
    for line in title_lines:
        x = x_for(line, font_t)
        baseline = y - mt.fAscent
        luma = local_luma(x, baseline - ht, x + font_t.measureText(line), baseline + ht)
        sh = shadow_paint_for(luma, 'title')
        # pseudo-stroke (subtle) for very bright areas
        if luma >= 175:
            for dx, dy in [(0, 3), (1.5, 3), (-1.5, 3), (0, 2)]:
                draw_string_with_fallback(canvas, line, x + dx, baseline + dy, font_t, font_t_fb, sh)
        else:
            draw_string_with_fallback(canvas, line, x, baseline + 3, font_t, font_t_fb, sh)
        draw_string_with_fallback(canvas, line, x, baseline, font_t, font_t_fb, white)
        y += ht + 2

    # Body
    if body_lines:
        # If split mode, jump to body position; otherwise continue from title
        if split_mode:
            y = y_top_body
        else:
            y += spec.gap
        for line in body_lines:
            x = x_for(line, font_b)
            baseline = y - mb.fAscent
            luma = local_luma(x, baseline - hb, x + font_b.measureText(line), baseline + hb)
            sh = shadow_paint_for(luma, 'body')
            draw_string_with_fallback(canvas, line, x, baseline + 1.5, font_b, font_b_fb, sh)
            draw_string_with_fallback(canvas, line, x, baseline, font_b, font_b_fb, white_soft)
            y += hb + 8

    # Slide number: removed per request
    # (intentionally blank)


def draw_logo(canvas: skia.Canvas, logo: skia.Image, spec: Spec, corner: str = 'tl'):
    boxes = logo_boxes(spec)
    x0, y0, x1, y1 = boxes.get(corner, boxes['tl'])
    rect = skia.Rect.MakeLTRB(x0, y0, x1, y1)
    # subtle shadow
    shadow_rect = skia.Rect.MakeLTRB(x0, y0 + 3, x1, y1 + 3)
    canvas.drawImageRect(logo, shadow_rect, paint=skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(90, 0, 0, 0)))
    canvas.drawImageRect(logo, rect, paint=skia.Paint(AntiAlias=True))


def render_slide(
    bg_path: str,
    out_path: str,
    spec: Spec,
    slide: SlideText,
    slide_num: int,
    total: int,
    logo_img: skia.Image,
    centering=(0.5, 0.35),
    override_layout: tuple[str, str, str] | None = None,
):
    bg, pil_rgba = pil_to_skia_image(bg_path, spec.W, spec.H, centering=centering)

    if override_layout is not None:
        align, anchor, fade = override_layout
    else:
        align, anchor, fade = best_layout_for_slide(spec, pil_rgba, slide, logo_corner='tl')

    # Cover slide (slide 1) gets special treatment - EXTRA large, bold text
    is_cover = slide_num == 1 and slide.kicker is None
    if is_cover:
        # Create a special spec for cover slide with GRAND fonts
        cover_spec = Spec(
            W=spec.W,
            H=spec.H,
            margin_l=spec.margin_l,
            margin_r=spec.margin_r,
            margin_t=spec.margin_t,
            margin_b=spec.margin_b,
            gap=24,
            kicker_size=40,  # Larger for cover
            title_size=140,  # GRAND size for maximum impact
            body_size=40,    # Subtitle size
            slide_num_size=spec.slide_num_size,
            grad_h_ratio=spec.grad_h_ratio,
            grad_alpha_bottom=spec.grad_alpha_bottom,
            logo_size=spec.logo_size,
            logo_pad=spec.logo_pad,
        )
        spec = cover_spec

    surface = skia.Surface(spec.W, spec.H)
    canvas = surface.getCanvas()
    canvas.drawImageRect(bg, skia.Rect.MakeWH(spec.W, spec.H))

    # Fade disabled per user request - no black shadow banners
    draw_linear_fade(canvas, spec, where="none")
    draw_logo(canvas, logo_img, spec, corner='tl')
    draw_text_block(canvas, spec, slide.normalized(), slide_num, total, align=align, anchor=anchor, contrast_hint=pil_rgba)

    img = surface.makeImageSnapshot()
    data = img.encodeToData(skia.EncodedImageFormat.kJPEG, 95)
    with open(out_path, "wb") as f:
        f.write(bytes(data))


def main():
    spec = Spec()

    # Optional config file (used by Streamlit UI)
    cfg_path = os.environ.get("IG_CAROUSEL_CONFIG")
    cfg = None
    if cfg_path and os.path.exists(cfg_path):
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # allow overriding logo size/pad
        spec_over = (cfg.get("spec") or {})
        if "logo_pad" in spec_over:
            spec.logo_pad = int(spec_over["logo_pad"])
        if "logo_size" in spec_over:
            spec.logo_size = int(spec_over["logo_size"])

    if cfg:
        photos_dir = cfg["photos_dir"]
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        logo_path = cfg.get("logo_path") or "/Users/sirasasitorn/.openclaw/workspace/assets/logos/Logo JL.png"
        logo = load_logo(logo_path, spec.logo_size)

        slides_in = cfg.get("slides") or []
        slides: list[tuple[str, SlideText, tuple[float, float]]] = []
        for s in slides_in:
            photo = os.path.join(photos_dir, s.get("photo"))
            kicker = s.get("kicker")
            title = s.get("title") or ""
            body = s.get("body")
            split_layout = s.get("split_layout", False)
            centering = tuple(s.get("centering") or [0.5, 0.35])
            slides.append((photo, SlideText(kicker, title, body, split_layout).normalized(), centering))

        overrides_raw = cfg.get("overrides") or {}
        overrides: dict[int, tuple[str, str, str]] = {}
        for k, v in overrides_raw.items():
            try:
                overrides[int(k)] = (v[0], v[1], v[2])
            except Exception:
                pass

        total = len(slides)
        for i, (bg, txt, centering) in enumerate(slides, start=1):
            out = os.path.join(out_dir, f"Artboard {i}.jpg")
            render_slide(bg, out, spec, txt, i, total, logo, centering=centering, override_layout=overrides.get(i))

        print(out_dir)
        return

    # Default legacy behavior (Test Project hardcoded)
    root = "/Users/sirasasitorn/Desktop/Test Project"
    out_dir = os.path.join(root, "Output", "Mito HollyHock Carousel (Skia)")
    os.makedirs(out_dir, exist_ok=True)

    logo = load_logo("/Users/sirasasitorn/.openclaw/workspace/assets/logos/Logo JL.png", spec.logo_size)

    paths = {
        "action1": os.path.join(root, "JLG25144_098.jpg"),
        "action2": os.path.join(root, "JLG25170_056.jpg"),
        "action3": os.path.join(root, "JLG25223_055.jpg"),
        "crowd": os.path.join(root, "JLG25083_097.jpg"),
        "stadium1": os.path.join(root, "Assets/External/stadium_1.jpg"),
        "stadium2": os.path.join(root, "Assets/External/stadium_2.jpg"),
        "station": os.path.join(root, "Assets/External/station_1.jpg"),
        "training": os.path.join(root, "Assets/External/training_1.jpg"),
        "holly": os.path.join(root, "Assets/External/commons_mito/Hollypitch.png"),
    }

    slides: list[tuple[str, SlideText, tuple[float, float]]] = [
        (paths["action1"], SlideText("J.League Club Spotlight", "MITO HOLLYHOCK", "A quick intro for new fans — from Ibaraki, Japan."), (0.78, 0.30)),
        (paths["crowd"], SlideText("Who are they?", "HOME TOWN PRIDE", "A community-first club — supported by Ibaraki, through every moment."), (0.52, 0.52)),
        (paths["station"], SlideText("Getting there", "EASY DAY TRIP", "Reach Mito by train from the Tokyo area — great for a football weekend."), (0.50, 0.40)),
    ]

    overrides: dict[int, tuple[str, str, str]] = {1: ("left", "mid", "bottom"), 3: ("right", "bottom", "none")}

    total = len(slides)
    for i, (bg, txt, centering) in enumerate(slides, start=1):
        out = os.path.join(out_dir, f"Artboard {i}.jpg")
        render_slide(bg, out, spec, txt, i, total, logo, centering=centering, override_layout=overrides.get(i))

    print(out_dir)


if __name__ == "__main__":
    main()
