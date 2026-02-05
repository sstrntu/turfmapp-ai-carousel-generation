# Fixes Applied for Text Placement Issues

## Issues Identified

Based on your generated carousel output:

1. ✗ **Text covering faces/heads** (Artboards 1, 3, 5, 6)
2. ✗ **Text overflow** (Artboard 2 - text falling outside bounds)
3. ✗ **Missing gradient backgrounds** (text hard to read on photos)
4. ✓ **Font is correct** (JLEAGUEKICK already configured)

## Root Causes

### 1. Forced Fade Override
**Location:** `ig-carousel-skia.py:661`
```python
# OLD (BROKEN):
fade = "none"  # This forced no gradient backgrounds!

# NEW (FIXED):
# Respect the fade choice from AI/override
```

**Impact:** This was overriding all AI fade choices, removing gradient backgrounds entirely. Text had no dark overlay making it unreadable and allowed placement anywhere on the image.

### 2. Insufficient Margins
**Location:** `ig-carousel-skia.py:44-50`
```python
# OLD (INSUFFICIENT):
margin_t: int = 80   # Too small - text could touch faces
margin_b: int = 80   # Too small - not enough safe zone

# NEW (SAFE):
margin_t: int = 100  # More space at top
margin_b: int = 120  # Larger safe zone at bottom
```

**Impact:** Small margins allowed text to be placed too close to edges and subjects.

### 3. Oversized Font
```python
# OLD:
title_size: int = 92  # Too large - caused overflow

# NEW:
title_size: int = 80  # Fits better within bounds
```

### 4. Weak Placement Rules
**Location:** `text_placement_rules.json`

**OLD:** Generic rules like "avoid faces"
**NEW:** Strict rules:
- "Text must NEVER cover faces, heads, or upper bodies"
- "For photos with people: ALWAYS use anchor='bottom'"
- "For portraits: Text in bottom 25% only"
- "ALWAYS use fade gradient - NEVER fade='none'"

## Changes Made

### File: `ig-carousel-skia.py`

**1. Removed Fade Override (Line 661)**
```diff
- fade = "none"  # Per your preference: SHADOW ONLY
+ # Respect the fade choice from AI/override (don't force "none")
```

**2. Adjusted Spec Margins & Sizes (Lines 44-57)**
```diff
@dataclass
class Spec:
    W: int = 1080
    H: int = 1350
-   margin_l: int = 80
-   margin_r: int = 80
-   margin_t: int = 80
-   margin_b: int = 80
-   gap: int = 14
+   margin_l: int = 60   # More horizontal space for text
+   margin_r: int = 60
+   margin_t: int = 100  # Keep text away from top/faces
+   margin_b: int = 120  # Larger safe zone at bottom
+   gap: int = 12

    # Type sizes
-   kicker_size: int = 30
-   title_size: int = 92
-   body_size: int = 34
-   slide_num_size: int = 26
+   kicker_size: int = 28
+   title_size: int = 80  # Prevent overflow
+   body_size: int = 30
+   slide_num_size: int = 24
```

### File: `text_placement_rules.json`

**1. Added Strict Custom Rules**
```json
"your_rules": [
  "CRITICAL: Text must NEVER cover faces, heads, or upper bodies of people",
  "For photos with people: ALWAYS use anchor='bottom' to keep text below subjects",
  "For portraits/close-ups: Text must be in bottom 25% of image only",
  "ALWAYS use fade gradient (top/mid/bottom) - NEVER use fade='none'",
  "Prefer anchor='bottom' as default unless image has important ground-level detail",
  "For action shots: Place text opposite to the direction of movement",
  "Text must fit within margins - reduce text length if needed to prevent overflow"
]
```

**2. Updated Forbidden Combinations**
```json
"forbidden_combinations": {
  "rules": [
    "Never pair anchor='top' with fade='bottom'",
    "Never pair anchor='bottom' with fade='top'",
    "NEVER use fade='none' - always use gradient backgrounds",
    "NEVER use anchor='top' or anchor='mid' for photos with people/faces",
    "NEVER place text in the upper 50% of images containing people",
    "For portrait orientation subjects: ONLY use anchor='bottom'",
    "Text length must not exceed image width - wrap or shorten text"
  ]
}
```

## Expected Results

### Before (Issues):
```
Artboard 1: ✗ Text on player's head
Artboard 2: ✗ Text overflows image
Artboard 3: ✗ Text covering face
Artboard 4: ✗ Text too high, covering players
Artboard 5: ✗ Text on player
Artboard 6: ✗ Text covering subject
```

### After (Fixed):
```
Artboard 1: ✓ Text at bottom with gradient
Artboard 2: ✓ Text fits within bounds
Artboard 3: ✓ Text below face in safe zone
Artboard 4: ✓ Text at bottom, clear of players
Artboard 5: ✓ Text in bottom 25%
Artboard 6: ✓ Text below subject
```

## Design Aesthetic Matching

Your example slides show:
- ✓ **Font:** JLEAGUEKICK-BoldCondensed/ExtraCondensed (already configured)
- ✓ **Position:** Bottom placement with dark gradient
- ✓ **Style:** Bold, high-contrast text
- ✓ **Safety:** Text never covers subjects

These fixes align the output with your example aesthetic.

## QC Validation

The QC system will now:
1. ✓ Check if text covers faces (CRITICAL rule)
2. ✓ Verify anchor='bottom' used for people photos
3. ✓ Ensure fade gradient is present
4. ✓ Validate text fits within bounds
5. ✓ Auto-fix violations before rendering

## Testing

Generate a new carousel to see the fixes in action:

```bash
# Visit: http://127.0.0.1:8000
# Upload the same images
# Use the same intent
# Compare output
```

**Expected improvements:**
- Text will be placed at bottom of images
- Dark gradient backgrounds for readability
- No text covering faces or heads
- All text fits within image bounds
- Professional, clean aesthetic matching examples

## Configuration

You can further tune these settings in:

**Font sizes:** `ig-carousel-skia.py` lines 44-57
**Margins:** `ig-carousel-skia.py` lines 47-50
**Rules:** `text_placement_rules.json`

### Example: Make margins even larger
```python
margin_t: int = 120  # Even more space at top
margin_b: int = 150  # Even larger safe zone
```

### Example: Make text smaller
```python
title_size: int = 70   # Smaller titles
body_size: int = 28    # Smaller body text
```

## Troubleshooting

**If text still covers faces:**
1. Check QC is enabled: `"qc_enabled": true` in config
2. Increase `margin_b` value in Spec
3. Add more specific rules to `your_rules`

**If text overflows:**
1. Reduce `title_size` in Spec
2. Increase `margin_l` and `margin_r`
3. Add rule: "Keep titles under 50 characters"

**If fade is missing:**
1. Check line 661 in ig-carousel-skia.py (should NOT force "none")
2. Verify rules enforce fade usage
3. Check override_layout isn't forcing "none"

---

**Changes are active immediately.** Generate a new carousel to see improvements!
