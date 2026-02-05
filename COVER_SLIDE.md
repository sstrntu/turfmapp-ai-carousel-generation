# Cover Slide Feature

Every carousel now starts with a captivating cover slide - the opening slide that grabs attention and introduces your story.

## What is a Cover Slide?

The **cover slide** is slide #1 in your carousel - the first impression that viewers see. Think of it like a book cover or movie poster.

### Characteristics:

- ✓ **Bold, large title** (100px vs 80px for content slides)
- ✓ **Optional subtitle** for context
- ✓ **Most striking image** from your uploads
- ✓ **Centered text** for maximum impact
- ✓ **Dramatic positioning** (mid or bottom anchor)
- ✓ **No kicker** (just title + subtitle)

---

## How It Works

### 1. AI Selection

The AI automatically:
- **Analyzes all uploaded images** to find the most striking one
- **Generates a bold headline** (3-8 words) that captures the story
- **Creates an optional subtitle** for context
- **Chooses optimal text placement** for drama and impact

### 2. Automatic Generation

When you generate a carousel:
```
Your Input:
├── Intent: "Tell the history of Mito Holly Hock"
├── Upload: 9 images
└── Allow external: Yes

AI Creates:
├── Slide 1 (COVER): Bold title introducing the story
├── Slide 2: First content slide
├── Slide 3: Second content slide
└── ... etc.
```

### 3. Special Rendering

Cover slide gets special treatment:
```
COVER SLIDE (Slide 1):
├── Title font: 100px (vs 80px normal)
├── Subtitle font: 36px (vs 30px body)
├── No kicker text
├── Center alignment (default)
├── Mid/bottom anchor (drama)
└── Larger spacing

CONTENT SLIDES (Slides 2+):
├── Title font: 80px
├── Body font: 30px
├── Kicker: 28px
├── Varied alignment
└── Standard spacing
```

---

## Example Cover Slides

### Example 1: Sports Team History
```json
{
  "cover_slide": {
    "title": "MITO HOLLYHOCK",
    "subtitle": "A Journey Through J-League History",
    "image": {"source": "upload", "filename": "stadium_wide.jpg"},
    "layout": {
      "align": "center",
      "anchor": "mid",
      "fade": "mid"
    }
  }
}
```

**Result:**
```
┌─────────────────────────────────────┐
│                                     │
│      [Stadium Image - Wide Shot]    │
│                                     │
│      ─────────────────────          │
│      MITO HOLLYHOCK                 │
│      A Journey Through J-League     │
│      History                        │
│      ─────────────────────          │
│                                     │
└─────────────────────────────────────┘
```

### Example 2: Product Launch
```json
{
  "cover_slide": {
    "title": "INTRODUCING THE FUTURE",
    "subtitle": "Innovation meets design",
    "image": {"source": "external", "query": "modern tech product hero shot"},
    "layout": {
      "align": "center",
      "anchor": "bottom",
      "fade": "bottom"
    }
  }
}
```

**Result:**
```
┌─────────────────────────────────────┐
│                                     │
│      [Product Image - Top 2/3]      │
│                                     │
│      ─────────────────────          │
│      INTRODUCING THE FUTURE         │
│      Innovation meets design        │
│      ─────────────────────          │
└─────────────────────────────────────┘
```

---

## AI Guidelines for Cover Slides

The AI follows these rules when creating cover slides:

### Image Selection
1. **Most striking image** from uploads
2. **Visual impact** - action, drama, or beauty
3. **Representative** of the overall story
4. **High quality** and well-composed

### Title Creation
1. **Bold and attention-grabbing**
2. **3-8 words maximum**
3. **All caps or title case** for impact
4. **Captures the essence** of the story
5. **No generic phrases** like "Welcome" or "Introduction"

### Subtitle Creation
1. **Optional but recommended**
2. **Provides context** or tagline
3. **1 line maximum**
4. **Complements the title**

### Layout Preferences
1. **Align**: Center (default for maximum impact)
2. **Anchor**: Mid (dramatic) or Bottom (classic)
3. **Fade**: Mid (center focus) or Bottom (grounded look)

---

## Customizing Cover Slides

### Via Text Placement Rules

Edit `text_placement_rules.json`:

```json
{
  "slide_type_preferences": {
    "cover_slide": {
      "preferred_align": "center",
      "preferred_anchor": "bottom",
      "preferred_fade": "bottom",
      "note": "I prefer grounded cover slides with text at bottom"
    }
  },
  "custom_rules": {
    "your_rules": [
      "Cover slide title must be ALL CAPS",
      "Cover slide should always use the first uploaded image",
      "Cover subtitle must mention the brand name"
    ]
  }
}
```

### Via QC Validation

The QC system automatically validates:
- ✓ Cover title is attention-grabbing
- ✓ Image is appropriate for a cover
- ✓ Layout follows preferences
- ✓ Text doesn't cover important subjects

---

## Rendering Details

### Font Sizes

| Element | Cover Slide | Content Slides |
|---------|------------|----------------|
| Title   | 100px      | 80px           |
| Subtitle/Body | 36px | 30px           |
| Kicker  | N/A        | 28px           |

### Layout Defaults

```python
Cover Slide:
├── align: "center"      # Maximum impact
├── anchor: "mid"        # Dramatic center
└── fade: "mid"          # Center focus

Content Slides:
├── align: varies        # Based on image
├── anchor: varies       # Based on subjects
└── fade: varies         # Based on anchor
```

### Special Detection

The renderer detects cover slides by:
1. `slide_num == 1` (first slide)
2. `kicker is None` (no kicker = cover)

When detected, uses `cover_spec` with larger fonts.

---

## Workflow Example

### Input
```
User: "Create a carousel about coffee culture"
Images: [cafe1.jpg, latte_art.jpg, beans.jpg, barista.jpg]
```

### AI Processing
```
1. Analyze images
   → Most striking: latte_art.jpg (beautiful composition)

2. Generate cover
   Title: "THE ART OF COFFEE"
   Subtitle: "A journey through modern café culture"
   Layout: center, mid, mid

3. Generate content slides
   Slide 2: "Origins" - beans.jpg
   Slide 3: "The Craft" - barista.jpg
   Slide 4: "The Experience" - cafe1.jpg
```

### Output
```
Slide 1 (COVER):
┌────────────────────────────────┐
│      [Latte Art Close-up]      │
│                                │
│   ──────────────────────       │
│   THE ART OF COFFEE            │
│   A journey through modern     │
│   café culture                 │
│   ──────────────────────       │
└────────────────────────────────┘

Slide 2 (CONTENT):
┌────────────────────────────────┐
│      [Coffee Beans Image]      │
│                                │
│ Origins                        │
│ Where it all begins            │
│ From farm to cup, the story   │
│ of coffee starts with the bean│
└────────────────────────────────┘
```

---

## Benefits

### 1. Professional Presentation
- First impression matters
- Sets the tone for the entire carousel
- Looks polished and intentional

### 2. Better Engagement
- Captures attention immediately
- Clear introduction to the topic
- Viewers know what to expect

### 3. Storytelling Structure
- Cover → Introduction → Content → Closing
- Natural narrative flow
- Professional carousel structure

### 4. Brand Consistency
- Title/subtitle format
- Consistent opening style
- Recognizable pattern

---

## Testing

Generate a carousel and observe:

1. **Slide 1** should be the cover:
   - Large, bold title
   - Optional subtitle
   - Centered text
   - Striking image

2. **Slides 2+** should be content:
   - Regular font sizes
   - Kicker + title + body
   - Varied text positions
   - Story progression

---

## Troubleshooting

### Issue: Cover title too long
**Solution:** Add rule:
```json
"your_rules": ["Cover title must be 5 words or less"]
```

### Issue: Wrong image selected for cover
**Solution:** Add rule:
```json
"your_rules": ["Cover slide must use the first uploaded image"]
```

### Issue: Text too small on cover
**Solution:** Adjust in `ig-carousel-skia.py`:
```python
title_size=120,  # Even larger (default: 100)
```

### Issue: Want subtitle on separate line
**Default behavior:** Subtitle is the body field, automatically on separate line

---

**Your carousels now start with impact!** 🎬
