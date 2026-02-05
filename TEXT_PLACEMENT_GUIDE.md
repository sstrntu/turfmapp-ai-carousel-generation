# Text Placement Rules for IG Carousel

The AI now follows smart text placement rules based on image composition analysis.

## Layout System

Every slide has 4 placement parameters:

### 1. **ALIGN** (Horizontal Position)
Controls left-right text placement:

```
┌─────────────────────────────────┐
│ LEFT    │   CENTER   │   RIGHT  │
│ Text    │    Text    │    Text  │
│ Here    │    Here    │    Here  │
└─────────────────────────────────┘
```

**Rules:**
- **`left`**: Use when main subject is on the RIGHT, or empty space on LEFT
- **`center`**: Use for centered/symmetrical compositions
- **`right`**: Use when main subject is on the LEFT, or empty space on RIGHT

### 2. **ANCHOR** (Vertical Position)
Controls top-bottom text placement:

```
┌─────────────────────────────────┐
│ TOP:     Text Here              │ ← Use when bottom has detail
│                                 │
│ MID:     Text Here              │ ← Use for balanced images
│                                 │
│ BOTTOM:  Text Here              │ ← Use when top is important
└─────────────────────────────────┘
```

**Rules:**
- **`top`**: Lower portion has subjects (people, buildings, action)
- **`mid`**: Evenly distributed composition
- **`bottom`**: Upper portion is important (sky, mountains, architecture)

### 3. **FADE** (Text Background)
Adds gradient for readability:

- **`top`**: Dark gradient from top (pair with `anchor='top'`)
- **`mid`**: Dark center band (pair with `anchor='mid'`)
- **`bottom`**: Dark gradient from bottom (pair with `anchor='bottom'`)
- **`none`**: No background - only for clear, simple backgrounds

**Best Practice:** Match fade with anchor position
```json
{"anchor": "top", "fade": "top"}     ✓ Good
{"anchor": "top", "fade": "bottom"}  ✗ Bad
```

### 4. **CENTERING** [x, y]
Sets image focal point (what part of image to emphasize):

```
[0.0, 0.0] = Top-left corner
[0.5, 0.5] = Center
[1.0, 1.0] = Bottom-right corner
```

**Common values:**
- `[0.5, 0.35]` - Default (center, slightly up)
- `[0.3, 0.35]` - Focus on left side
- `[0.7, 0.35]` - Focus on right side
- `[0.5, 0.5]` - Center subject

## Complete Example

### Example 1: Person on Right Side
```json
{
  "kicker": "Profile",
  "title": "Meet the founder",
  "body": "Building dreams since 2020",
  "image": {"source": "upload", "filename": "portrait.jpg"},
  "centering": [0.6, 0.4],
  "layout": {
    "align": "left",      // Text on left, person on right
    "anchor": "mid",      // Middle height to avoid head/feet
    "fade": "mid"         // Match with anchor
  }
}
```

### Example 2: Landscape with Sky
```json
{
  "kicker": "Location",
  "title": "The perfect sunset",
  "body": "Golden hour magic in the mountains",
  "image": {"source": "external", "query": "mountain sunset golden hour"},
  "centering": [0.5, 0.4],
  "layout": {
    "align": "center",    // Centered composition
    "anchor": "bottom",   // Text at bottom, sky at top
    "fade": "bottom"      // Match with anchor
  }
}
```

### Example 3: Action Shot - Subject on Left
```json
{
  "kicker": "Action",
  "title": "Game winning moment",
  "body": "The goal that changed everything",
  "image": {"source": "upload", "filename": "soccer.jpg"},
  "centering": [0.35, 0.45],
  "layout": {
    "align": "right",     // Text on right, action on left
    "anchor": "top",      // Keep text away from ground level
    "fade": "top"         // Match with anchor
  }
}
```

## AI Best Practices

The AI now automatically:

1. ✓ **Analyzes image composition** before choosing placement
2. ✓ **Avoids covering faces** and key subjects
3. ✓ **Uses empty areas** of the image for text
4. ✓ **Varies placement** across slides (won't put all text in same spot)
5. ✓ **Matches fade with anchor** for visual consistency
6. ✓ **Adjusts centering** to focus on important parts of the image

## Testing Your Carousel

When you generate a carousel, the AI will:

1. Look at each uploaded image
2. Identify where subjects/details are located
3. Find empty/simple areas for text placement
4. Choose align, anchor, and fade that avoid covering important elements
5. Vary the layout across slides for visual interest

**Result:** Professional-looking slides with smart text placement! 🎨
