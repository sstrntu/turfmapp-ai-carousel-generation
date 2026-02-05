# Text Placement Configuration Guide

The `text_placement_rules.json` file lets you customize how the AI chooses text positions on your carousel slides.

## Quick Start

1. **Edit the config file**: `text_placement_rules.json`
2. **Save your changes**
3. **Generate a new carousel** - your rules are applied automatically!

No server restart needed - rules are loaded on each carousel generation.

## Config File Structure

### `enabled` (boolean)
Turn text placement rules on/off:
```json
"enabled": true
```
Set to `false` to use minimal default rules.

### `global_preferences`
General preferences applied to all slides:
```json
"global_preferences": {
  "vary_placement": true,        // Mix up positions across slides
  "avoid_faces": true,            // Never cover faces with text
  "prefer_fade_backgrounds": true, // Use gradients for readability
  "default_centering": [0.5, 0.35] // Default image focal point
}
```

### `placement_rules`
Core rules for choosing text position:

#### Align (Horizontal Position)
```json
"align": {
  "left": "Your rule for when to use left alignment",
  "center": "Your rule for when to use center alignment",
  "right": "Your rule for when to use right alignment"
}
```

**Example:**
```json
"align": {
  "left": "Use for portraits facing right, or when subject is on right side",
  "center": "Use for landscapes, symmetrical compositions, or title slides",
  "right": "Use for portraits facing left, or when subject is on left side"
}
```

#### Anchor (Vertical Position)
```json
"anchor": {
  "top": "Your rule for when to place text at top",
  "mid": "Your rule for when to place text in middle",
  "bottom": "Your rule for when to place text at bottom"
}
```

#### Fade (Background Overlay)
```json
"fade": {
  "top": "Dark gradient from top - when to use",
  "mid": "Dark center band - when to use",
  "bottom": "Dark gradient from bottom - when to use",
  "none": "No background - when to use"
}
```

### `slide_type_preferences`
Optional hints for different slide types:

```json
"slide_type_preferences": {
  "opening_slide": {
    "preferred_align": "center",
    "preferred_anchor": "mid",
    "preferred_fade": "mid",
    "note": "First slide - bold centered statement"
  },
  "portrait_photo": {
    "preferred_align": "left",
    "preferred_anchor": "top",
    "preferred_fade": "top",
    "note": "For photos with people - keep text away from faces"
  }
}
```

**Add your own slide types:**
```json
"your_custom_type": {
  "preferred_align": "right",
  "preferred_anchor": "bottom",
  "preferred_fade": "bottom",
  "note": "Your description here"
}
```

### `custom_rules`
Add your own specific requirements:

```json
"custom_rules": {
  "your_rules": [
    "Always use bottom anchor for slides with prominent sky",
    "Prefer right alignment for text-heavy slides",
    "Never use center alignment on slides 2-4",
    "Use fade='none' only on solid color backgrounds"
  ]
}
```

### `forbidden_combinations`
Prevent bad layout choices:

```json
"forbidden_combinations": {
  "rules": [
    "Never pair anchor='top' with fade='bottom'",
    "Never pair anchor='bottom' with fade='top'",
    "Avoid fade='none' on busy backgrounds with lots of detail"
  ]
}
```

## Example Configurations

### Minimal Configuration
Keep it simple - let the AI decide:
```json
{
  "enabled": true,
  "global_preferences": {
    "vary_placement": true,
    "avoid_faces": true
  },
  "custom_rules": {
    "your_rules": []
  }
}
```

### Strict Configuration
Specific requirements for brand consistency:
```json
{
  "enabled": true,
  "global_preferences": {
    "vary_placement": false,
    "avoid_faces": true,
    "prefer_fade_backgrounds": true
  },
  "custom_rules": {
    "your_rules": [
      "Always use left alignment for all slides",
      "Always use bottom anchor",
      "Always use fade='bottom'",
      "Never use center alignment except for first slide"
    ]
  }
}
```

### Photography-Focused Configuration
Optimize for photo-heavy carousels:
```json
{
  "enabled": true,
  "global_preferences": {
    "vary_placement": true,
    "avoid_faces": true,
    "prefer_fade_backgrounds": true
  },
  "custom_rules": {
    "your_rules": [
      "For landscape photos, always use anchor='bottom' to preserve sky",
      "For portrait photos, place text opposite to face direction",
      "For close-up shots, use anchor='top' or anchor='bottom' to avoid center",
      "Prefer fade='bottom' for outdoor scenes with sky"
    ]
  }
}
```

## How It Works

1. **On carousel generation**, the pipeline loads `text_placement_rules.json`
2. **Rules are converted** into natural language instructions
3. **Instructions are added** to the AI prompt
4. **AI analyzes each image** and applies your rules when choosing layout
5. **Slides are generated** with your preferred text placement style

## Testing Your Rules

After editing the config:

1. **Generate a test carousel** with 3-5 images
2. **Review the text placement** in the output slides
3. **Adjust rules** in the JSON file if needed
4. **Regenerate** to see the changes

## Environment Variable Override

Use a different config file location:

```bash
export TEXT_PLACEMENT_RULES_FILE="/path/to/custom_rules.json"
```

Or disable rules entirely:

```json
{
  "enabled": false
}
```

## Tips

✓ **Start simple** - Add rules gradually as you discover what works
✓ **Be specific** - Clear rules get better results
✓ **Test frequently** - Generate carousels to see how rules apply
✓ **Use examples** - Reference specific scenarios (e.g., "landscape with mountains")
✓ **Iterate** - Refine rules based on generated output

## Common Use Cases

### Brand Guidelines
```json
"custom_rules": {
  "your_rules": [
    "Always use right alignment to match brand style",
    "Always use fade='bottom' for consistency",
    "Never use anchor='mid' - only top or bottom"
  ]
}
```

### Social Media Optimization
```json
"custom_rules": {
  "your_rules": [
    "Vary anchor position across slides for visual rhythm",
    "Use center alignment on first and last slide only",
    "Prefer fade backgrounds for better mobile readability"
  ]
}
```

### Photography Portfolio
```json
"custom_rules": {
  "your_rules": [
    "Minimize text - let images speak",
    "Only use fade='none' to avoid covering photo details",
    "Always place text in least interesting part of composition"
  ]
}
```

---

**Questions?** Check `TEXT_PLACEMENT_GUIDE.md` for visual examples of each layout option.
