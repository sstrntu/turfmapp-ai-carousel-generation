# Interactive Text Editor

After generating a carousel, you can now adjust text positions by dragging elements around if text overlaps important parts of the image.

## How It Works

### 1. Generate Your Carousel
First, generate a carousel normally through the web interface:
- Enter your intent
- Upload images
- Click "Generate Carousel"

### 2. View Results
Once generation completes, you'll see thumbnails of all generated slides.

### 3. Open Lightbox
Click any thumbnail to open it in the lightbox (full-screen preview).

### 4. Edit Text Position
In the lightbox, you'll see an "Edit Text Position" button at the bottom right. Click it to open the editor.

### 5. Drag Text Elements
The editor shows:
- **Canvas**: The slide image in the background
- **Draggable overlays**: Yellow bordered boxes representing text elements
  - Title (large text)
  - Kicker (small text above title)
  - Body (text below title)

Simply **click and drag** any yellow box to reposition that text element.

### 6. Save Changes
When you're happy with the new positions:
1. Click **"Save & Re-render"**
2. The system will re-render just that slide with the new text positions
3. The page will refresh automatically to show the updated slide

If you don't want to save, click **"Cancel"** to close the editor without changes.

---

## Features

### Visual Editing
- ✓ See the actual image while editing
- ✓ Drag text elements with mouse
- ✓ Visual feedback (yellow highlights)
- ✓ Real-time position updates

### Responsive Design
- ✓ Works on desktop and tablet
- ✓ Scales to different screen sizes
- ✓ Touch-friendly on mobile

### Safety
- ✓ Changes are not saved until you click "Save"
- ✓ Cancel anytime without affecting the original
- ✓ Only re-renders the specific slide you edited
- ✓ Original config is preserved

---

## Use Cases

### 1. Text Covering Faces
**Problem:** Generated text overlaps someone's face

**Solution:**
1. Open the slide in lightbox
2. Click "Edit Text Position"
3. Drag text elements down or to the side
4. Save & Re-render

### 2. Text Over Important Details
**Problem:** Title covers a logo or important object

**Solution:**
1. Open editor
2. Drag title to empty area of image
3. Save

### 3. Better Visual Balance
**Problem:** Text looks off-center or awkward

**Solution:**
1. Open editor
2. Adjust all elements for better composition
3. Save

---

## Technical Details

### Text Elements

Each slide can have up to 3 text elements:

| Element | Description | Typical Position |
|---------|-------------|------------------|
| **Kicker** | Small label/category text | Above title |
| **Title** | Main headline | Center |
| **Body** | Descriptive text | Below title |

Not all slides have all elements (e.g., cover slides don't have kickers).

### Position System

Positions are stored as normalized coordinates (0-1):
- **x = 0.5** = center horizontally
- **y = 0.35** = 35% from top
- **x = 0** = left edge
- **x = 1** = right edge

The editor converts these to/from pixel positions for easy dragging.

### Re-rendering

When you save:
1. New positions are saved to `config.json`
2. The renderer is called for just that slide
3. Original image and text are used
4. Only the text position changes
5. Output file is updated: `slide_XX.jpg`

This takes ~2-3 seconds per slide.

---

## Keyboard Shortcuts

When lightbox is open:
- **Escape** - Close lightbox
- **Arrow Left** - Previous slide
- **Arrow Right** - Next slide

When editor is open:
- Drag with mouse or touch

---

## Troubleshooting

### Editor button doesn't appear
- Make sure generation completed successfully
- Refresh the page
- Check that you're viewing results, not still generating

### Can't drag elements
- Make sure you're clicking directly on the yellow box
- Try clicking and holding before dragging
- Check browser console for errors

### Save fails
- Check that the server is still running
- Look at browser console for error messages
- Try canceling and opening editor again

### Changes don't appear
- The page should auto-refresh after save
- If not, manually refresh the page
- Check the thumbnail - it should be updated

### Text is cut off or outside canvas
- Be careful not to drag text too far to edges
- If this happens, edit again and move back
- The system tries to keep text within margins

---

## API Endpoints

For developers integrating with the carousel generator:

### Get Slide Config
```http
GET /api/slide-config/{run_id}/{slide_num}
```

Returns:
```json
{
  "title": "MITO HOLLYHOCK",
  "title_pos": {"x": 540, "y": 472},
  "kicker": "J.League Spotlight",
  "kicker_pos": {"x": 540, "y": 422},
  "body": "A journey through history",
  "body_pos": {"x": 540, "y": 572}
}
```

### Update Slide
```http
POST /api/update-slide/{run_id}/{slide_num}
Content-Type: application/json

{
  "positions": [
    {"type": "title", "x": 540, "y": 500},
    {"type": "kicker", "x": 540, "y": 450},
    {"type": "body", "x": 540, "y": 600}
  ]
}
```

---

## Best Practices

### 1. Start with AI Placement
Let the AI choose initial positions based on:
- Image composition
- Subject detection
- Text placement rules

Only edit if there's a specific issue.

### 2. Maintain Visual Hierarchy
Keep relative positions consistent:
- Kicker above title
- Body below title
- Don't make gaps too large or small

### 3. Stay Within Safe Zones
Avoid placing text:
- Too close to edges (within 60px)
- Over faces or important subjects
- In areas with complex backgrounds

### 4. Test Readability
After editing, ask yourself:
- Can I read all text clearly?
- Does text have enough contrast?
- Is the composition balanced?

### 5. Be Consistent
If you edit one slide:
- Consider editing similar slides
- Maintain consistent positioning across carousel
- Keep the story flow intact

---

## Examples

### Before Editing
```
┌─────────────────────────────────┐
│                                 │
│     ┌─── Text Here ────┐       │
│     │ TITLE COVERING   │       │
│     │   THE FACE      │       │
│     └─────────────────┘       │
│          👤                    │
│         /|\                    │
└─────────────────────────────────┘
```

### After Editing
```
┌─────────────────────────────────┐
│          👤                    │
│         /|\                    │
│                                 │
│     ┌─── Text Here ────┐       │
│     │ TITLE BELOW      │       │
│     │  THE PERSON      │       │
│     └─────────────────┘       │
└─────────────────────────────────┘
```

Much better! The person's face is visible and the text is still readable.

---

## Future Enhancements

Planned features:
- [ ] Undo/Redo functionality
- [ ] Snap-to-grid option
- [ ] Alignment guides
- [ ] Resize text elements
- [ ] Font size adjustment
- [ ] Color picker for text/overlay
- [ ] Batch edit multiple slides
- [ ] Preset positions (top-left, center, etc.)

---

**Your carousels, your way!** 🎨✨
