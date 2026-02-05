# Quality Control (QC) Validation

The QC validation step automatically reviews and fixes slides before final output. The AI checks everything to ensure quality and accuracy.

## How It Works

```
User Input → Generate Slides → QC Review → Fix Issues → Final Output
                                    ↓
                            ✓ Story alignment
                            ✓ Text-image match
                            ✓ Placement rules
                            ✓ Consistency
                            ✓ Completeness
```

## What Gets Checked

### 1. Story Alignment
- Does the overall narrative match user intent?
- Are slides telling a coherent story?
- Is the sequence logical?

**Example Fix:**
```
User Intent: "History of Apple Inc."
Generated Slide: "Microsoft's early days"
QC Fix: → Replace with relevant Apple history content
```

### 2. Text-Image Match
- Does each slide's image match its text?
- Are images relevant to the content?
- Is the visual storytelling consistent?

**Example Fix:**
```
Slide Text: "Mountain climbing adventure"
Image: Beach photo
QC Fix: → Request new image search or use different uploaded image
```

### 3. Text Placement Rules
- Does layout follow your custom rules?
- Is text positioned correctly (align, anchor, fade)?
- Are faces and subjects avoided?

**Example Fix:**
```
Rule: "Always use bottom anchor for sky photos"
Generated: anchor='top' with sky-heavy landscape
QC Fix: → Change to anchor='bottom'
```

### 4. Consistency
- Do slides flow well together?
- Is tone and style consistent?
- Are there narrative gaps?

**Example Fix:**
```
Slide 2: Casual, friendly tone
Slide 3: Formal, corporate tone
QC Fix: → Adjust slide 3 to match casual style
```

### 5. Completeness
- Are all required fields filled?
- Is layout properly configured?
- Are image sources specified?

**Example Fix:**
```
Generated: Missing "fade" in layout
QC Fix: → Add appropriate fade value based on anchor
```

## Configuration

Edit `text_placement_rules.json`:

```json
{
  "qc_config": {
    "enabled": true,
    "checks": {
      "story_alignment": true,
      "text_image_match": true,
      "placement_rules": true,
      "consistency": true,
      "completeness": true
    },
    "auto_fix": true,
    "log_findings": true
  }
}
```

### Configuration Options

#### `enabled` (boolean)
Turn QC validation on/off:
```json
"enabled": true   // Run QC validation (recommended)
"enabled": false  // Skip QC, output immediately
```

#### `checks` (object)
Enable/disable specific validation checks:

```json
"checks": {
  "story_alignment": true,   // Check if story matches intent
  "text_image_match": true,  // Check if images match text
  "placement_rules": true,   // Check text placement rules
  "consistency": true,       // Check narrative flow
  "completeness": true       // Check all fields are filled
}
```

**Use Cases:**
- Disable `story_alignment` if you trust initial generation
- Disable `text_image_match` if manually selecting all images
- Keep `placement_rules` enabled to enforce your custom rules

#### `auto_fix` (boolean)
Control whether AI fixes issues automatically:

```json
"auto_fix": true   // Fix issues automatically (recommended)
"auto_fix": false  // Only report issues, don't modify
```

**When to disable auto_fix:**
- You want to review issues manually
- You're testing your rules
- You prefer to regenerate instead of auto-correct

#### `log_findings` (boolean)
Show QC findings in the logs:

```json
"log_findings": true   // Show what QC found/fixed
"log_findings": false  // Silent validation
```

## QC Output

### In Logs

When generating a carousel, you'll see:

```
✓ Calling LLM to plan slides
✓ Building slide plan
✓ Running QC validation on slide plan
✓ QC notes: Fixed 2 issues:
  - Slide 2: Changed anchor from 'top' to 'bottom' to follow rule
  - Slide 4: Replaced image to better match text content
✓ Rendering slides
```

### No Issues Found

```
✓ Running QC validation on slide plan
✓ QC notes: No issues found
```

## Example QC Scenarios

### Scenario 1: Rule Violation

**Custom Rule:**
```json
"your_rules": ["Always use bottom anchor for landscape photos"]
```

**Generated Plan:**
```json
{
  "image": {"source": "external", "query": "mountain landscape"},
  "layout": {"anchor": "top", ...}
}
```

**QC Fix:**
```json
{
  "image": {"source": "external", "query": "mountain landscape"},
  "layout": {"anchor": "bottom", ...}  // ← Fixed
}
```

### Scenario 2: Text-Image Mismatch

**Slide Text:** "The birth of smartphones"
**Image:** Desktop computer from 1980s

**QC Fix:** Updates image query to "first smartphone iPhone 2007"

### Scenario 3: Missing Layout

**Generated:**
```json
{
  "title": "Hello World",
  "layout": {"align": "left", "anchor": "top"}  // Missing 'fade'
}
```

**QC Fix:**
```json
{
  "title": "Hello World",
  "layout": {"align": "left", "anchor": "top", "fade": "top"}  // ← Added
}
```

### Scenario 4: Story Doesn't Match Intent

**User Intent:** "History of jazz music"
**Slide 3 Text:** "The rise of rock and roll"

**QC Fix:** Rewrites slide 3 to focus on bebop jazz instead

## Disabling QC

If you want to skip QC validation:

### Option 1: In Config File
```json
{
  "qc_config": {
    "enabled": false
  }
}
```

### Option 2: Disable Specific Checks
```json
{
  "qc_config": {
    "enabled": true,
    "checks": {
      "story_alignment": true,
      "text_image_match": false,  // Skip this check
      "placement_rules": true,
      "consistency": true,
      "completeness": false       // Skip this check
    }
  }
}
```

## Performance Impact

- **Time:** QC adds ~5-15 seconds per carousel (one additional AI call)
- **Cost:** One extra API call to OpenAI
- **Quality:** Significantly improved accuracy and consistency

**Recommendation:** Keep QC enabled. The quality improvement is worth the small time cost.

## Troubleshooting

### QC Takes Too Long
- Check your internet connection
- Reduce number of slides
- OpenAI API might be slow - retry later

### QC Changes Too Much
- Set `auto_fix: false` to see issues without modifications
- Review and refine your custom rules
- Check if your rules are too strict or unclear

### QC Doesn't Catch Issues
- Make sure `enabled: true` in config
- Check if specific check is enabled
- Ensure your custom rules are clearly written
- AI might interpret rules differently - add examples

### QC Breaks My Slides
- Set `auto_fix: false` temporarily
- Review QC notes to see what changed
- Adjust rules to be more specific
- Report issue if AI is misinterpreting rules

## Best Practices

✓ **Keep QC enabled** - Quality is worth the extra time
✓ **Review QC logs** - Learn what issues occur frequently
✓ **Refine rules based on QC** - If QC keeps fixing the same thing, update your rules
✓ **Use auto_fix=true** - Let AI handle corrections
✓ **Enable all checks** - Comprehensive validation catches more issues
✓ **Write clear custom rules** - Specific rules get better QC results

## Technical Details

**QC Process:**
1. Initial plan generated by AI
2. Plan + original images + rules sent to QC reviewer AI
3. QC AI analyzes each slide against checklist
4. Issues identified and fixes proposed
5. Corrected plan returned with QC notes
6. Slides rendered from validated plan

**What QC Sees:**
- Original user intent
- All uploaded images (visual analysis)
- Generated slide plan
- Your custom placement rules
- Forbidden combinations

**QC Capabilities:**
- Change text content
- Modify layout (align, anchor, fade)
- Adjust centering values
- Request different images
- Reorder slides (if needed)
- Add/remove fields

---

**Questions?** Check `TEXT_PLACEMENT_CONFIG.md` for rule configuration help.
