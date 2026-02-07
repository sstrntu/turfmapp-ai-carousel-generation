from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI

from env import load_env

load_env()

from config import MAX_SLIDES, OPENAI_MODEL
from image_analyzer import analyze_all_images
from image_search import download_image, search_images, search_web

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_DIR / "ig-carousel-skia.py"
TEXT_PLACEMENT_RULES_PATH = PROJECT_DIR / "text_placement_rules.json"


def _call_openai(client: OpenAI, messages: list[dict], response_format=None, retry_count=3) -> str:
    """Helper to call OpenAI API and extract text response with retry logic."""
    import time

    for attempt in range(retry_count):
        try:
            kwargs = {"model": OPENAI_MODEL, "messages": messages}
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        except Exception as e:
            import traceback
            error_msg = str(e)

            # Log the error
            print(f"OpenAI API error (attempt {attempt + 1}/{retry_count}): {error_msg}")

            if attempt == 0:
                # Only print detailed info on first attempt
                print(f"Model: {OPENAI_MODEL}")
                print(f"Error details: {traceback.format_exc()}")

            # Check if it's a 500 error (server-side issue) that we should retry
            if "500" in error_msg or "InternalServerError" in error_msg:
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"→ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

            # For other errors or last attempt, return empty
            if attempt == retry_count - 1:
                print(f"❌ All {retry_count} attempts failed")
            return ""

    return ""


def _load_text_placement_rules() -> dict[str, Any] | None:
    """Load user-defined text placement rules from JSON config file."""
    if not TEXT_PLACEMENT_RULES_PATH.exists():
        return None
    try:
        with open(TEXT_PLACEMENT_RULES_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
        return rules if rules.get("enabled", True) else None
    except Exception:
        return None


def _build_text_placement_prompt(rules: dict[str, Any] | None) -> str:
    """Build text placement instructions from config rules."""
    if not rules:
        # Default minimal rules
        return (
            "\n\nTEXT PLACEMENT: Analyze each image and choose layout that avoids covering subjects."
            " Use align (left/center/right), anchor (top/mid/bottom), and fade (top/mid/bottom/none)."
        )

    prompt_parts = ["\n\nTEXT PLACEMENT RULES:"]

    # Global preferences
    global_prefs = rules.get("global_preferences", {})
    if global_prefs.get("vary_placement"):
        prompt_parts.append("- Vary text placement across slides for visual interest")
    if global_prefs.get("avoid_faces"):
        prompt_parts.append("- NEVER cover faces or main subjects with text")
    if global_prefs.get("prefer_fade_backgrounds"):
        prompt_parts.append("- Prefer fade backgrounds for readability on busy images")

    # Placement rules
    placement = rules.get("placement_rules", {})
    if placement.get("align"):
        prompt_parts.append("\nALIGN (horizontal):")
        for pos, desc in placement["align"].items():
            prompt_parts.append(f"  • {pos}: {desc}")

    if placement.get("anchor"):
        prompt_parts.append("\nANCHOR (vertical):")
        for pos, desc in placement["anchor"].items():
            prompt_parts.append(f"  • {pos}: {desc}")

    if placement.get("fade"):
        prompt_parts.append("\nFADE (background):")
        for style, desc in placement["fade"].items():
            prompt_parts.append(f"  • {style}: {desc}")

    # Custom rules
    custom = rules.get("custom_rules", {}).get("your_rules", [])
    if custom and any(r and r != "Add your custom rules here" for r in custom):
        prompt_parts.append("\nCUSTOM RULES:")
        for rule in custom:
            if rule and rule != "Add your custom rules here":
                prompt_parts.append(f"  • {rule}")

    # Forbidden combinations
    forbidden = rules.get("forbidden_combinations", {}).get("rules", [])
    if forbidden:
        prompt_parts.append("\nAVOID:")
        for rule in forbidden:
            prompt_parts.append(f"  • {rule}")

    # Slide type preferences (optional hints)
    slide_types = rules.get("slide_type_preferences", {})
    if slide_types:
        prompt_parts.append("\nSLIDE TYPE HINTS (use as guidelines, not strict requirements):")
        for slide_type, prefs in slide_types.items():
            if isinstance(prefs, dict) and "note" in prefs:
                prompt_parts.append(f"  • {slide_type}: {prefs['note']}")

    return "\n".join(prompt_parts)


def _encode_image_b64(path: str, max_size: int = 1024) -> tuple[str, str]:
    """Encode image to base64, resizing to max_size to reduce payload."""
    from PIL import Image
    import io

    try:
        # Open and resize image to reduce payload size
        img = Image.open(path)

        # Convert to RGB if necessary (remove alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background

        # Resize if larger than max_size
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Encode to JPEG with good quality
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        data = buffer.getvalue()

        return "image/jpeg", base64.b64encode(data).decode("utf-8")
    except Exception as e:
        print(f"Warning: Failed to resize image {path}: {e}, using original")
        # Fallback to original encoding
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/jpeg"
        with open(path, "rb") as f:
            data = f.read()
        return mime, base64.b64encode(data).decode("utf-8")


def _default_plan(intent: str, image_names: list[str]) -> dict[str, Any]:
    first = image_names[0] if image_names else None
    slides = [
        {
            "kicker": "Story",
            "title": intent[:60].upper() if intent else "CAROUSEL",
            "body": intent if intent else "",
            "image": {"source": "upload", "filename": first} if first else None,
            "centering": [0.5, 0.35],
        }
    ]
    return {"slides": slides}


def _qc_slide_plan(
    intent: str,
    image_paths: list[str],
    plan: dict[str, Any],
    placement_rules: dict[str, Any] | None,
    log=None
) -> dict[str, Any]:
    """Quality control check - AI reviews and fixes the slide plan before rendering."""

    # Check if QC is enabled
    if not placement_rules or not placement_rules.get("qc_config", {}).get("enabled", True):
        if log:
            log("→ QC validation skipped (disabled in config)")
        return plan

    client = OpenAI()

    if log:
        log("→ Validating slide plan with AI reviewer...")
        qc_checks = placement_rules.get("qc_config", {}).get("checks", {})
        enabled_checks = [k for k, v in qc_checks.items() if v]
        if enabled_checks:
            log(f"  • Checks: {', '.join(enabled_checks)}")

    qc_config = placement_rules.get("qc_config", {})

    image_names = [Path(p).name for p in image_paths]
    slides = plan.get("slides", [])
    checks = qc_config.get("checks", {})

    qc_prompt = (
        "You are a quality control reviewer for Instagram carousel slides."
        " Your PRIMARY jobs: (1) Ensure detailed body text (120-180 chars), (2) Match facts to images, (3) Use actual data."
        "\n\n🚨 CRITICAL RULES:"
        "\n1. ALL text content (titles, body text) MUST be in English only"
        "\n2. Body text must be ACTUAL CONTENT for readers - NEVER meta-instructions"
        "\n3. If you fix something, replace with REAL content, not notes like 'needs verification' or 'update this'"
        "\n\n❌ FORBIDDEN IN BODY TEXT:"
        "\n• 'must be verified'"
        "\n• 'needs confirmation'"
        "\n• 'update this slide with'"
        "\n• 'replace with confirmed data'"
        "\n• Any instructions to verify or update"
        "\n\n✅ BODY TEXT MUST BE:"
        "\n• Actual facts users will read"
        "\n• Informative and specific"
        "\n• Complete sentences about the topic"
        "\n\n⚠️ BODY TEXT LENGTH & DETAIL CHECK:"
        "\nEach slide's body MUST be:"
        "\n• 1-2 sentences"
        "\n• 120-180 characters for meaningful detail"
        "\n• Includes specific facts, numbers, and context"
        "\n• Scannable yet informative"
        "\nIf body is too short → Add detail and specific facts"
        "\nExample: 'Top scorer' (TOO SHORT - lacks context)"
        "\n→ Fix to: 'Matheus Jesus scored 19 goals this season, leading J2 League and driving the promotion push.' (96 chars, needs more detail)"
        "\n→ Better: 'Matheus Jesus scored 19 goals this season, leading J2 League in scoring and driving the promotion push with crucial late-season performances.' (143 chars, good detail)"
        "\n\n⚠️ IMAGE-FACT MATCHING:"
        "\nCheck each slide:"
        "\n• Does the image match the fact being discussed?"
        "\n• Celebration photo should have victory/promotion fact"
        "\n• Player photo should have that player's stats"
        "\n• If fact doesn't match any uploaded image → suggest external search"
        "\n\n⚠️ REJECT PLACEHOLDER SLIDES:"
        "\n❌ Ranges: (0-38), (1-20), meta-instructions"
        "\n✅ Use actual data: '19 goals', '0-3 loss', '2nd place'"
        "\n\n⚠️ DO NOT include source URLs or citations in slide text."
        "\n\n"
        "VALIDATION CHECKLIST:\n"
    )

    checklist = []
    if checks.get("story_alignment", True):
        checklist.append("1. STORY ALIGNMENT: Does the overall narrative match the user's intent?")
    if checks.get("text_image_match", True):
        checklist.append("2. TEXT-IMAGE MATCH: Does each slide's image match its text content?")
    if checks.get("placement_rules", True):
        checklist.append("3. TEXT PLACEMENT: Does the layout follow the placement rules?")
    if checks.get("consistency", True):
        checklist.append("4. CONSISTENCY: Is the story coherent and flows well across slides?")
    if checks.get("completeness", True):
        checklist.append("5. COMPLETENESS: Are all fields properly filled (kicker, title, body, layout)?")

    qc_prompt += "\n".join(checklist) + "\n\n"

    if checks.get("placement_rules", True):
        qc_prompt += "PLACEMENT RULES TO VERIFY:\n"

    # Add placement rules to QC prompt
    if placement_rules:
        placement_prompt = _build_text_placement_prompt(placement_rules)
        qc_prompt += placement_prompt
    else:
        qc_prompt += "- Text should not cover faces or main subjects\n"
        qc_prompt += "- Layout should use align, anchor, and fade appropriately\n"

    qc_prompt += (
        "\n\n"
        "CUSTOM RULES TO VERIFY:\n"
    )
    if placement_rules:
        custom = placement_rules.get("custom_rules", {}).get("your_rules", [])
        if custom and any(r and r != "Add your custom rules here" for r in custom):
            for rule in custom:
                if rule and rule != "Add your custom rules here":
                    qc_prompt += f"- {rule}\n"

    auto_fix = qc_config.get("auto_fix", True)
    qc_prompt += (
        "\n\n"
        "INSTRUCTIONS:\n"
        "1. Review each slide carefully\n"
        "2. Check for issues: wrong image choice, text-image mismatch, rule violations, unclear story\n"
    )

    if auto_fix:
        qc_prompt += (
            "3. If issues found: Fix them and return the corrected plan\n"
            "4. If everything is correct: Return the plan as-is\n"
        )
    else:
        qc_prompt += (
            "3. If issues found: Note them but do NOT modify the plan\n"
            "4. Return the plan as-is\n"
        )

    qc_prompt += (
        "5. Add a 'qc_notes' field with your findings (issues found and fixes made, or 'No issues found')\n"
        "\n"
        "Output the validated/corrected plan as valid JSON."
    )

    user_content_text = (
        f"USER INTENT: {intent}\n\n"
        f"AVAILABLE IMAGES: {', '.join(image_names)}\n\n"
        f"GENERATED SLIDE PLAN:\n{json.dumps(plan, indent=2)}\n\n"
        "SPECIAL ATTENTION TO COVER SLIDE:\n"
        "- Cover slide is the first impression - must be captivating\n"
        "- Image should be the most striking/representative\n"
        "- Title should be bold and attention-grabbing (3-8 words)\n"
        "- Subtitle should provide context (1 line)\n"
        "- Layout should use center alignment for maximum impact\n"
        "- Anchor should be mid or bottom for dramatic effect\n\n"
        "Review this plan and return the validated/corrected version with 'qc_notes' field."
    )

    # Build content with text and images
    content = [{"type": "text", "text": user_content_text}]

    # Include images for visual validation
    for path in image_paths:
        mime, b64 = _encode_image_b64(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    messages = [
        {"role": "system", "content": qc_prompt},
        {"role": "user", "content": content}
    ]

    text = _call_openai(client, messages, response_format={"type": "json_object"})

    if not text:
        if log:
            log("⚠️ QC validation failed, using original plan")
        return plan

    try:
        validated_plan = json.loads(text)
        qc_notes = validated_plan.get("qc_notes", "")
        if log:
            if qc_notes:
                if qc_notes.lower() in ["no issues found", "no issues", "none"]:
                    log(f"✓ QC passed: No issues found")
                else:
                    log(f"✓ QC complete with adjustments:")
                    # Show first 200 chars of notes
                    preview = qc_notes[:200] + "..." if len(qc_notes) > 200 else qc_notes
                    log(f"  {preview}")
            else:
                log("✓ QC validation complete")
        return validated_plan
    except Exception:
        # QC failed, return original plan
        if log:
            log("⚠️ Failed to parse QC response, using original plan")
        return plan


def _intelligent_research(intent: str, log=None) -> str:
    """Use GPT with function calling for web search.

    GPT decides what to search for, calls search functions, and synthesizes results.
    Returns a structured summary of key moments with specific data.
    """
    if log:
        log("🔍 STEP 1: GPT Research with Function Calling")
        log(f"→ GPT analyzing research needs: '{intent}'")

    client = OpenAI()

    # Define web search tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information. Use this to find specific facts, statistics, match results, and data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'V-Varen Nagasaki 2025 match results')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # System message for research
    system_msg = (
        "You are a sports research analyst. Your task is to research the user's request and provide a detailed summary."
        "\n\nYou have access to web search. Use it to find SPECIFIC facts:"
        "\n• Match results (scores, opponents, dates)"
        "\n• Statistics (goals, positions, points)"
        "\n• Player data (names, achievements)"
        "\n• Events (managerial changes, unbeaten runs)"
        "\n\nAfter gathering information, create a structured summary with 4-6 key moments like this:"
        "\n\n🔑 1. Title\nSpecific details with numbers and names..."
        "\n\n⚠️ 2. Title\nMore specific details..."
        "\n\nBe thorough - search multiple times if needed to get complete information."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Research this topic and provide a detailed summary:\n\n{intent}"}
    ]

    max_iterations = 10
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1

            # Call GPT with function calling
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if GPT wants to call a function
            if assistant_message.tool_calls:
                if log and iteration == 1:
                    log(f"→ GPT requesting {len(assistant_message.tool_calls)} web search(es)...")

                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    if function_name == "search_web":
                        query = arguments.get("query")
                        max_results = arguments.get("max_results", 10)

                        if log:
                            log(f"  • Search: '{query}'")

                        # Execute search
                        results = search_web(query, max_results=max_results)

                        # Format results
                        results_text = f"Found {len(results)} results:\n\n"
                        for i, r in enumerate(results[:10], 1):
                            results_text += f"{i}. {r['title']}\n{r['snippet']}\n\n"

                        # Add function result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": results_text
                        })

            else:
                # GPT has finished and provided final answer
                final_answer = assistant_message.content

                if log:
                    log(f"✓ GPT research complete after {iteration} iteration(s)")
                    if final_answer:
                        lines = final_answer.split('\n')
                        preview = '\n'.join(lines[:4])
                        if len(lines) > 4:
                            preview += "\n..."
                        log(f"→ Preview:\n{preview}")

                return final_answer or ""

        # Max iterations reached
        if log:
            log(f"⚠️ Max iterations reached, using last response")
        return messages[-1].get("content", "")

    except Exception as e:
        if log:
            log(f"❌ GPT research failed: {e}")
            import traceback
            log(f"   Error: {str(e)}")
        return ""


def plan_slides(intent: str, image_paths: list[str], allow_external: bool, log=None) -> dict[str, Any]:
    client = OpenAI()

    # Intelligent research: LLM analyzes, searches, and synthesizes like ChatGPT
    research_data = _intelligent_research(intent, log=log)

    if log:
        log("")
        log("📝 STEP 2: Slide Planning")

    image_names = [Path(p).name for p in image_paths]

    if log:
        log(f"→ Available images: {len(image_names)}")
        if research_data:
            log("→ Using researched facts for slide content")
        else:
            log("→ Generating slides without research data")

    # Load user-defined text placement rules
    placement_rules = _load_text_placement_rules()
    placement_prompt = _build_text_placement_prompt(placement_rules)

    system_prompt = (
        "You are a data-driven Instagram carousel creator. Your slides MUST be detailed yet scannable and match the images."
        "\n\n🔍 IMAGE ANALYSIS - FIRST PRIORITY:"
        "\nBEFORE planning any slide, carefully examine EACH uploaded image:"
        "\n• WHO: Is this a player, manager, coach, referee, or crowd?"
        "\n• WHAT: Action shot, celebration, portrait, stadium, tactical setup?"
        "\n• CONTEXT: Match day, training, press conference, award ceremony?"
        "\n• BACKGROUND: Stadium visible? Specific venue? Event atmosphere?"
        "\n• DETAILS: Jersey numbers, expressions, body language, setting, timing cues"
        "\n\n🎯 MATCHING PHILOSOPHY - TWO TYPES OF CONNECTIONS:"
        "\n"
        "\n1️⃣ DIRECT MATCHING (Obvious):"
        "\n• Manager/coach photo → Management changes, tactical achievements"
        "\n• Player action shot → That specific player's performance"
        "\n• Celebration → Victories, promotions achieved"
        "\n• Stadium/crowd → Attendance, venue facts"
        "\n"
        "\n2️⃣ CONTEXTUAL MATCHING (Nuanced - for fans who know the story):"
        "\n• Player photo at specific stadium → Stadium opening/renovation topic (background connection)"
        "\n• Team celebration from key match → Season summary or promotion journey (event context)"
        "\n• Action shot from rival match → Rivalry history or that match's significance"
        "\n• Manager celebrating → Could work for team achievement if manager led that success"
        "\n• Training photo → Pre-season prep, squad building, tactical setup topics"
        "\n"
        "\n✅ EXAMPLES OF GOOD CONTEXTUAL MATCHING:"
        "\n• Photo: Player at new stadium → Topic: Stadium opens (background shows venue)"
        "\n• Photo: Celebration from Match A → Topic: Season success (that match was part of journey)"
        "\n• Photo: Crowd atmosphere → Topic: Fanbase growth (captures community feeling)"
        "\n• Photo: Manager tactical talk → Topic: Formation change (shows coaching approach)"
        "\n"
        "\n❌ AVOID CONTRADICTIONS:"
        "\n• Don't use defeat/struggle imagery for victory topics"
        "\n• Don't use Player A's photo for Player B's statistics (unless team context)"
        "\n• Don't use old venue for new venue topic (unless showing before/after)"
        "\n\n⚠️ CRITICAL REQUIREMENTS:"
        "\n1. IMAGE MATCHING IS TOP PRIORITY: Analyze images FIRST, then match facts to what you actually see"
        "\n2. BODY TEXT: 1-2 sentences, 120-180 characters - Provide meaningful detail while staying Instagram-friendly"
        "\n3. USE RESEARCHED DATA: Include specific numbers, scores, names from research"
        "\n4. NEVER CREATE PLACEHOLDERS: No ranges like (0-38), no meta-slides"
        "\n5. NO META-INSTRUCTIONS: Body text must be ACTUAL CONTENT for readers, never instructions like 'verify this' or 'update with confirmed data'"
        "\n   ❌ WRONG: 'The coaching-change claim needs confirmation; update this slide...'"
        "\n   ✅ RIGHT: 'Manager Takagi took charge in June 2025, leading the team to promotion'"
        "\n\nBODY TEXT EXAMPLES:"
        "\n✓ GOOD: 'Matheus Jesus scored 19 goals this season, leading J2 League in scoring and driving the promotion push with crucial late-season performances.' (143 chars)"
        "\n✓ GOOD: 'A devastating 0-3 home loss to Mito HollyHock exposed defensive issues and marked the low point of a difficult mid-season stretch.' (131 chars)"
        "\n❌ BAD: 'After a strong start, the team struggled' (TOO VAGUE, NO SPECIFIC DATA)"
        "\n❌ BAD: 'Goals (0-38), assists (0-38)' (PLACEHOLDER RANGES - FORBIDDEN)"
        "\n\nIMAGE MATCHING:"
        "\n• Celebration photo → Promotion/victory fact"
        "\n• Player close-up → That player's statistics"
        "\n• Stadium/crowd → Attendance/atmosphere fact"
        "\n• Action shot → Match result from that game"
        "\n• If no uploaded image fits a key fact → use external search"
        "\n\n⚠️ CRITICAL: For EVERY slide, you MUST provide 'reasoning' explaining why you chose that specific image"
        "\n• DESCRIBE WHAT YOU SEE: Start by stating EXACTLY what's in the image from your inventory"
        "\n  - 'Crowd of fans celebrating' NOT 'Manager celebrating' if you see crowd!"
        "\n  - 'Player in blue jersey #10' NOT vague 'player'"
        "\n  - 'Team photo of 11 players' NOT 'manager' if it's the team!"
        "\n• THEN EXPLAIN CONNECTION:"
        "\n  - DIRECT: 'Manager photo directly shows the coaching figure mentioned'"
        "\n  - CONTEXTUAL: 'Crowd celebration captures promotion atmosphere, though manager not visible'"
        "\n  - BACKGROUND: 'Stadium in background shows the venue being discussed'"
        "\n• BE HONEST: If what you SEE doesn't perfectly match, explain the contextual connection"
        "\n• User verification: Your description will be checked - don't claim to see things that aren't there!"
        "\n\nEXAMPLES OF GOOD DATA-DRIVEN CONTENT:"
        "\n✓ Title: 'Mid-Season Crisis' | Body: '0-3 loss to Mito HollyHock exposed defensive issues'"
        "\n✓ Title: 'Takagi Takes Charge' | Body: 'New manager sparked 15-game unbeaten run'"
        "\n✓ Title: 'Jesus Leads Attack' | Body: 'Matheus Jesus scored 19 goals - J2 top scorer'"
        "\n\nEXAMPLES OF BAD GENERIC CONTENT:"
        "\n❌ Title: 'Belief Becomes Momentum' (too vague)"
        "\n❌ Title: 'The Journey Continues' (no data)"
        "\n❌ Body: 'The team's success was remarkable' (no specifics)"
        "\n\nWhen research data is provided, EVERY slide must reference specific facts from it."
        "\n\n🌐 EXTERNAL IMAGE SEARCH IS AVAILABLE!"
        "\nUploaded images are PREFERRED but NOT required. You can request external images when:"
        "\n• No uploaded image matches a key fact (e.g., mascot slide but no mascot photo)"
        "\n• The topic requires a specific visual (e.g., logo/crest changes, historical photos)"
        "\n• A better image would make the story clearer"
        "\n\nTo use external search, set:"
        "\n  \"source\": \"external\","
        "\n  \"query\": \"descriptive search terms (e.g., 'V-Varen Nagasaki mascot Vivi-kun')\" "
        "\n\nDon't force poor matches! If you have a slide about the club mascot but only crowd photos,"
        "\nuse external search to find an actual mascot image."
        f"{placement_prompt}"
        "\n\nCRITICAL: ALL text content (titles, subtitles, kickers, body text) MUST be in English only."
        " Do NOT use any other language - output English text only for all slide content."
        "\n\n⚠️ DO NOT include source URLs or website names in slide text."
        "\n\nOutput ONLY valid JSON matching the schema."
    )

    schema = {
        "cover_slide": {
            "title": "main headline (short, bold, attention-grabbing)",
            "subtitle": "optional tagline or context (1 line)",
            "image": {
                "source": "upload or external",
                "filename": "best image for cover - most striking/representative",
                "query": "search query if external",
                "description": "REQUIRED: Describe what you see in this image (WHO: people/subjects visible, WHAT: action/scene, SETTING: location/context). Be specific and honest - if you can't identify someone, say so.",
                "reasoning": "REQUIRED: Explain why this image is best for the cover based on what you described above"
            },
            "centering": [0.5, 0.35],
            "layout": {
                "align": "center recommended for cover impact",
                "anchor": "mid for centered impact, or bottom for grounded look",
                "fade": "mid for dramatic center focus, or bottom for classic look"
            },
        },
        "slides": [
            {
                "kicker": "short label (e.g., 'Origins', 'Chapter 1')",
                "title": "short headline (under 60 chars)",
                "body": "Detailed yet scannable summary - 1-2 sentences, 120-180 characters with specific facts",
                "image": {
                    "source": "upload or external",
                    "filename": "use when source=upload (e.g., 'photo1.jpg')",
                    "query": "use when source=external (e.g., 'Tokyo skyline sunset')",
                    "description": "REQUIRED: Describe exactly what you see (WHO is in photo, WHAT they're doing, SETTING/context). Example: 'Team lineup photo before match - 11 players posing,No.10 visible in back row' or 'Crowd of fans waving flags - no individual subjects identifiable'",
                    "reasoning": "REQUIRED: Explain why this image matches this slide's content, referencing what you described above"
                },
                "centering": [0.5, 0.35],
                "layout": {
                    "align": "left or center or right - analyze image composition",
                    "anchor": "top or mid or bottom - avoid covering subjects",
                    "fade": "top or mid or bottom or none - match with anchor position"
                },
            }
        ]
    }

    user_text = (
        f"User intent: {intent}\n"
        f"Uploaded images: {', '.join(image_names) if image_names else 'none'}\n"
        f"External search allowed: {str(allow_external)}\n\n"
        "🔍 MANDATORY FIRST STEP - IMAGE INVENTORY:\n"
        "BEFORE planning any slides, you MUST create an inventory of ALL uploaded images.\n"
        "For EACH image, write a brief description identifying:\n"
        "• WHO: Manager/coach/player/referee/crowd/team? (Be specific!)\n"
        "• WHAT: Action/celebration/portrait/stadium/training?\n"
        "• SETTING: Match day/training/press conference/venue?\n"
        "\n"
        "Example format:\n"
        "Image 1 (photo1.jpg): Crowd of fans in stadium - no specific individuals visible\n"
        "Image 2 (photo2.jpg): Manager in suit celebrating - likely coach/manager\n"
        "Image 3 (photo3.jpg): Player #10 in action during match\n"
        "\n"
        "⚠️ BE HONEST: If you can't clearly identify WHO is in the photo, say so!\n"
        "❌ WRONG: Claiming crowd photo shows 'manager celebrating'\n"
        "✅ RIGHT: 'Crowd/fans celebrating - no clear individual subject'\n"
        "\n"
    )

    # Add research data if available
    if research_data:
        user_text += (
            "=" * 60 + "\n"
            "RESEARCHED KEY MOMENTS - TURN THESE INTO SLIDES\n"
            "=" * 60 + "\n"
            f"{research_data}\n"
            "=" * 60 + "\n\n"
            "MANDATORY: USE THE RESEARCHED DATA IN YOUR SLIDES\n"
            "=" * 60 + "\n"
            "1. Turn EACH key moment above into ONE content slide"
            "2. Use the EXACT facts, numbers, and names from the research"
            "3. DO NOT create placeholder slides (no ranges like '0-38', '1-20')"
            "4. DO NOT create meta-slides about what data is needed"
            "5. Use ACTUAL researched values (e.g., '19 goals', '0-3 loss', '2nd place')"
            "\n\nEXAMPLE - CORRECT:"
            "\nResearch: 'Matheus Jesus scored 19 goals - J2 top scorer'"
            "\n→ Slide: {"
            '\n    "kicker": "TOP SCORER",'
            '\n    "title": "Matheus Jesus: 19 Goals",'
            '\n    "body": "Led J2 League in scoring - crucial to promotion push"'
            "\n  }"
            "\n\nEXAMPLE - WRONG (DO NOT DO THIS):"
            "\n❌ Slide: {"
            '\n    "title": "Forward #10: Matheus Jesus",'
            '\n    "body": "goals (0-38), assists (0-38)" <- NO PLACEHOLDERS!'
            "\n  }"
            "\n\n❌ FORBIDDEN:"
            "\n• Placeholder ranges: (0-38), (1-20), (0-114)"
            "\n• Meta-instructions: 'Fill these with...', 'Use this slide to...'"
            "\n• Data requests: 'verified stats needed', 'data placeholders'"
            "\n\n✅ REQUIRED:"
            "\n• Actual numbers: '19 goals', '63 goals scored', '2nd place'"
            "\n• Specific matches: '0-3 loss to Mito HollyHock'"
            "\n• Real events: 'Manager changed from Shimotaira to Takagi'"
            "\n\n⚠️ If research data is incomplete, use what you have - never create placeholder slides.\n\n"
        )

    user_text += (
        "TASK:\n"
        "1. Create a COVER SLIDE first - the opening/title slide that introduces the carousel\n"
        "   - Choose the most striking/impactful image for the cover\n"
        "   - Title should be bold, attention-grabbing, main headline (3-8 words)\n"
        "   - Subtitle is optional but recommended for context (1 line)\n"
        "   - Use center alignment for maximum impact\n"
        "   - Use mid or bottom anchor for dramatic positioning\n"
        "2. Then create 3-6 content slides for the story\n"
        "3. CRITICAL - BODY TEXT LENGTH:\n"
        "   - Body text should be DETAILED yet SCANNABLE: 1-2 sentences, 120-180 characters\n"
        "   - Include specific facts, numbers, and context for meaningful storytelling\n"
        "   - Example GOOD: 'Matheus Jesus scored 19 goals this season, leading J2 League and driving the promotion push with crucial performances.' (120 chars)\n"
        "   - Example BAD: 'Top scorer' (TOO SHORT - lacks detail and context)\n"
        "   - Example BAD: 'After a great start to the season, the team went through many ups and downs with various players contributing to wins and losses throughout the campaign...' (TOO LONG - over 180 chars)\n"
        "   - Example TERRIBLE: 'The coaching change needs verification; update with confirmed details' (META-INSTRUCTION - NOT ACTUAL CONTENT!)\n"
        "   - Balance detail with scannability - Instagram users appreciate substance but won't read paragraphs\n"
        "   - CRITICAL: Body text is what users READ on the carousel - it must be informative content, NOT instructions to verify or update!\n"
        "4. CRITICAL - MATCH FACTS TO IMAGES (THINK LIKE A FAN!):\n"
        "   - STEP 1: Create image inventory (WHO/WHAT in each photo) - see instructions above\n"
        "   - STEP 2: For each fact/slide, reference your inventory to find BEST matching image\n"
        "   - STEP 3: Verify the WHO matches: Manager topic → Use image where you identified manager, NOT crowd!\n"
        "   - STEP 4: If no good match exists in inventory, use external search\n"
        "\n"
        "   DIRECT MATCHING (Obvious):\n"
        "   - Player X stats → Photo of Player X\n"
        "   - Manager change → Photo of manager\n"
        "   - Specific match result → Photo from that match\n"
        "\n"
        "   CONTEXTUAL MATCHING (Nuanced - fans will understand):\n"
        "   - Stadium topic → Any photo showing that stadium (even if focus is player)\n"
        "   - Season success → Key match celebration (that match was part of the journey)\n"
        "   - Promotion story → Any photo from promotion season (captures the era)\n"
        "   - Fanbase growth → Crowd/atmosphere photo (shows community spirit)\n"
        "\n"
        "   ⚠️ QUALITY CHECK:\n"
        "   - Does this image tell the story? (Consider both direct and background connections)\n"
        "   - Would fans recognize the connection? (Context matters!)\n"
        "   - Is there a contradiction? (Don't use defeat imagery for victory topics)\n"
        "5. Make the story clear and social-friendly\n"
        "6. Prioritize uploaded images, but use external search when needed for specific facts\n"
        "\n"
        "IMPORTANT - TEXT PLACEMENT ANALYSIS:\n"
        "For each uploaded image, carefully analyze:\n"
        "- Where are the main subjects/people located?\n"
        "- Which areas are busy with detail vs. simple/empty?\n"
        "- Where would text be most readable without covering important elements?\n"
        "- How can you create visual variety across the 3-6 slides?\n"
        "\n"
        "Choose layout (align, anchor, fade) that:\n"
        "✓ Avoids covering faces and key subjects\n"
        "✓ Uses empty/simple areas of the image\n"
        "✓ Creates visual balance and breathing room\n"
        "✓ Varies placement across slides (mix left/center/right and top/mid/bottom)\n"
        "\n"
        f"JSON schema example: {json.dumps(schema)}"
    )

    # Build content with text and images (limit to avoid request size errors)
    content = [{"type": "text", "text": user_text}]

    # Limit to 8 images max to avoid OpenAI request size limits
    # Images are resized to 1024px before encoding to further reduce payload
    max_images = 8
    images_to_send = image_paths[:max_images]

    if log:
        if len(image_paths) > max_images:
            log(f"→ Limiting to {max_images} images (of {len(image_paths)}) to avoid API limits")
        log(f"→ Resizing images to 1024px max for optimal API performance")

    for path in images_to_send:
        mime, b64 = _encode_image_b64(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    if log:
        log(f"→ Calling AI ({OPENAI_MODEL}) to generate slide plan...")

    text = _call_openai(client, messages, response_format={"type": "json_object"})

    if not text:
        if log:
            log("❌ Failed to get response from AI")
        return _default_plan(intent, image_names)

    try:
        plan = json.loads(text)
    except Exception:
        if log:
            log("❌ Failed to parse AI response, using default plan")
        return _default_plan(intent, image_names)

    if not isinstance(plan, dict) or "slides" not in plan:
        if log:
            log("⚠️ Invalid plan structure, using default plan")
        return _default_plan(intent, image_names)

    slides = plan.get("slides") or []
    cover_slide = plan.get("cover_slide")

    if log:
        log(f"✓ Slide plan generated successfully")
        if cover_slide:
            log(f"  • Cover slide: \"{cover_slide.get('title', 'N/A')}\"")
        log(f"  • Content slides: {len(slides)}")

    if len(slides) > MAX_SLIDES:
        if log:
            log(f"→ Trimming to {MAX_SLIDES} slides (max limit)")
        plan["slides"] = slides[:MAX_SLIDES]

    if log:
        log("")
        log("🔍 STEP 3: Quality Control Validation")

    # QC Step: Validate and fix issues before finalizing
    validated_plan = _qc_slide_plan(intent, image_paths, plan, placement_rules, log=log)

    if log:
        log("")
        log("✓ Slide planning complete")

    return validated_plan


def _ensure_photos_dir(run_dir: Path) -> Path:
    photos = run_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    return photos


def _copy_uploads(uploads: list[str], photos_dir: Path) -> list[str]:
    names: list[str] = []
    for p in uploads:
        src = Path(p)
        dst = photos_dir / src.name
        shutil.copy2(src, dst)
        names.append(src.name)
    return names


def _validate_downloaded_image(path: Path, min_size: int = 800, log=None) -> bool:
    """Validate a downloaded image for integrity and minimum dimensions.

    Args:
        path: Path to the downloaded image
        min_size: Minimum width and height in pixels (default 800)
        log: Optional logging function

    Returns:
        True if image is valid, False otherwise
    """
    from PIL import Image

    try:
        # First check file integrity
        with Image.open(path) as img:
            img.verify()  # Verify file integrity

        # Re-open to check dimensions (verify() closes the file)
        with Image.open(path) as img:
            width, height = img.size
            if width < min_size or height < min_size:
                if log:
                    log(f"  ⚠️ Image too small: {width}x{height} (min {min_size}x{min_size})")
                return False

        return True
    except Exception as e:
        if log:
            log(f"  ⚠️ Image validation failed: {e}")
        return False


def _fetch_external_images(slides: list[dict[str, Any]], photos_dir: Path, log=None) -> dict[int, str]:
    replacements: dict[int, str] = {}
    for idx, s in enumerate(slides):
        img = s.get("image") or {}
        if img.get("source") != "external":
            continue
        query = img.get("query") or ""
        if not query.strip():
            continue
        if log:
            log(f"Searching external images: {query}")
        # Search the entire web - no domain restrictions
        urls = search_images(query, allowed_domains=None)
        if not urls:
            continue

        # Try up to 3 URLs to find a valid image
        max_attempts = min(3, len(urls))
        for attempt, url in enumerate(urls[:max_attempts]):
            if log:
                log(f"Downloading image ({attempt + 1}/{max_attempts}): {url}")
            ext = os.path.splitext(url)[-1]
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            filename = f"external_{idx+1}{ext}"
            out_path = photos_dir / filename

            if download_image(url, str(out_path)):
                # Validate the downloaded image
                if _validate_downloaded_image(out_path, min_size=800, log=log):
                    replacements[idx] = filename
                    break
                else:
                    # Delete invalid image and try next URL
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                    if log and attempt < max_attempts - 1:
                        log(f"  → Trying next URL...")
            else:
                if log and attempt < max_attempts - 1:
                    log(f"  → Download failed, trying next URL...")
    return replacements


def _verify_image_descriptions(plan: dict[str, Any], uploads: list[str], log=None) -> dict[str, Any]:
    """
    Stage 1: Verify image descriptions are accurate (double-check with second AI call).
    Catches misidentifications like "player #10" when it's actually "coach/manager".
    """
    if log:
        log("🔍 Stage 1: Verifying image descriptions...")

    client = OpenAI()

    # Collect all images that need verification
    images_to_verify = []

    # Cover slide
    cover = plan.get("cover_slide")
    if cover:
        img = cover.get("image", {})
        if img.get("description"):
            images_to_verify.append({
                "slide": "Cover",
                "description": img.get("description"),
                "filename": img.get("filename", "")
            })

    # Content slides
    for i, slide in enumerate(plan.get("slides", []), 1):
        img = slide.get("image", {})
        if img.get("description"):
            images_to_verify.append({
                "slide": i,
                "description": img.get("description"),
                "filename": img.get("filename", "")
            })

    if not images_to_verify:
        return plan

    # Load and encode images for second opinion
    image_data_list = []
    for item in images_to_verify:
        filename = item["filename"]
        # Find the actual image file
        image_path = None
        for upload_path in uploads:
            if Path(upload_path).name == filename:
                image_path = upload_path
                break

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
                image_data_list.append({
                    "slide": item["slide"],
                    "original_description": item["description"],
                    "base64": base64_image,
                    "filename": filename
                })

    if not image_data_list:
        return plan

    # Verify images ONE AT A TIME to avoid payload size limits
    all_results = []
    inaccurate_count = 0
    issues = []

    for item in image_data_list:
        verification_prompt = (
            "You are verifying an image description for accuracy.\n\n"
            f"Original description: \"{item['original_description']}\"\n\n"
            "Your job: Check if this description ACCURATELY describes what's in the image.\n\n"
            "Common errors to catch:\n"
            "❌ Claiming 'player #10' when image shows coach/manager\n"
            "❌ Claiming 'action shot' when image shows celebration/portrait\n"
            "❌ Claiming specific people when only crowd visible\n"
            "❌ Claiming 'controlling ball' when person is celebrating\n\n"
            "Respond with JSON:\n"
            '{"accurate": true/false, "issue": "explanation if inaccurate", "corrected": "accurate description if needed"}\n'
        )

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": verification_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item['base64']}"}}
                    ]
                }],
            )

            result_text = response.choices[0].message.content
            all_results.append(f"Slide {item['slide']}: {result_text}")

            # Parse result and APPLY CORRECTIONS
            try:
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    if not result_data.get("accurate", True):
                        inaccurate_count += 1
                        corrected_desc = result_data.get('corrected', '')
                        issue_text = f"Slide {item['slide']}: {result_data.get('issue', 'Inaccurate')}"
                        if corrected_desc:
                            issue_text += f" → {corrected_desc}"
                            # ACTUALLY FIX the description in the plan
                            slide_key = item['slide']
                            if slide_key == "Cover":
                                if plan.get("cover_slide", {}).get("image"):
                                    plan["cover_slide"]["image"]["description"] = corrected_desc
                                    if log:
                                        log(f"  ✓ Fixed Cover slide description")
                            else:
                                slide_idx = int(slide_key) - 1
                                if 0 <= slide_idx < len(plan.get("slides", [])):
                                    if plan["slides"][slide_idx].get("image"):
                                        plan["slides"][slide_idx]["image"]["description"] = corrected_desc
                                        if log:
                                            log(f"  ✓ Fixed Slide {slide_key} description")
                        issues.append(issue_text)
            except:
                pass

        except Exception as item_err:
            if log:
                log(f"  ⚠️ Failed to verify slide {item['slide']}: {item_err}")

    verification_result = "\n".join(all_results)

    if log:
        if inaccurate_count > 0:
            log(f"⚠️ Stage 1: {inaccurate_count} descriptions corrected")
        else:
            log("✓ Stage 1: All image descriptions verified accurate")

    # Store verification results
    plan["_description_verification"] = {
        "result": verification_result,
        "inaccurate_count": inaccurate_count,
        "issues": issues
    }

    return plan


def _validate_image_matches(plan: dict[str, Any], uploads: list[str] = None, photos_dir: Path = None, log=None) -> dict[str, Any]:
    """
    Stage 2: Validate that image descriptions match slide content WITH ACTUAL IMAGES.
    Uses AI to check if selected images actually align with text by examining real pixels.
    Returns updated plan with validation notes.

    Args:
        plan: The slide plan with image assignments
        uploads: List of original upload paths (for finding images)
        photos_dir: Directory where photos are stored (for finding images)
        log: Optional logging function
    """
    if log:
        if uploads or photos_dir:
            log("🔍 Stage 2: Validating image-text matches with actual images...")
        else:
            log("🔍 Stage 2: Validating image-text matches (descriptions only)...")

    client = OpenAI()

    # Build list of slides to validate
    slides_to_validate = []

    # Check cover slide
    cover = plan.get("cover_slide")
    if cover:
        img = cover.get("image", {})
        slides_to_validate.append({
            "slide_num": "Cover",
            "title": cover.get("title", ""),
            "body": cover.get("subtitle", ""),
            "image_description": img.get("description", ""),
            "image_reasoning": img.get("reasoning", ""),
            "filename": img.get("filename", ""),
        })

    # Check content slides
    for i, slide in enumerate(plan.get("slides", []), 1):
        img = slide.get("image", {})
        slides_to_validate.append({
            "slide_num": i,
            "kicker": slide.get("kicker", ""),
            "title": slide.get("title", ""),
            "body": slide.get("body", ""),
            "image_description": img.get("description", ""),
            "image_reasoning": img.get("reasoning", ""),
            "filename": img.get("filename", ""),
        })

    # Process ONE slide at a time to avoid payload limits when using actual images
    all_results = []
    weak_count = 0
    bad_count = 0
    issues = []

    for slide_data in slides_to_validate:
        # Try to find and encode the actual image
        image_b64 = None
        filename = slide_data.get("filename", "")

        if filename and (uploads or photos_dir):
            image_path = None

            # Try to find in uploads
            if uploads:
                for upload_path in uploads:
                    if Path(upload_path).name == filename:
                        image_path = upload_path
                        break

            # Try to find in photos_dir
            if not image_path and photos_dir:
                potential_path = photos_dir / filename
                if potential_path.exists():
                    image_path = str(potential_path)

            # Encode the image if found
            if image_path and os.path.exists(image_path):
                try:
                    mime, b64 = _encode_image_b64(image_path, max_size=512)  # Smaller for validation
                    image_b64 = f"data:{mime};base64,{b64}"
                except Exception as e:
                    if log:
                        log(f"  ⚠️ Failed to encode image for slide {slide_data['slide_num']}: {e}")

        # Build validation prompt for this slide
        validation_prompt = (
            "You are a STRICT validator checking if an image matches the slide text for an Instagram carousel.\n\n"
        )

        if image_b64:
            validation_prompt += (
                "I'm providing you with the ACTUAL IMAGE. Look at it carefully and verify:\n"
                "1. Does the image description accurately describe what's in the image?\n"
                "2. Does the image content match what the slide text is about?\n\n"
            )
        else:
            validation_prompt += (
                "Based on the image description provided, verify if the image matches the slide text.\n\n"
            )

        validation_prompt += (
            "Validation rules:\n"
            "✓ GOOD: Image clearly shows the subject mentioned in text\n"
            "✓ CONTEXTUAL: Image shows related scene that MAKES SENSE for the topic. Examples:\n"
            "   - Man in suit/business attire + ownership/business story → CONTEXTUAL (good match)\n"
            "   - Executive/manager celebrating + company takeover story → CONTEXTUAL (good match)\n"
            "   - Crowd celebrating + team success story → CONTEXTUAL (good match)\n"
            "   - Stadium/venue shot + stadium facts → CONTEXTUAL (good match)\n"
            "⚠️ WEAK: Image is too generic OR you cannot verify the connection (e.g., random crowd for a specific person's story)\n"
            "❌ BAD: Image clearly CONTRADICTS the text (e.g., defeat image for victory story, wrong team/venue)\n\n"
            "BE REASONABLE - think like a fan:\n"
            "- Man in suit at stadium + ownership story → CONTEXTUAL (business figure fits the narrative)\n"
            "- Crowd shot + fan/atmosphere story → GOOD\n"
            "- Player action shot + match result story → CONTEXTUAL\n"
            "- Generic crowd + specific person's biography → WEAK (no clear connection)\n"
            "- Image of Team A + story about Team B → BAD (contradiction)\n\n"
            f"Slide {slide_data['slide_num']}:\n"
            f"  Title: {slide_data['title']}\n"
            f"  Body: {slide_data['body']}\n"
            f"  Image description: {slide_data['image_description']}\n"
            f"  AI reasoning: {slide_data['image_reasoning']}\n\n"
            "Respond with ONLY JSON:\n"
            '{"status": "good|contextual|weak|bad", "issue": "explanation if weak/bad"}\n'
        )

        try:
            # Build message content
            content = []
            content.append({"type": "text", "text": validation_prompt})

            if image_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_b64}
                })

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": content}],
            )

            result_text = response.choices[0].message.content
            all_results.append(f"Slide {slide_data['slide_num']}: {result_text}")

            # Parse result
            try:
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    status = result_data.get("status", "").lower()
                    issue_text = result_data.get("issue", "")
                    if status == "weak":
                        weak_count += 1
                        issues.append({
                            "slide_num": slide_data['slide_num'],
                            "status": "WEAK",
                            "issue": issue_text,
                            "title": slide_data['title'],
                            "body": slide_data.get('body', ''),
                        })
                    elif status == "bad":
                        bad_count += 1
                        issues.append({
                            "slide_num": slide_data['slide_num'],
                            "status": "BAD",
                            "issue": issue_text,
                            "title": slide_data['title'],
                            "body": slide_data.get('body', ''),
                        })
            except Exception:
                pass

        except Exception as e:
            if log:
                log(f"  ⚠️ Failed to validate slide {slide_data['slide_num']}: {e}")

    validation_result = "\n".join(all_results)

    if log:
        if bad_count > 0:
            log(f"⚠️ Validation: {bad_count} BAD image-text matches found")
            for issue in [i for i in issues if i.get("status") == "BAD"]:
                log(f"  Slide {issue['slide_num']}: BAD - {issue['issue']}")
        elif weak_count > 0:
            log(f"⚠️ Validation: {weak_count} WEAK image-text matches found")
            for issue in [i for i in issues if i.get("status") == "WEAK"]:
                log(f"  Slide {issue['slide_num']}: WEAK - {issue['issue']}")
        else:
            log("✓ Validation: All image-text matches look good")

    # Store validation in plan metadata
    plan["_validation"] = {
        "result": validation_result,
        "weak_count": weak_count,
        "bad_count": bad_count,
        "issues": issues
    }

    # Flag slides with BAD matches for external image search
    # This will trigger searching for better matching images
    bad_slides_for_search = []
    bad_slide_nums = set()

    for issue in issues:
        if issue.get("status") == "BAD":
            slide_num = issue['slide_num']
            bad_slide_nums.add(str(slide_num))

            # Build search query from slide content
            title = issue.get('title', '')
            body = issue.get('body', '')
            search_query = f"{title} {body}".strip()[:100]  # Limit query length

            bad_slides_for_search.append({
                "slide_num": slide_num,
                "search_query": search_query,
                "issue": issue['issue'],
            })

    if bad_slides_for_search:
        # Mark these slides for external image search
        plan["_needs_external_images"] = bad_slides_for_search
        plan["_conservative_layout_slides"] = list(bad_slide_nums)
        if log:
            log(f"  → Flagged {len(bad_slides_for_search)} slides for external image search")

    return plan


def _build_config(
    intent: str,
    run_dir: Path,
    uploads: list[str],
    allow_external: bool,
    log=None,
) -> dict[str, Any]:
    photos_dir = _ensure_photos_dir(run_dir)
    upload_names = _copy_uploads(uploads, photos_dir)

    # Analyze images for optimal text placement (with split layout info)
    if log:
        log("🔍 Analyzing images for optimal text placement...")
    image_paths = [str(photos_dir / name) for name in upload_names]
    placement_map = analyze_all_images(image_paths, return_metadata=True)
    if log:
        log(f"✓ Analyzed {len(placement_map)} images for text placement")

    plan = plan_slides(intent, uploads, allow_external=allow_external, log=log)

    # Two-stage validation
    # Stage 1: Verify image descriptions are accurate (catch misidentifications)
    plan = _verify_image_descriptions(plan, uploads, log=log)

    # Stage 2: Validate image-text matches with actual images
    plan = _validate_image_matches(plan, uploads=uploads, photos_dir=photos_dir, log=log)

    # Extract cover slide and content slides
    cover_slide_in = plan.get("cover_slide")
    slides_in = plan.get("slides") or []

    # Fetch external images for both cover and content slides
    external_map: dict[int, str] = {}
    cover_external_filename = None

    # Check if there are BAD matches that need replacement
    needs_external = plan.get("_needs_external_images", [])
    if needs_external and not allow_external:
        if log:
            log(f"⚠️ {len(needs_external)} slides have BAD image-text matches but external search is disabled")
            log(f"   Enable external search to automatically fix these mismatches")

    if allow_external:
        # Fetch cover slide image if needed
        if cover_slide_in:
            img = cover_slide_in.get("image") or {}
            if img.get("source") == "external":
                query = img.get("query") or ""
                if query.strip():
                    if log:
                        log(f"Searching cover image: {query}")
                    urls = search_images(query, allowed_domains=None)
                    if urls:
                        # Try up to 3 URLs to find a valid image
                        max_attempts = min(3, len(urls))
                        for attempt, url in enumerate(urls[:max_attempts]):
                            if log:
                                log(f"Downloading cover image ({attempt + 1}/{max_attempts}): {url}")
                            ext = os.path.splitext(url)[-1]
                            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                                ext = ".jpg"
                            filename = f"cover_image{ext}"
                            out_path = photos_dir / filename
                            if download_image(url, str(out_path)):
                                # Validate the downloaded image
                                if _validate_downloaded_image(out_path, min_size=800, log=log):
                                    cover_external_filename = filename
                                    break
                                else:
                                    # Delete invalid image and try next URL
                                    try:
                                        out_path.unlink()
                                    except Exception:
                                        pass
                                    if log and attempt < max_attempts - 1:
                                        log(f"  → Trying next URL...")
                            else:
                                if log and attempt < max_attempts - 1:
                                    log(f"  → Download failed, trying next URL...")

        # Fetch content slides images
        external_map = _fetch_external_images(slides_in, photos_dir, log=log)

        # Fetch REPLACEMENT images for slides with BAD image-text matches
        needs_external = plan.get("_needs_external_images", [])
        replacement_map: dict[int, str] = {}

        if needs_external:
            if log:
                log(f"🔄 Fetching replacement images for {len(needs_external)} BAD matches...")

            for item in needs_external:
                slide_num = item["slide_num"]
                search_query = item["search_query"]

                # For content slides, slide_num is 1-indexed in validation
                # but 0-indexed in slides_in array
                if slide_num == "Cover":
                    # Handle cover slide replacement
                    if log:
                        log(f"  → Searching replacement for Cover: {search_query}")
                    urls = search_images(search_query, allowed_domains=None)
                    if urls:
                        for attempt, url in enumerate(urls[:3]):
                            ext = os.path.splitext(url)[-1]
                            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                                ext = ".jpg"
                            filename = f"cover_replacement{ext}"
                            out_path = photos_dir / filename
                            if download_image(url, str(out_path)):
                                if _validate_downloaded_image(out_path, min_size=800, log=log):
                                    cover_external_filename = filename
                                    if log:
                                        log(f"    ✓ Found replacement for Cover")
                                    break
                                else:
                                    try:
                                        out_path.unlink()
                                    except Exception:
                                        pass
                else:
                    # Handle content slide replacement
                    slide_idx = int(slide_num) - 1  # Convert to 0-indexed
                    if log:
                        log(f"  → Searching replacement for Slide {slide_num}: {search_query}")
                    urls = search_images(search_query, allowed_domains=None)
                    if urls:
                        for attempt, url in enumerate(urls[:3]):
                            ext = os.path.splitext(url)[-1]
                            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                                ext = ".jpg"
                            filename = f"replacement_{slide_num}{ext}"
                            out_path = photos_dir / filename
                            if download_image(url, str(out_path)):
                                if _validate_downloaded_image(out_path, min_size=800, log=log):
                                    replacement_map[slide_idx] = filename
                                    if log:
                                        log(f"    ✓ Found replacement for Slide {slide_num}")
                                    break
                                else:
                                    try:
                                        out_path.unlink()
                                    except Exception:
                                        pass

        # Merge replacement_map into external_map (replacements override)
        external_map.update(replacement_map)

        # Analyze newly downloaded external images
        new_images = []
        if cover_external_filename:
            new_images.append(str(photos_dir / cover_external_filename))
        for filename in external_map.values():
            if filename:
                new_images.append(str(photos_dir / filename))

        if new_images:
            if log:
                log(f"🔍 Analyzing {len(new_images)} external images for text placement...")
            new_placements = analyze_all_images(new_images, return_metadata=True)
            placement_map.update(new_placements)

    slides: list[dict[str, Any]] = []
    overrides: dict[int, list[str]] = {}

    # Get conservative layout flags from validation
    conservative_slides = set(plan.get("_conservative_layout_slides", []))

    # Process cover slide first (becomes slide 1)
    if cover_slide_in:
        img = cover_slide_in.get("image") or {}
        source = img.get("source")
        filename = None

        if source == "upload":
            filename = img.get("filename")
        elif source == "external" and cover_external_filename:
            filename = cover_external_filename

        if not filename and upload_names:
            # Pick the best/first uploaded image for cover
            filename = upload_names[0]

        # Use intelligent placement based on image analysis
        centering = cover_slide_in.get("centering")
        subject_region = None
        if filename and filename in placement_map:
            metadata = placement_map[filename]
            subject_region = metadata.get('subject_region')
            if not centering:
                centering = list(metadata['placement'])
        if not centering:
            centering = [0.5, 0.35]  # Fallback to default

        # Preserve image description and reasoning if provided
        image_data = cover_slide_in.get("image") or {}
        image_description = image_data.get("description", "")
        image_reasoning = image_data.get("reasoning", "")

        cover_slide = {
            "photo": filename,
            "kicker": None,  # No kicker for cover
            "title": cover_slide_in.get("title") or intent,
            "body": cover_slide_in.get("subtitle"),  # Subtitle becomes body
            "centering": centering,
            "subject_region": subject_region,  # For render-time split decision
            "image_description": image_description,  # Preserve description
            "image_reasoning": image_reasoning,  # Preserve reasoning
        }
        slides.append(cover_slide)

        # Store layout override for cover (slide 1)
        layout = cover_slide_in.get("layout") or {}
        align = layout.get("align") or "center"
        fade = layout.get("fade") or "mid"

        # Use image analysis to determine anchor position instead of AI suggestion
        anchor = "mid"  # default
        if filename and filename in placement_map:
            metadata = placement_map[filename]
            y_pos = metadata['placement'][1]
            if y_pos < 0.4:
                anchor = "top"
            elif y_pos > 0.6:
                anchor = "bottom"
            else:
                anchor = "mid"

        # Force conservative layout (bottom anchor) for flagged slides
        if "Cover" in conservative_slides or "1" in conservative_slides:
            anchor = "bottom"
            if log:
                log(f"  → Cover slide: forcing bottom anchor (flagged for conservative layout)")

        # Sync fade with anchor to avoid forbidden pairs (e.g., anchor=top + fade=bottom)
        fade = anchor

        overrides[1] = [align, anchor, fade]

    # Process content slides (become slides 2, 3, 4, etc.)
    slide_offset = 1 if cover_slide_in else 0  # Offset by 1 if we have a cover slide

    for i, s in enumerate(slides_in):
        img = s.get("image") or {}
        source = img.get("source")
        filename = None

        # Check for REPLACEMENT image first (from BAD match fix)
        # This takes priority over original assignment
        if i in external_map:
            filename = external_map[i]
            if log and filename and filename.startswith("replacement_"):
                log(f"  → Slide {i+1}: using replacement image {filename}")
        elif source == "upload":
            filename = img.get("filename")
        elif source == "external":
            filename = external_map.get(i)

        if not filename and upload_names:
            filename = upload_names[i % len(upload_names)]

        # Use intelligent placement based on image analysis
        centering = s.get("centering")
        subject_region = None
        if filename and filename in placement_map:
            metadata = placement_map[filename]
            subject_region = metadata.get('subject_region')
            if not centering:
                centering = list(metadata['placement'])
        if not centering:
            centering = [0.5, 0.35]  # Fallback to default

        # Preserve image description and reasoning if provided
        image_data = s.get("image") or {}
        image_description = image_data.get("description", "")
        image_reasoning = image_data.get("reasoning", "")

        slide = {
            "photo": filename,
            "kicker": s.get("kicker"),
            "title": s.get("title") or intent,
            "body": s.get("body"),
            "centering": centering,
            "subject_region": subject_region,  # For render-time split decision
            "image_description": image_description,  # Preserve description
            "image_reasoning": image_reasoning,  # Preserve reasoning
        }
        slides.append(slide)

        layout = s.get("layout") or {}
        align = layout.get("align") or "center"
        fade = layout.get("fade") or "mid"

        # Use image analysis to determine anchor position instead of AI suggestion
        anchor = "mid"  # default
        if filename and filename in placement_map:
            metadata = placement_map[filename]
            y_pos = metadata['placement'][1]
            if y_pos < 0.4:
                anchor = "top"
            elif y_pos > 0.6:
                anchor = "bottom"
            else:
                anchor = "mid"

        # Force conservative layout (bottom anchor) for flagged slides
        # Content slide numbers in validation are 1-indexed
        content_slide_num = str(i + 1)
        if content_slide_num in conservative_slides:
            anchor = "bottom"
            if log:
                log(f"  → Slide {content_slide_num}: forcing bottom anchor (flagged for conservative layout)")

        # Sync fade with anchor to avoid forbidden pairs (e.g., anchor=top + fade=bottom)
        fade = anchor

        # Add offset to slide number (2, 3, 4, etc. if we have cover)
        overrides[i + 1 + slide_offset] = [align, anchor, fade]

    output_dir = run_dir / "output"

    config = {
        "photos_dir": str(photos_dir),
        "output_dir": str(output_dir),
        "slides": slides,
        "overrides": overrides,
    }
    return config


def run_renderer(config: dict[str, Any], run_dir: Path, log=None) -> None:
    cfg_path = run_dir / "run_config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["IG_CAROUSEL_CONFIG"] = str(cfg_path)

    num_slides = len(config.get("slides", []))

    if log:
        log(f"→ Rendering {num_slides} slides with Skia graphics engine...")

    result = subprocess.run(["python3", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    if result.returncode != 0:
        if log:
            log(f"❌ Rendering failed: {result.stderr or result.stdout}")
        raise RuntimeError(result.stderr or result.stdout or "Renderer failed")

    if log:
        log(f"✓ All {num_slides} slides rendered successfully")


def generate_carousel(
    intent: str,
    uploads: list[str],
    allow_external: bool,
    workdir: str | None = None,
    run_id: str | None = None,
    log=None,
) -> dict[str, Any]:
    run_id = run_id or uuid.uuid4().hex[:10]
    base_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ig-carousel-"))
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log("")
        log("=" * 50)
        log("🚀 STARTING CAROUSEL GENERATION")
        log("=" * 50)
        log(f"Run ID: {run_id}")
        log(f"Uploaded images: {len(uploads)}")

    config = _build_config(intent, run_dir, uploads, allow_external=allow_external, log=log)

    if log:
        log("")
        log("🎨 STEP 4: Rendering Slides")

    run_renderer(config, run_dir, log=log)

    output_dir = Path(config["output_dir"])
    outputs = sorted([p.name for p in output_dir.glob("*.jpg")])

    if log:
        log("")
        log("=" * 50)
        log(f"✅ COMPLETE! Generated {len(outputs)} slides")
        log("=" * 50)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "outputs": outputs,
        "config": config,
    }


def generate_slide_plan(
    intent: str,
    uploads: list[str],
    allow_external: bool,
    workdir: str | None = None,
    run_id: str | None = None,
    log=None,
) -> dict[str, Any]:
    """Phase 1: Generate slide structure/plan only (no rendering)"""
    run_id = run_id or uuid.uuid4().hex[:10]
    base_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ig-carousel-"))
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log("")
        log("=" * 50)
        log("🚀 GENERATING SLIDE PLAN")
        log("=" * 50)
        log(f"Run ID: {run_id}")
        log(f"Uploaded images: {len(uploads)}")

    config = _build_config(intent, run_dir, uploads, allow_external=allow_external, log=log)

    if log:
        log("")
        log("=" * 50)
        log(f"✅ PLAN READY! {len(config['slides'])} slides structured")
        log("=" * 50)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": config,
        "intent": intent,
        "allow_external": allow_external,
        "uploads": uploads,
    }


def render_from_plan(
    plan: dict[str, Any],
    workdir: str | None = None,
    run_id: str | None = None,
    log=None,
) -> dict[str, Any]:
    """Phase 2: Render graphics from an approved plan"""
    run_id = run_id or plan.get("run_id") or uuid.uuid4().hex[:10]
    run_dir = Path(plan["run_dir"]) if "run_dir" in plan else Path(workdir) / run_id
    config = plan["config"]

    if log:
        log("")
        log("🎨 STEP 4: Rendering Slides")

    run_renderer(config, run_dir, log=log)

    output_dir = Path(config["output_dir"])
    outputs = sorted([p.name for p in output_dir.glob("*.jpg")])

    if log:
        log("")
        log("=" * 50)
        log(f"✅ COMPLETE! Generated {len(outputs)} slides")
        log("=" * 50)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "outputs": outputs,
        "config": config,
    }
