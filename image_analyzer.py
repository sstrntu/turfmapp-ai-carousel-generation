"""
Intelligent image analysis for optimal text placement.

Analyzes images to determine the best position for text overlays based on:
- Background brightness/darkness (prefer dark areas for white text)
- Subject location (avoid covering important subjects)
- Visual complexity (prefer simpler areas)
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def analyze_brightness_regions(image_path: str) -> dict[str, float]:
    """
    Analyze brightness of different regions in the image.

    Returns dict with average brightness (0-255) for each region:
    - top: top 33% of image
    - middle: middle 33% of image
    - bottom: bottom 33% of image
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    arr = np.array(img)

    height = arr.shape[0]
    third = height // 3

    top_region = arr[:third, :]
    middle_region = arr[third:2*third, :]
    bottom_region = arr[2*third:, :]

    return {
        'top': float(np.mean(top_region)),
        'middle': float(np.mean(middle_region)),
        'bottom': float(np.mean(bottom_region))
    }


def analyze_subject_location(image_path: str) -> Optional[str]:
    """
    Detect where the main subject is located using simple edge detection.

    Returns 'top', 'middle', or 'bottom' based on where most visual activity is.
    This is a simple heuristic - more complex logic could use face detection or OpenAI Vision.
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img)

    # Simple edge detection using gradient
    gradient_y = np.abs(np.gradient(arr, axis=0))

    height = arr.shape[0]
    third = height // 3

    top_activity = np.sum(gradient_y[:third, :])
    middle_activity = np.sum(gradient_y[third:2*third, :])
    bottom_activity = np.sum(gradient_y[2*third:, :])

    activities = {
        'top': top_activity,
        'middle': middle_activity,
        'bottom': bottom_activity
    }

    # Return region with most activity (likely where subject is)
    return max(activities, key=activities.get)


def get_optimal_text_placement(
    image_path: str,
    prefer_dark: bool = True,
    avoid_subject: bool = True,
    return_metadata: bool = False
) -> Tuple[float, float] | dict:
    """
    Determine optimal text placement based on image analysis.

    Args:
        image_path: Path to the image file
        prefer_dark: If True, prefer darker regions for better text contrast
        avoid_subject: If True, avoid placing text over detected subjects
        return_metadata: If True, return dict with placement + subject info

    Returns:
        If return_metadata=False: Tuple of (x_center, y_center)
        If return_metadata=True: Dict with 'placement', 'subject_region', 'split_text'
    """
    brightness = analyze_brightness_regions(image_path)
    subject_region = analyze_subject_location(image_path) if avoid_subject else None

    # Score each region (lower score = better for text)
    scores = {}

    for region in ['top', 'middle', 'bottom']:
        score = 0

        # Brightness scoring (prefer dark regions for white text)
        if prefer_dark:
            # Dark areas (low brightness) get low scores (good)
            # Bright areas (high brightness) get high scores (bad)
            score += brightness[region] / 255.0 * 100

        # Subject avoidance scoring
        if avoid_subject and subject_region == region:
            # Add penalty for regions with detected subjects
            score += 50

        scores[region] = score

    # Choose region with lowest score
    best_region = min(scores, key=scores.get)

    # Convert region to y-coordinate
    region_to_y = {
        'top': 0.20,      # 20% from top - AVOID LOGO in top-left
        'middle': 0.50,   # Center
        'bottom': 0.75    # 75% from top (near bottom)
    }

    y_position = region_to_y[best_region]

    # Horizontal positioning with logo avoidance
    # J.League logo is in top-left corner - avoid left side when placing at top
    if best_region == 'top':
        # Place slightly right of center to avoid logo
        x_position = 0.55
    else:
        # Center for middle/bottom
        x_position = 0.5

    if return_metadata:
        # Determine if text should be split
        # Split when subject is in middle (text from top/bottom might extend into it)
        # OR when text is placed in adjacent region (risk of overlap)
        should_split = False
        if subject_region:
            if subject_region == 'middle':
                # Subject in middle - text from top or bottom might extend into it
                should_split = True
            elif (best_region == 'top' and subject_region == 'middle') or \
                 (best_region == 'bottom' and subject_region == 'middle') or \
                 (best_region == 'top' and subject_region == 'top') or \
                 (best_region == 'bottom' and subject_region == 'bottom'):
                # Text adjacent to or overlapping subject
                should_split = True

        return {
            'placement': (x_position, y_position),
            'subject_region': subject_region,
            'text_region': best_region,
            'split_text': should_split
        }
    else:
        return (x_position, y_position)


def analyze_all_images(image_paths: list[str], return_metadata: bool = False) -> dict:
    """
    Analyze multiple images and return optimal text placements.

    Args:
        image_paths: List of image file paths
        return_metadata: If True, include subject info for text splitting

    Returns:
        Dict mapping image filename to placement info (tuple or dict based on return_metadata)
    """
    placements = {}

    for path in image_paths:
        filename = Path(path).name
        try:
            result = get_optimal_text_placement(path, return_metadata=return_metadata)
            placements[filename] = result

            # Logging
            if return_metadata:
                y_pos = result['placement'][1]
                split_flag = " (split layout)" if result.get('split_text') else ""
            else:
                y_pos = result[1]
                split_flag = ""

            if y_pos < 0.4:
                anchor = "top"
            elif y_pos > 0.6:
                anchor = "bottom"
            else:
                anchor = "mid"

            print(f"✓ {filename}: text at {anchor}{split_flag} (y={y_pos:.2f})")
        except Exception as e:
            print(f"⚠ Failed to analyze {filename}: {e}, using default")
            if return_metadata:
                placements[filename] = {
                    'placement': (0.5, 0.35),
                    'subject_region': None,
                    'text_region': 'middle',
                    'split_text': False
                }
            else:
                placements[filename] = (0.5, 0.35)

    return placements
