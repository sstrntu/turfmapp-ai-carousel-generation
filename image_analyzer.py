"""
Intelligent image analysis for optimal text placement.

Analyzes images to determine the best position for text overlays based on:
- Background brightness/darkness (prefer dark areas for white text)
- Subject location (avoid covering important subjects)
- Visual complexity (prefer simpler areas)
- Face detection (prioritized over edge detection when faces present)
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

# Try to import OpenCV for face detection (graceful degradation if not available)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


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


def _detect_faces_region(image_path: str) -> Optional[str]:
    """
    Detect faces using OpenCV Haar cascades and return the region containing them.

    Returns 'top', 'middle', or 'bottom' based on where faces are detected.
    Returns None if no faces found or OpenCV not available.
    """
    if not OPENCV_AVAILABLE:
        return None

    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        third = height // 3

        # Load Haar cascade classifiers for frontal and profile faces
        face_cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # Detect frontal faces
        faces_frontal = face_cascade_frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Detect profile faces
        faces_profile = face_cascade_profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Combine all detected faces
        all_faces = list(faces_frontal) + list(faces_profile)

        if len(all_faces) == 0:
            return None

        # Calculate the bounding box containing all faces
        min_y = float('inf')
        max_y = 0

        for (x, y, w, h) in all_faces:
            min_y = min(min_y, y)
            max_y = max(max_y, y + h)

        # Find the center of the face region
        face_center_y = (min_y + max_y) / 2

        # Determine which region the faces are in
        if face_center_y < third:
            return 'top'
        elif face_center_y < 2 * third:
            return 'middle'
        else:
            return 'bottom'

    except Exception as e:
        print(f"  ⚠️ Face detection failed: {e}")
        return None


def _analyze_subject_by_edges(image_path: str) -> str:
    """
    Detect where the main subject is located using edge detection.

    Returns 'top', 'middle', or 'bottom' based on where most visual activity is.
    This works well for crowds, landscapes, and scenes without clear faces.
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


def analyze_subject_location(image_path: str) -> Optional[str]:
    """
    Detect where the main subject is located using face detection with edge fallback.

    Priority:
    1. Try face detection first (if OpenCV available)
    2. If faces found → return region containing faces
    3. If NO faces found → fall back to edge detection

    Edge detection fallback handles:
    - Crowd shots (finds where the crowd is)
    - Landscapes (finds visual mass)
    - Stadium views (identifies main elements)

    Returns 'top', 'middle', or 'bottom' based on where the subject is.
    """
    # Try face detection first
    face_region = _detect_faces_region(image_path)
    if face_region is not None:
        return face_region

    # Fall back to edge detection for crowds, landscapes, etc.
    return _analyze_subject_by_edges(image_path)


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
        If return_metadata=True: Dict with 'placement', 'subject_region', 'text_region'
            Note: split_text decision is now made at render time based on actual text height
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

        # Subject avoidance scoring (HIGH PRIORITY - avoid subject at all costs)
        if avoid_subject and subject_region == region:
            # Add HEAVY penalty for regions with detected subjects
            # This ensures we NEVER place text over subjects regardless of brightness
            score += 500

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
        # Return subject_region for use in render-time split decision
        # The actual split_text decision is now made at render time based on
        # actual text height, not pre-computed here
        return {
            'placement': (x_position, y_position),
            'subject_region': subject_region,
            'text_region': best_region,
        }
    else:
        return (x_position, y_position)


def analyze_all_images(image_paths: list[str], return_metadata: bool = False) -> dict:
    """
    Analyze multiple images and return optimal text placements.

    Args:
        image_paths: List of image file paths
        return_metadata: If True, include subject info for render-time split decision

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
                subject_info = f", subject={result.get('subject_region', 'unknown')}" if result.get('subject_region') else ""
            else:
                y_pos = result[1]
                subject_info = ""

            if y_pos < 0.4:
                anchor = "top"
            elif y_pos > 0.6:
                anchor = "bottom"
            else:
                anchor = "mid"

            print(f"✓ {filename}: text at {anchor} (y={y_pos:.2f}{subject_info})")
        except Exception as e:
            print(f"⚠ Failed to analyze {filename}: {e}, using default")
            if return_metadata:
                placements[filename] = {
                    'placement': (0.5, 0.35),
                    'subject_region': None,
                    'text_region': 'middle',
                }
            else:
                placements[filename] = (0.5, 0.35)

    return placements
