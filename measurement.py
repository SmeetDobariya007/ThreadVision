# measurement.py — ThreadVision AI Thread Measurement Module
# =============================================================
# Extracts 5 critical thread dimensions from edge-detected images:
#   1. Major Diameter (mm)
#   2. Pitch (mm)
#   3. Minor Diameter (mm)
#   4. Thread Depth (mm) — derived from 1 & 3
#   5. Flank Angle (degrees)
#
# All linear measurements are converted from pixels using PIXEL_TO_MM.
# All algorithms follow the specifications in the master prompt Part 6.
#
# Usage:
#   from measurement import measure
#   results = measure(edges, gray)
#   # results = {'major_diameter': 8.001, 'minor_diameter': 6.512, ...}

import logging
import math
import numpy as np
import cv2
from scipy.signal import find_peaks

from config import (
    PIXEL_TO_MM,
    PEAK_MIN_HEIGHT_RATIO,
    PEAK_MIN_DISTANCE_PX,
)

logger = logging.getLogger("ThreadVision.measurement")


class MeasurementError(Exception):
    """Raised when a measurement cannot be computed from the image data."""
    pass


def measure(edges, gray):
    """
    Extract all 5 thread dimensions from preprocessed images.

    This is the core measurement engine of ThreadVision. Each dimension
    uses a different algorithm optimized for the backlit silhouette
    imaging setup.

    Args:
        edges (np.ndarray): Binary edge map (uint8, 0/255) from preprocessor.
        gray (np.ndarray): Enhanced grayscale image (uint8) from preprocessor.

    Returns:
        dict: Measurement results with keys:
            - 'major_diameter': float (mm) or None if unmeasurable
            - 'minor_diameter': float (mm) or None if unmeasurable
            - 'pitch':          float (mm) or None if unmeasurable
            - 'thread_depth':   float (mm) or None if unmeasurable
            - 'flank_angle':    float (degrees) or None if unmeasurable
            - 'warnings':       list of str — non-fatal diagnostic messages
            - 'bolt_contour':   np.ndarray — largest contour (for GUI overlay)
            - 'bolt_bbox':      tuple (x, y, w, h) — bounding box in pixels
            - 'peaks':          np.ndarray — peak positions (for GUI overlay)

    Raises:
        MeasurementError: If no bolt is detected or image is unusable.
    """
    if edges is None or gray is None:
        raise MeasurementError("Received None input — preprocessing failed.")

    warnings = []
    mm_per_px = 1.0 / PIXEL_TO_MM  # Conversion factor: multiply px by this

    # ===================================================================
    # STEP 0: Find the bolt contour and bounding box
    # ===================================================================
    contour, bbox = _find_bolt_contour(edges)
    x, y, w, h = bbox
    logger.info(
        f"Bolt detected: bbox=({x},{y},{w},{h}), "
        f"width={w * mm_per_px:.2f}mm, height={h * mm_per_px:.2f}mm"
    )

    # ===================================================================
    # DIMENSION 1: Major Diameter
    # ===================================================================
    major_diam_mm = _measure_major_diameter(contour, bbox, mm_per_px, warnings)

    # ===================================================================
    # DIMENSION 2: Pitch (via peak detection on intensity profile)
    # ===================================================================
    pitch_mm, peaks = _measure_pitch(gray, bbox, mm_per_px, warnings)

    # ===================================================================
    # DIMENSION 3: Minor Diameter
    # ===================================================================
    minor_diam_mm = _measure_minor_diameter(edges, bbox, mm_per_px, warnings)

    # ===================================================================
    # DIMENSION 4: Thread Depth (derived from major and minor)
    # ===================================================================
    thread_depth_mm = None
    if major_diam_mm is not None and minor_diam_mm is not None:
        thread_depth_mm = (major_diam_mm - minor_diam_mm) / 2.0
        if thread_depth_mm < 0:
            warnings.append(
                f"Thread depth is negative ({thread_depth_mm:.3f}mm) — "
                "minor > major. Check contour detection."
            )
            thread_depth_mm = abs(thread_depth_mm)
        logger.info(f"  Thread depth: {thread_depth_mm:.3f} mm (derived)")
    else:
        warnings.append("Thread depth unmeasurable — major or minor diameter missing.")

    # ===================================================================
    # DIMENSION 5: Flank Angle
    # ===================================================================
    flank_angle_deg = _measure_flank_angle(edges, bbox, mm_per_px, warnings)

    # Log warnings
    for w_msg in warnings:
        logger.warning(f"  ⚠ {w_msg}")

    return {
        'major_diameter': major_diam_mm,
        'minor_diameter': minor_diam_mm,
        'pitch':          pitch_mm,
        'thread_depth':   thread_depth_mm,
        'flank_angle':    flank_angle_deg,
        'warnings':       warnings,
        'bolt_contour':   contour,
        'bolt_bbox':      bbox,
        'peaks':          peaks,
    }


# =========================================================================
# Internal measurement functions
# =========================================================================

def _find_bolt_contour(edges):
    """
    Find the largest contour (the bolt silhouette) in the edge map.

    The bolt is expected to be the dominant object in the backlit image.
    We select the contour with the largest area, which corresponds to
    the full bolt outline.

    Args:
        edges: Binary edge map.

    Returns:
        tuple: (contour, (x, y, w, h)) — the bolt contour and its bounding rect.

    Raises:
        MeasurementError: If no contour is found or bolt appears invalid.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise MeasurementError(
            "No contours found in edge map. "
            "Check: Is the bolt in frame? Is the backlight on?"
        )

    # Select largest contour by bounding rect area (not cv2.contourArea,
    # which returns near-zero for thin Canny edge contours)
    def _bbox_area(c):
        _, _, cw, ch = cv2.boundingRect(c)
        return cw * ch

    bolt_contour = max(contours, key=_bbox_area)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(bolt_contour)
    bbox_area = w * h

    # Minimum area sanity check (bolt should occupy significant frame area)
    frame_area = edges.shape[0] * edges.shape[1]
    area_ratio = bbox_area / frame_area
    if area_ratio < 0.005:  # Less than 0.5% of frame
        raise MeasurementError(
            f"Largest contour bbox is too small ({area_ratio*100:.2f}% of frame). "
            "Bolt may not be in frame or image is too noisy."
        )

    # Validate physical size
    mm_per_px = 1.0 / PIXEL_TO_MM
    width_mm = w * mm_per_px
    if width_mm > 20.0:
        raise MeasurementError(
            f"Detected bolt width ({width_mm:.1f}mm) exceeds 20mm. "
            "Possible noise or multiple objects in frame."
        )
    if width_mm < 4.0:
        raise MeasurementError(
            f"Detected bolt width ({width_mm:.1f}mm) is under 4mm. "
            "Bolt may be too far from camera or calibration is wrong."
        )

    return bolt_contour, (x, y, w, h)


def _measure_major_diameter(contour, bbox, mm_per_px, warnings):
    """
    Measure the major (outer) diameter of the thread.

    The major diameter is the widest point across the thread crests.
    We use the bounding rectangle width of the bolt contour as a
    first approximation, then refine by looking at the actual
    horizontal extents of the contour.

    Args:
        contour: The bolt's largest contour.
        bbox: (x, y, w, h) bounding rectangle.
        mm_per_px: Pixel-to-mm conversion factor.
        warnings: List to append diagnostic messages to.

    Returns:
        float: Major diameter in mm, or None if unmeasurable.
    """
    x, y, w, h = bbox

    # Refined measurement: find the actual maximum horizontal extent
    # of the contour points (not just bounding rect, which includes noise)
    contour_pts = contour.reshape(-1, 2)  # (N, 2) array of (x, y)

    # Use the middle 60% of the bolt height to avoid head/tip effects
    y_min = y + int(h * 0.2)
    y_max = y + int(h * 0.8)

    # Filter contour points to the middle region
    mask = (contour_pts[:, 1] >= y_min) & (contour_pts[:, 1] <= y_max)
    mid_pts = contour_pts[mask]

    if len(mid_pts) < 10:
        warnings.append("Too few contour points in middle region for major diameter.")
        # Fall back to bounding rect width
        major_mm = w * mm_per_px
    else:
        # For each row, find the maximum horizontal spread
        # This gives us the crest-to-crest width per row
        unique_rows = np.unique(mid_pts[:, 1])
        max_widths = []

        for row_y in unique_rows:
            row_pts = mid_pts[mid_pts[:, 1] == row_y]
            if len(row_pts) >= 2:
                row_width = row_pts[:, 0].max() - row_pts[:, 0].min()
                max_widths.append(row_width)

        if max_widths:
            # Use 97th percentile of widths = crest-to-crest diameter
            # 97th (not 90th) because ISO M8 tolerance is only ±0.028mm
            # (~3.4px at 120px/mm) — 90th percentile underestimates
            major_px = np.percentile(max_widths, 97)
            major_mm = major_px * mm_per_px
        else:
            major_mm = w * mm_per_px
            warnings.append("Could not compute per-row widths; using bounding rect.")

    logger.info(f"  Major diameter: {major_mm:.3f} mm")
    return major_mm


def _measure_pitch(gray, bbox, mm_per_px, warnings):
    """
    Measure thread pitch using peak detection on intensity profile.

    Extracts a vertical intensity profile along the LEFT EDGE of the
    bolt silhouette (not the center, which is uniformly dark). Thread
    crests and roots create periodic intensity oscillations at the
    silhouette edge where the bolt width varies.

    This is fundamentally a signal processing problem: we're looking
    for periodic structure in a noisy 1D signal.

    Args:
        gray: Enhanced grayscale image.
        bbox: (x, y, w, h) bounding rectangle of the bolt.
        mm_per_px: Pixel-to-mm conversion factor.
        warnings: List to append diagnostic messages to.

    Returns:
        tuple: (pitch_mm, peaks) — pitch in mm and peak positions array.
               pitch_mm is None if fewer than 3 peaks found.
    """
    x, y, w, h = bbox
    y_end = min(y + h, gray.shape[0])

    # Extract an 8-pixel-wide strip along the LEFT edge of the bolt.
    # The center column is inside the dark silhouette and shows minimal
    # intensity variation. The edge is where thread crests (wider) and
    # roots (narrower) modulate the light/dark boundary.
    strip_x_start = max(0, x)
    strip_x_end = min(x + 8, gray.shape[1])
    strip = gray[y:y_end, strip_x_start:strip_x_end].astype(np.float64)

    if strip.size == 0 or strip.shape[0] < 20:
        warnings.append("Intensity profile too short for pitch detection.")
        return None, np.array([])

    # Average across the 8px strip width for noise reduction
    profile = strip.mean(axis=1)

    # Determine inversion by sampling the BACKGROUND region (left of bolt),
    # not the profile mean, which varies with edge position.
    # For backlit: background is bright, bolt edge is dark.
    bg_x_end = max(0, x - 5)
    bg_sample = gray[y + h // 2, 0:bg_x_end] if bg_x_end > 0 else np.array([])
    bg_mean = bg_sample.mean() if len(bg_sample) > 0 else 200.0
    bolt_is_dark_on_light = bg_mean > 128

    if bolt_is_dark_on_light:
        # Invert so peaks correspond to thread features (crests)
        profile = 255.0 - profile

    # Smooth the profile slightly to reduce high-frequency noise
    from scipy.ndimage import uniform_filter1d
    profile_smooth = uniform_filter1d(profile, size=5)

    # Set peak detection thresholds
    p_max = profile_smooth.max()
    p_min = profile_smooth.min()
    height_threshold = p_min + PEAK_MIN_HEIGHT_RATIO * (p_max - p_min)

    # Find peaks
    peaks, properties = find_peaks(
        profile_smooth,
        height=height_threshold,
        distance=PEAK_MIN_DISTANCE_PX,
        prominence=(p_max - p_min) * 0.1,  # Minimum 10% prominence
    )

    logger.info(
        f"  Pitch detection: {len(peaks)} peaks found in "
        f"{len(profile)} px profile"
    )

    if len(peaks) < 3:
        warnings.append(
            f"Only {len(peaks)} peaks found (need ≥3 for pitch). "
            "Thread may be out of focus or bolt is misaligned."
        )
        return None, peaks

    # Compute inter-peak distances
    diffs = np.diff(peaks).astype(np.float64)

    if len(diffs) == 0:
        warnings.append("No inter-peak distances computed.")
        return None, peaks

    # Remove outliers: keep only diffs within 1 std dev of median
    median_diff = np.median(diffs)
    std_diff = np.std(diffs)

    if std_diff > 0:
        mask = np.abs(diffs - median_diff) <= std_diff
        filtered_diffs = diffs[mask]
    else:
        filtered_diffs = diffs

    if len(filtered_diffs) == 0:
        filtered_diffs = diffs  # Fall back to unfiltered

    # Use median of filtered diffs (more robust than mean)
    pitch_px = np.median(filtered_diffs)
    pitch_mm = pitch_px * mm_per_px

    # Check variance — high variance suggests damaged or irregular threads
    if std_diff > 0 and (std_diff / median_diff) > 0.2:
        warnings.append(
            f"High inter-peak variance ({std_diff/median_diff*100:.1f}%). "
            "Possible thread damage or irregular thread form."
        )

    logger.info(
        f"  Pitch: {pitch_mm:.3f} mm "
        f"({pitch_px:.1f} px, {len(filtered_diffs)} valid intervals)"
    )

    return pitch_mm, peaks


def _measure_minor_diameter(edges, bbox, mm_per_px, warnings):
    """
    Measure the minor (root) diameter of the thread.

    Scans each row within the bolt bounding box, finds the horizontal
    extent of edge pixels, and uses the 10th percentile of widths
    as the minor diameter. The 10th percentile (not minimum) is used
    because minimum would be dominated by noise and single-pixel
    artifacts.

    Args:
        edges: Binary edge map.
        bbox: (x, y, w, h) bounding rectangle of the bolt.
        mm_per_px: Pixel-to-mm conversion factor.
        warnings: List to append diagnostic messages to.

    Returns:
        float: Minor diameter in mm, or None if unmeasurable.
    """
    x, y, w, h = bbox

    row_widths = []

    # Use the middle 60% of bolt height (avoid head and tip)
    y_start = y + int(h * 0.2)
    y_end = min(y + int(h * 0.8), edges.shape[0])

    for row in range(y_start, y_end):
        # Extract edge pixels in this row within the bolt region
        x_end = min(x + w, edges.shape[1])
        row_edges = edges[row, x:x_end]

        # Find nonzero (edge) pixel positions
        positions = np.nonzero(row_edges)[0]

        if len(positions) >= 2:
            width_px = positions[-1] - positions[0]
            if width_px > 0:
                row_widths.append(width_px)

    if len(row_widths) < 10:
        warnings.append(
            f"Only {len(row_widths)} valid row widths found for minor diameter. "
            "Edge detection may be insufficient."
        )
        if len(row_widths) == 0:
            return None

    # Sort and remove top/bottom 10% as outliers
    row_widths_arr = np.array(row_widths, dtype=np.float64)
    n = len(row_widths_arr)
    trim = max(1, int(n * 0.1))
    trimmed = np.sort(row_widths_arr)[trim:-trim] if n > 2 * trim else row_widths_arr

    # 10th percentile = thread root width (minimum functional width)
    minor_px = np.percentile(trimmed, 10)
    minor_mm = minor_px * mm_per_px

    logger.info(
        f"  Minor diameter: {minor_mm:.3f} mm "
        f"(10th percentile of {len(trimmed)} rows)"
    )

    return minor_mm


def _measure_flank_angle(edges, bbox, mm_per_px, warnings):
    """
    Measure the thread flank angle using least-squares line fitting.

    The flank angle is the included angle of the thread tooth profile.
    For ISO metric threads (M-series), this should be 60°.

    Uses ALL contour points in the middle 40% of the bolt height
    (no left/right split). Segments points by Y-direction change in X
    into rising and falling flanks, fits lines to each, and computes:
        included_angle = rising_half_angle + falling_half_angle

    This avoids the error of doubling a single side's half-angle.

    Args:
        edges: Binary edge map.
        bbox: (x, y, w, h) bounding rectangle of the bolt.
        mm_per_px: Pixel-to-mm conversion factor.
        warnings: List to append diagnostic messages to.

    Returns:
        float: Flank angle in degrees, or None if unmeasurable.
    """
    x, y, w, h = bbox

    # Find contours in the bolt region only
    roi = edges[y:y+h, x:x+w].copy()
    contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        warnings.append("No contours found in bolt ROI for flank angle.")
        return None

    # Combine all contour points in the middle 40% height band
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    h_start = int(h * 0.3)
    h_end = int(h * 0.7)
    mid_pts = all_pts[
        (all_pts[:, 1] >= h_start) & (all_pts[:, 1] <= h_end)
    ]

    if len(mid_pts) < 20:
        warnings.append(
            f"Too few contour points ({len(mid_pts)}) in middle band for flank angle."
        )
        return None

    # Compute rising and falling half-angles separately
    rising_angles, falling_angles = _fit_flank_half_angles(mid_pts, warnings)

    if not rising_angles and not falling_angles:
        warnings.append("Could not fit lines to any flank edges.")
        return None

    # The included flank angle = rising_half_angle + falling_half_angle
    # For ISO metric: each half is ~30° from vertical → included = 60°
    if rising_angles and falling_angles:
        avg_rising = np.median(rising_angles)
        avg_falling = np.median(falling_angles)
        flank_angle = avg_rising + avg_falling
        logger.debug(
            f"  Flank: rising={avg_rising:.1f}° + falling={avg_falling:.1f}° "
            f"= {flank_angle:.1f}° (from {len(rising_angles)}+{len(falling_angles)} segments)"
        )
    elif rising_angles:
        # Only rising flanks found — estimate included = 2 × rising
        avg_rising = np.median(rising_angles)
        flank_angle = 2.0 * avg_rising
        warnings.append("Only rising flanks measured; falling flanks missing.")
    else:
        # Only falling flanks found — estimate included = 2 × falling
        avg_falling = np.median(falling_angles)
        flank_angle = 2.0 * avg_falling
        warnings.append("Only falling flanks measured; rising flanks missing.")

    # Validate range
    if flank_angle < 45.0 or flank_angle > 75.0:
        warnings.append(
            f"Flank angle {flank_angle:.1f}° is outside expected range "
            f"(45°–75°). SUSPECT — VERIFY measurement."
        )

    logger.info(f"  Flank angle: {flank_angle:.1f}°")
    return flank_angle


def _fit_flank_half_angles(points, warnings):
    """
    Segment contour points into rising and falling flanks, fit lines,
    and return separate lists of half-angles from vertical.

    Each thread tooth has a rising flank (X increasing with Y) and a
    falling flank (X decreasing with Y). We segment the points by
    detecting direction changes in X as we walk along sorted Y, then
    fit a least-squares line (x = m*y + b) to each segment.

    The half-angle from vertical = arctan(|slope|), where slope = dx/dy.

    Args:
        points: (N, 2) array of edge points in the middle height band.
        warnings: List to append diagnostic messages to.

    Returns:
        tuple: (rising_angles, falling_angles) — two lists of float,
               each entry is a half-angle from vertical in degrees.
    """
    if len(points) < 10:
        warnings.append(
            f"Too few flank points ({len(points)}) for line fitting."
        )
        return [], []

    # Sort points by Y coordinate
    sorted_pts = points[points[:, 1].argsort()]
    x_vals = sorted_pts[:, 0].astype(np.float64)
    y_vals = sorted_pts[:, 1].astype(np.float64)

    # Segment into rising (X increasing) and falling (X decreasing) groups
    segments_rising = []
    segments_falling = []

    current_segment = [(x_vals[0], y_vals[0])]
    is_rising = True

    for i in range(1, len(x_vals)):
        if len(current_segment) > 0:
            prev_x = current_segment[-1][0]
            curr_rising = x_vals[i] >= prev_x

            if curr_rising == is_rising or len(current_segment) < 3:
                current_segment.append((x_vals[i], y_vals[i]))
            else:
                if len(current_segment) >= 5:
                    if is_rising:
                        segments_rising.append(np.array(current_segment))
                    else:
                        segments_falling.append(np.array(current_segment))
                current_segment = [(x_vals[i], y_vals[i])]
                is_rising = curr_rising
        else:
            current_segment.append((x_vals[i], y_vals[i]))

    # Don't forget the last segment
    if len(current_segment) >= 5:
        if is_rising:
            segments_rising.append(np.array(current_segment))
        else:
            segments_falling.append(np.array(current_segment))

    # Fit lines and compute half-angle from vertical for each segment
    rising_angles = []
    falling_angles = []

    for segments, angle_list, label in [
        (segments_rising, rising_angles, "rising"),
        (segments_falling, falling_angles, "falling"),
    ]:
        for seg in segments:
            if len(seg) < 5:
                continue

            seg_x = seg[:, 0]
            seg_y = seg[:, 1]

            # Least-squares line fit: x = m*y + b (fit x as function of y
            # because flanks are nearly vertical)
            A = np.vstack([seg_y, np.ones(len(seg_y))]).T
            try:
                result = np.linalg.lstsq(A, seg_x, rcond=None)
                slope = result[0][0]  # dx/dy

                # Half-angle from vertical: arctan(|slope|)
                half_angle = math.degrees(math.atan(abs(slope)))
                angle_list.append(half_angle)
            except np.linalg.LinAlgError:
                continue

    logger.debug(
        f"  Flank fitting: {len(rising_angles)} rising segments, "
        f"{len(falling_angles)} falling segments"
    )

    return rising_angles, falling_angles


# ---------------------------------------------------------------------------
# Self-test: run `python measurement.py` to test with synthetic data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Measurement Module Self-Test")
    print("=" * 60)

    # Generate a synthetic bolt image for testing
    try:
        from capture import CameraManager
        cam = CameraManager()
        frame = cam.capture_frame()
        cam.release()
    except Exception:
        # Create a simple synthetic image
        print("  No camera — using synthetic bolt image...")
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        cx, cy = 320, 240
        bolt_w, bolt_h = 40, 300
        x1, x2 = cx - bolt_w//2, cx + bolt_w//2
        y1, y2 = cy - bolt_h//2, cy + bolt_h//2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        for i in range(15):
            ty = y1 + i * 20
            pts_l = np.array([[x1, ty], [x1-5, ty+10], [x1, ty+20]], np.int32)
            pts_r = np.array([[x2, ty], [x2+5, ty+10], [x2, ty+20]], np.int32)
            cv2.fillPoly(frame, [pts_l], (30, 30, 30))
            cv2.fillPoly(frame, [pts_r], (30, 30, 30))

    # Preprocess
    from preprocessor import preprocess
    edges, gray = preprocess(frame)

    # Measure
    try:
        results = measure(edges, gray)
        print("\n  Measurement Results:")
        print("  " + "-" * 40)
        for key in ['major_diameter', 'minor_diameter', 'pitch',
                     'thread_depth', 'flank_angle']:
            val = results[key]
            unit = "°" if key == "flank_angle" else "mm"
            if val is not None:
                print(f"    {key:20s} = {val:7.3f} {unit}")
            else:
                print(f"    {key:20s} = UNMEASURED")

        if results['warnings']:
            print("\n  Warnings:")
            for w in results['warnings']:
                print(f"    ⚠ {w}")

        print("\n✓ Measurement test passed.")
    except MeasurementError as e:
        print(f"\n✗ Measurement error: {e}", file=sys.stderr)
        sys.exit(1)
