# preprocessor.py — ThreadVision AI Image Preprocessing Pipeline
# ================================================================
# Converts a raw BGR camera frame into edge-detected and enhanced
# grayscale images ready for measurement extraction.
#
# Pipeline: BGR → Grayscale → Gaussian Blur → CLAHE → Canny Edges
#
# Usage:
#   from preprocessor import preprocess
#   edges, gray = preprocess(frame)

import logging
import numpy as np
import cv2

from config import (
    CANNY_LOW, CANNY_HIGH,
    BLUR_KERNEL,
    CLAHE_CLIP, CLAHE_GRID,
)

logger = logging.getLogger("ThreadVision.preprocessor")


class PreprocessingError(Exception):
    """Raised when the preprocessing pipeline encounters invalid input."""
    pass


def preprocess(frame):
    """
    Full image preprocessing pipeline for thread inspection.

    Converts a raw BGR camera frame through a multi-stage pipeline
    optimized for extracting thread geometry from backlit silhouettes:

    1. **Grayscale conversion**: Reduces 3-channel BGR to single channel.
    2. **Gaussian blur**: Reduces high-frequency noise that causes
       false edge detections. Kernel size from config.py.
    3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization):
       Enhances local contrast so thread features at bolt edges
       are as sharp as the centre. Critical for uniform edge detection
       across the full image.
    4. **Canny edge detection**: Produces binary edge map with
       hysteresis thresholding. Low/high thresholds from config.py.

    Args:
        frame (np.ndarray): BGR image from camera (any resolution).
                            Must be a valid 3-channel uint8 image.

    Returns:
        tuple: (edges, gray) where:
            - edges (np.ndarray): Binary edge map (uint8, 0 or 255).
                                  Same size as input frame.
            - gray (np.ndarray): Enhanced grayscale image (uint8).
                                 CLAHE-equalized, suitable for
                                 intensity profiling in measurement.py.

    Raises:
        PreprocessingError: If input frame is None, empty, or invalid.
    """
    # === Input validation ===
    if frame is None:
        raise PreprocessingError("Input frame is None — camera may have failed.")

    if not isinstance(frame, np.ndarray):
        raise PreprocessingError(
            f"Expected np.ndarray, got {type(frame).__name__}."
        )

    if frame.size == 0:
        raise PreprocessingError("Input frame is empty (0 pixels).")

    if frame.ndim not in (2, 3):
        raise PreprocessingError(
            f"Expected 2D or 3D array, got {frame.ndim}D (shape: {frame.shape})."
        )

    logger.info(f"Preprocessing frame: {frame.shape[1]}×{frame.shape[0]}")

    # === Step 1: Convert to grayscale ===
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    elif frame.ndim == 2:
        gray = frame.copy()
    else:
        raise PreprocessingError(
            f"Unsupported channel count: {frame.shape[2]}."
        )
    logger.debug("  Step 1: Grayscale conversion done.")

    # === Step 2: Gaussian blur ===
    # Reduces sensor noise and minor texture that would create false edges.
    # Kernel must be odd — config.py specifies (5, 5) by default.
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, sigmaX=0)
    logger.debug(f"  Step 2: Gaussian blur applied (kernel={BLUR_KERNEL}).")

    # === Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization) ===
    # Standard histogram equalization can blow out contrast globally.
    # CLAHE operates on tiles, so thread edges at the periphery of the
    # image get the same contrast boost as the centre.
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(blurred)
    logger.debug(
        f"  Step 3: CLAHE applied (clip={CLAHE_CLIP}, grid={CLAHE_GRID})."
    )

    # === Step 4: Canny edge detection ===
    # Hysteresis thresholds: edges above CANNY_HIGH are kept. Edges between
    # CANNY_LOW and CANNY_HIGH are kept only if connected to strong edges.
    # This suppresses noise while preserving thread flanks.
    edges = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    logger.debug(
        f"  Step 4: Canny edges detected (low={CANNY_LOW}, high={CANNY_HIGH})."
    )

    # Count edge pixels for diagnostic logging
    edge_count = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_count / total_pixels * 100
    logger.info(
        f"  Result: {edge_count} edge pixels ({edge_ratio:.1f}% of frame)."
    )

    # Sanity check: if too few edges, image may be blank or overexposed
    if edge_ratio < 0.1:
        logger.warning(
            "Very few edges detected (<0.1%). Image may be blank or "
            "severely overexposed. Check backlight and bolt placement."
        )

    # Sanity check: if too many edges, image may be underexposed or noisy
    if edge_ratio > 30.0:
        logger.warning(
            "Too many edges detected (>30%). Image may be underexposed "
            "or noisy. Check backlight intensity."
        )

    return edges, enhanced


def preprocess_for_display(edges, gray, original_frame=None):
    """
    Create annotated images suitable for GUI display.

    Generates colored overlays that help the operator visually verify
    that edges are being detected on the actual thread features.

    Args:
        edges (np.ndarray): Binary edge map from preprocess().
        gray (np.ndarray): Enhanced grayscale from preprocess().
        original_frame (np.ndarray, optional): Original BGR frame for overlay.

    Returns:
        dict: Dictionary with display images:
            - 'edges_colored': Edge map as green-on-black BGR image.
            - 'overlay': Edges overlaid on original frame (if provided).
            - 'enhanced': CLAHE-enhanced grayscale as BGR.
    """
    result = {}

    # Edge map as green lines on black background
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = 0   # Zero blue channel
    edges_colored[:, :, 2] = 0   # Zero red channel
    result['edges_colored'] = edges_colored

    # Enhanced grayscale as BGR (for display in color GUI)
    result['enhanced'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Overlay edges on original frame
    if original_frame is not None:
        overlay = original_frame.copy()
        # Draw edges as bright green on the original image
        overlay[edges > 0] = (0, 255, 0)
        result['overlay'] = overlay

    return result


# ---------------------------------------------------------------------------
# Self-test: run `python preprocessor.py` to test with a synthetic image
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Preprocessor Self-Test")
    print("=" * 60)

    # Generate or load a test image
    try:
        from capture import CameraManager
        cam = CameraManager()
        frame = cam.capture_frame()
        cam.release()
        print(f"  Using camera frame: {frame.shape[1]}×{frame.shape[0]}")
    except Exception:
        # Create a synthetic test image
        print("  No camera — generating synthetic test image...")
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Draw a dark rectangle (bolt simulation)
        cv2.rectangle(frame, (280, 50), (360, 430), (30, 30, 30), -1)
        # Add some thread-like horizontal lines
        for y in range(70, 420, 25):
            cv2.line(frame, (270, y), (370, y), (100, 100, 100), 1)

    # Run preprocessing
    try:
        edges, gray = preprocess(frame)
        print(f"  Edges shape  : {edges.shape}")
        print(f"  Gray shape   : {gray.shape}")
        print(f"  Edge pixels  : {np.count_nonzero(edges)}")

        # Save results
        cv2.imwrite("test_edges.png", edges)
        cv2.imwrite("test_gray_enhanced.png", gray)
        print("  Saved: test_edges.png, test_gray_enhanced.png")

        # Test display helpers
        displays = preprocess_for_display(edges, gray, frame)
        cv2.imwrite("test_overlay.png", displays.get('overlay', edges))
        print("  Saved: test_overlay.png")

        print("\n✓ Preprocessor test passed.")
    except PreprocessingError as e:
        print(f"\n✗ Preprocessing error: {e}", file=sys.stderr)
        sys.exit(1)
