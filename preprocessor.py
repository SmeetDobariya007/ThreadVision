# preprocessor.py — ThreadVision AI Image Preprocessing Pipeline
# ================================================================
# Converts a raw BGR camera frame into edge-detected and enhanced
# grayscale images ready for measurement extraction.
#
# FIXES IN THIS VERSION:
#   - Adaptive Canny thresholds derived from Otsu's method per image
#     (old version used fixed CANNY_LOW=50 / HIGH=150 from config,
#     which over/under-segments depending on image contrast)
#   - Noise assessment via Laplacian variance → CLEAN / MODERATE / NOISY
#   - NOISY path uses FastNlMeansDenoising before blurring
#   - MODERATE/NOISY paths derive thresholds from Otsu, not config constants
#   - Fixed CLAHE constants from config still used for CLEAN images
#
# Usage:
#   from preprocessor import preprocess
#   edges, gray = preprocess(frame)
#   edges, gray = preprocess(frame, force_mode="NOISY")

import logging
import numpy as np
import cv2

from config import (
    BLUR_KERNEL,
    CLAHE_CLIP, CLAHE_GRID,
    CANNY_LOW, CANNY_HIGH,   # used as fallback for CLEAN images
)

logger = logging.getLogger("ThreadVision.preprocessor")


class PreprocessingError(Exception):
    """Raised when the preprocessing pipeline encounters invalid input."""
    pass


# =========================================================================
# Public API
# =========================================================================

def preprocess(frame, force_mode=None):
    """
    Full adaptive preprocessing pipeline for thread inspection.

    Noise is assessed via the Laplacian variance of the grayscale image:
      CLEAN    (var > 500): GaussianBlur + fixed Canny
      MODERATE (var > 100): GaussianBlur + Otsu-derived Canny
      NOISY    (var ≤ 100): FastNlMeansDenoising + GaussianBlur +
                            Otsu-derived Canny + morphological close

    Args:
        frame (np.ndarray): BGR image from camera (uint8, 3-channel).
        force_mode (str):   Optional override: 'CLEAN', 'MODERATE', or 'NOISY'.
                            When None (default), mode is auto-detected.

    Returns:
        tuple: (edges, gray) where
            edges (np.ndarray): Binary edge map, same size as input (uint8).
            gray  (np.ndarray): CLAHE-enhanced grayscale, same size (uint8).

    Raises:
        PreprocessingError: If input is None, empty, or wrong type.
    """
    gray = _validate_and_to_gray(frame)

    noise_var, mode = assess_noise(gray)
    if force_mode is not None:
        mode = force_mode.upper()
        logger.info(f"Preprocessing mode forced to: {mode}")
    else:
        logger.info(
            f"Preprocessing mode: {mode}  (Laplacian var={noise_var:.1f})"
        )

    if mode == "CLEAN":
        edges, enhanced = _pipeline_clean(gray)
    elif mode == "MODERATE":
        edges, enhanced = _pipeline_moderate(gray)
    else:                              # NOISY
        edges, enhanced = _pipeline_noisy(gray)

    # Diagnostic
    edge_count = int(np.count_nonzero(edges))
    total_px   = edges.shape[0] * edges.shape[1]
    edge_pct   = edge_count / total_px * 100
    logger.info(f"  Edges: {edge_count} px ({edge_pct:.2f}% of frame)")

    if edge_pct < 0.1:
        logger.warning(
            "Very few edges detected (<0.1%). "
            "Image may be blank or overexposed."
        )
    if edge_pct > 30.0:
        logger.warning(
            "Too many edges (>30%). "
            "Image may be underexposed or noisy."
        )

    return edges, enhanced


def assess_noise(gray_image):
    """
    Estimate image noise level via Laplacian variance.

    High variance ↔ sharp edges, low noise (CLEAN).
    Low variance  ↔ blurry or high noise (NOISY).

    Args:
        gray_image (np.ndarray): Single-channel uint8 image.

    Returns:
        tuple: (laplacian_variance: float, mode: str)
               mode is one of 'CLEAN', 'MODERATE', 'NOISY'.
    """
    lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if lap_var > 500:
        return lap_var, "CLEAN"
    elif lap_var > 100:
        return lap_var, "MODERATE"
    else:
        return lap_var, "NOISY"


# =========================================================================
# Private pipeline implementations
# =========================================================================

def _pipeline_clean(gray):
    """
    Fast pipeline for sharp, well-lit images.

    Uses fixed Canny thresholds from config — they work reliably when
    contrast is already high.
    """
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, sigmaX=1.0)
    enhanced = _apply_clahe(blurred)
    edges = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    logger.debug(f"  CLEAN: Canny({CANNY_LOW}, {CANNY_HIGH})")
    return edges, enhanced


def _pipeline_moderate(gray):
    """
    Standard pipeline for average-contrast images.

    Derives Canny thresholds from Otsu's method so the detector
    self-calibrates to each image's actual contrast range.
    Otsu threshold → high = otsu, low = 0.5 × otsu.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.5)
    enhanced = _apply_clahe(blurred)

    otsu_val, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low  = max(1.0, otsu_val * 0.5)
    high = otsu_val
    edges = cv2.Canny(enhanced, low, high)
    logger.debug(f"  MODERATE: Otsu={otsu_val:.0f}, Canny({low:.0f}, {high:.0f})")
    return edges, enhanced


def _pipeline_noisy(gray):
    """
    Aggressive pipeline for grainy / low-contrast images.

    1. FastNlMeansDenoising removes sensor noise before blurring.
    2. Larger Gaussian kernel further smooths residual noise.
    3. Lower Otsu multipliers keep more of the faint thread edges.
    4. Morphological close fills small gaps in thread-edge segments.
    """
    # Step 1: Non-local means denoising (slow but thorough)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=15, templateWindowSize=7, searchWindowSize=21
    )

    # Step 2: Gaussian blur on the denoised image
    blurred = cv2.GaussianBlur(denoised, (9, 9), sigmaX=2.0)
    enhanced = _apply_clahe(blurred)

    # Step 3: Otsu-derived Canny with lower thresholds to retain faint edges
    otsu_val, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low  = max(1.0, otsu_val * 0.3)
    high = otsu_val * 0.8
    edges = cv2.Canny(enhanced, low, high)

    # Step 4: Morphological close — fills small gaps in thread-edge lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    logger.debug(f"  NOISY: Otsu={otsu_val:.0f}, Canny({low:.0f}, {high:.0f}), morph-close")
    return edges, enhanced


def _apply_clahe(gray):
    """Apply CLAHE using constants from config."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)


def _validate_and_to_gray(frame):
    """Validate input and convert to grayscale."""
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
            f"Expected 2D or 3D array, got {frame.ndim}D."
        )
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    if frame.ndim == 2:
        return frame.copy()
    raise PreprocessingError(f"Unsupported channel count: {frame.shape[2]}")


# =========================================================================
# Display helpers (unchanged)
# =========================================================================

def preprocess_for_display(edges, gray, original_frame=None):
    """
    Create annotated images suitable for GUI display.

    Returns:
        dict: 'edges_colored', 'enhanced', and optionally 'overlay'.
    """
    result = {}

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[:, :, 0] = 0
    edges_colored[:, :, 2] = 0
    result['edges_colored'] = edges_colored
    result['enhanced'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if original_frame is not None:
        overlay = original_frame.copy()
        overlay[edges > 0] = (0, 255, 0)
        result['overlay'] = overlay

    return result


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Preprocessor Self-Test")
    print("=" * 60)

    # ── Test 1: CLEAN image ──────────────────────────────────────────
    print("\n--- Test 1: CLEAN synthetic image ---")
    clean = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(clean, (280, 50), (360, 430), (20, 20, 20), -1)
    edges, gray = preprocess(clean)
    assert edges.shape == (480, 640), "Shape mismatch"
    print(f"  ✓ edges shape: {edges.shape}, edge px: {np.count_nonzero(edges)}")

    # ── Test 2: NOISY image ──────────────────────────────────────────
    print("\n--- Test 2: NOISY synthetic image ---")
    noisy = clean.copy()
    noise = np.random.normal(0, 40, noisy.shape).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    lap_var, mode = assess_noise(gray_noisy)
    print(f"  Laplacian var={lap_var:.1f}, mode={mode}")
    edges_n, gray_n = preprocess(noisy)
    print(f"  ✓ NOISY edges: {np.count_nonzero(edges_n)} px")

    # ── Test 3: force_mode override ──────────────────────────────────
    print("\n--- Test 3: force_mode='NOISY' on clean image ---")
    edges_f, _ = preprocess(clean, force_mode="NOISY")
    print(f"  ✓ forced NOISY mode ran without error, "
          f"edges={np.count_nonzero(edges_f)} px")

    # ── Test 4: None input raises ─────────────────────────────────────
    print("\n--- Test 4: None input ---")
    try:
        preprocess(None)
        print("  ✗ Should have raised PreprocessingError")
        sys.exit(1)
    except PreprocessingError:
        print("  ✓ PreprocessingError raised correctly")

    print("\n✓ All preprocessor self-tests passed.")
