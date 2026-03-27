# ── TOP-LEVEL CONSTANTS (user-configurable) ───────────────────────
ARUCO_MARKER_SIZE_MM  = 21.0     # Physical side length of printed ArUco marker
FALLBACK_PIXEL_TO_MM  = 120.0    # Used if ArUco not detected (calibrate on real hardware)

# ── PHYSICAL BOUNDS CHECKS ────────────────────────────────────────
BOLT_WIDTH_MIN_MM     = 4.0      # Smallest supported bolt (M4)
BOLT_WIDTH_MAX_MM     = 50.0     # Largest supported bolt (M48)
FLANK_ANGLE_MIN       = 45.0     # Below this → measurement suspect
FLANK_ANGLE_MAX       = 75.0     # Above this → measurement suspect

# ── IMPORTS REQUIRED ─────────────────────────────────────────────
import cv2
import numpy as np
import scipy.signal
import scipy.ndimage
from cv2 import aruco
import argparse
import csv
import re
import os
import sys
from datetime import datetime
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# ── Optional MobileSAM import ─────────────────────────────────────
SAM_SEGMENTOR = None
try:
    from sam_segmentor import SAMSegmentor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

ARUCO_DICT = aruco.DICT_4X4_50
DEFAULT_STANDARD = "AUTO"

# ── CUSTOM EXCEPTIONS ─────────────────────────────────────────────
class CalibrationError(Exception): pass
class BoltNotFoundError(Exception): pass
class MeasurementError(Exception): pass

# ── ISO TOLERANCE TABLE ───────────────────────────────────────────
ISO_STANDARDS = {
    "M8x1.25":  {"major": (7.972, 8.000), "minor": (6.466, 6.647), "pitch": (1.225, 1.275), "depth": (0.613, 0.677), "flank": (59.0, 61.0)},
    "M8x1.0":   {"major": (7.972, 8.000), "minor": (6.917, 7.153), "pitch": (0.975, 1.025), "depth": (0.491, 0.542), "flank": (59.0, 61.0)},
    "M10x1.5":  {"major": (9.968, 10.000),"minor": (8.160, 8.376), "pitch": (1.470, 1.530), "depth": (0.920, 0.920), "flank": (59.0, 61.0)},
    "M12x1.75": {"major": (11.966,12.000),"minor": (9.853,10.106), "pitch": (1.715, 1.785), "depth": (0.920, 0.920), "flank": (59.0, 61.0)},
    "M16x2.0":  {"major": (15.962,16.000),"minor": (13.546,13.835),"pitch": (1.960, 2.040), "depth": (1.227, 1.227), "flank": (59.0, 61.0)},
    "M20x2.5":  {"major": (19.958,20.000),"minor": (16.933,17.294),"pitch": (2.450, 2.550), "depth": (1.534, 1.534), "flank": (59.0, 61.0)},
    "M24x3.0":  {"major": (23.952,24.000),"minor": (20.319,20.752),"pitch": (2.940, 3.060), "depth": (1.840, 1.840), "flank": (59.0, 61.0)},
    "M34x1.5":  {"major": (33.962,34.000),"minor": (32.376,32.752),"pitch": (1.470, 1.530), "depth": (0.920, 0.920), "flank": (59.0, 61.0)},
    "AUTO":     None,  # script auto-detects standard from measured major diameter
}

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return image

def normalize_orientation(image):
    H, W = image.shape[:2]

    # Portrait images (H > W): bolt is already vertical — never rotate.
    # The measurement pipeline expects a vertical bolt, so portrait = correct.
    if H >= W:
        return image, False

    # Landscape image: find the most bolt-like (elongated) contour to decide
    # whether to rotate, ignoring square blobs (ArUco marker).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False

    # Pick the most elongated contour (ignoring square/ArUco-like blobs)
    best = None
    best_aspect = 1.0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        squareness = min(cw, ch) / max(cw, ch)
        if squareness > 0.7:
            continue  # Skip square blobs (likely ArUco marker)
        aspect = max(cw, ch) / min(cw, ch)
        if aspect > best_aspect:
            best_aspect = aspect
            best = c

    if best is None:
        # Fallback: use the largest contour (original behaviour)
        best = max(contours, key=cv2.contourArea)

    _, _, w, h = cv2.boundingRect(best)
    if w > h:  # bolt is lying horizontally → rotate to make it vertical
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, True
    return image, False


def unrotate_points(pts, original_shape, was_rotated):
    if not was_rotated:
        return pts
    H, W = original_shape[:2]
    if len(pts) == 0: return pts
    return np.array([[p[1], W - p[0]] for p in pts], dtype=np.int32)

def detect_aruco_scale(image, marker_mm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    attempt = 1
    
    if not corners:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_gray = clahe.apply(gray)
        corners, ids, rejected = detector.detectMarkers(cl_gray)
        attempt = 2
        
    if not corners:
        ad_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        corners, ids, rejected = detector.detectMarkers(ad_gray)
        attempt = 3
        
    if not corners:
        raise CalibrationError("Marker not detected")
        
    corner = corners[0][0]
    side_0 = np.linalg.norm(corner[0] - corner[1])
    side_1 = np.linalg.norm(corner[1] - corner[2])
    side_2 = np.linalg.norm(corner[2] - corner[3])
    side_3 = np.linalg.norm(corner[3] - corner[0])
    
    sides = [side_0, side_1, side_2, side_3]
    marker_size_px = np.mean(sides)
    
    variance_ratio = float(np.std(sides) / marker_size_px)
    if variance_ratio > 0.05:
        confidence = "LOW — marker tilted"
    elif variance_ratio > 0.02:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"
        
    mm_per_pixel = marker_mm / marker_size_px
    conf_signals = {'variance_ratio': variance_ratio, 'attempt': attempt}
    return mm_per_pixel, confidence, corner, conf_signals

def assess_noise(gray_image):
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    if laplacian_var > 500:
        return laplacian_var, "CLEAN"
    elif laplacian_var > 100:
        return laplacian_var, "MODERATE"
    else:
        return laplacian_var, "NOISY"

def preprocess(image, mode):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "CLEAN":
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 50, 150)
    elif mode == "MODERATE":
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, otsu_thresh * 0.5, otsu_thresh)
    elif mode == "NOISY":
        denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        blurred = cv2.GaussianBlur(denoised, (9, 9), 2.0)
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, otsu_thresh * 0.3, otsu_thresh * 0.8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, gray

def _find_bolt_contour(edges, image_shape, PIXEL_TO_MM):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = image_shape[0], image_shape[1]
    frame_area = H * W

    # Estimate ArUco marker pixel size so we can exclude it
    # (ArUco is typically a square; we reject contours that look like it)
    aruco_approx_px = None
    if PIXEL_TO_MM and PIXEL_TO_MM > 0:
        # PIXEL_TO_MM is px/mm; a 21mm marker would be ~21*PIXEL_TO_MM wide
        aruco_approx_px = 21.0 * PIXEL_TO_MM

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # --- Size filter: accept anything larger than 0.5% of frame ---
        # (lowered from 5% so small/distant bolts are not discarded)
        if area < frame_area * 0.005:
            continue
        if w < W * 0.01 or h < H * 0.01:
            continue

        # --- Exclude ArUco-like squares ---
        squareness = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aruco_approx_px is not None:
            aruco_tol = aruco_approx_px * 0.4
            if squareness > 0.75 and abs(w - aruco_approx_px) < aruco_tol:
                continue  # looks like the ArUco marker — skip

        # --- Score by bolt-likeness: elongated is better ---
        aspect = max(w, h) / max(min(w, h), 1)  # > 1 means elongated
        bolt_score = area * aspect  # bigger + more elongated = higher score

        candidates.append((bolt_score, c, x, y, w, h))

    if not candidates:
        raise BoltNotFoundError("No valid bolt contour found in image.")

    # Pick the highest-scoring single contour — do NOT merge via convexHull
    # (merging caused the bbox to span background noise across the whole image)
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_c, x, y, w, h = candidates[0]
    area = w * h

    aspect_ratio = float(max(w, h) / max(min(w, h), 1))
    bbox_fill = float(cv2.contourArea(best_c)) / area if area > 0 else 0.0
    conf_signals = {'aspect_ratio': aspect_ratio, 'bbox_fill': bbox_fill}

    return (best_c, x, y, w, h), conf_signals

def _measure_major_diameter(gray, bbox, PIXEL_TO_MM):
    c, x, y, w, h = bbox
    y_start = int(y + 0.2 * h)
    y_end = int(y + 0.8 * h)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    row_widths = []
    for row_y in range(y_start, y_end):
        row_pixels = binary[row_y, x:x+w]
        indices = np.where(row_pixels > 0)[0]
        if len(indices) > 0:
            row_widths.append(indices[-1] - indices[0])
            
    row_count = len(row_widths)
    if not row_widths: 
        return None, None, {'row_count': 0, 'row_width_std': 0.0, 'percentile_gap': 0.0}
        
    p97 = np.percentile(row_widths, 97)
    p90 = np.percentile(row_widths, 90)
    
    row_width_std = float(np.std(row_widths))
    percentile_gap = float(abs(p97 - p90) / p97) if p97 > 0 else 0.0
    
    major_px = float(p97)
    major_mm = major_px / PIXEL_TO_MM
    conf_signals = {'row_count': row_count, 'row_width_std': row_width_std, 'percentile_gap': percentile_gap}
    return major_mm, major_px, conf_signals

def _measure_minor_diameter(gray, bbox, PIXEL_TO_MM):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c, x, y, w, h = bbox
    y_start = int(y + 0.2 * h)
    y_end = int(y + 0.8 * h)
    
    row_widths = []
    for row_y in range(y_start, y_end):
        row_pixels = binary[row_y, x:x+w]
        indices = np.where(row_pixels > 0)[0]
        if len(indices) > 0:
            row_widths.append(indices[-1] - indices[0])
            
    row_count = len(row_widths)
    if not row_widths: 
        return None, None, {'row_count': 0, 'row_width_std': 0.0}
        
    row_width_std = float(np.std(row_widths))
    minor_px = float(np.percentile(row_widths, 10))
    minor_mm = minor_px / PIXEL_TO_MM
    conf_signals = {'row_count': row_count, 'row_width_std': row_width_std}
    return minor_mm, minor_px, conf_signals

def _measure_pitch(gray, bbox, PIXEL_TO_MM):
    c, x, y, w, h = bbox
    strip = gray[y:y+h, max(0, x):x+8]
    profile = strip.mean(axis=1).astype(np.float64)
    
    bg_sample = gray[y + h//2, max(0, x - 20):max(0, x - 2)]
    bg_mean = bg_sample.mean() if len(bg_sample) > 0 else 200.0
    if bg_mean > 128:
        profile = 255.0 - profile
        
    profile = scipy.ndimage.gaussian_filter1d(profile, sigma=2)
    # WEAKNESS PATCHED: Expected pitch max filter previously assumed M8x1.25.
    # If a fine-threaded M8x1.0 is scanned, it would drop legitimate peaks. 
    # Lowered the expected baseline to 0.8 to heavily strengthen fine-pitch identification.
    expected_pitch_px = 0.8 * PIXEL_TO_MM * 0.7
    min_distance = max(int(expected_pitch_px), 15)
    
    peaks, _ = scipy.signal.find_peaks(profile, height=np.percentile(profile, 60), distance=min_distance, prominence=10)
    
    profile_range = float(np.ptp(profile))
    
    if len(peaks) < 3:
        return None, None, None, {'peak_count': len(peaks), 'variance_ratio': 0.0, 'profile_range': profile_range}
        
    diffs = np.diff(peaks)
    median_diff = np.median(diffs)
    variance_ratio = float(np.std(diffs) / median_diff) if median_diff > 0 else 0.0
    
    filtered = diffs[np.abs(diffs - median_diff) < np.std(diffs)]
    
    conf_signals = {'peak_count': len(peaks), 'variance_ratio': variance_ratio, 'profile_range': profile_range}
    
    if len(filtered) == 0:
        return None, None, None, conf_signals
        
    pitch_px = float(np.median(filtered))
    pitch_mm = pitch_px / PIXEL_TO_MM
    return pitch_mm, pitch_px, (y + peaks[len(peaks)//2]), conf_signals

def _compute_thread_depth(major_mm, minor_mm):
    if major_mm is None or minor_mm is None: return None
    return (major_mm - minor_mm) / 2.0

def _measure_flank_angle(edges, bbox):
    c, x, y, w, h = bbox
    mid_x = x + w / 2.0
    
    c_pts = c[:, 0, :]
    y_min, y_max = y + 0.25 * h, y + 0.75 * h
    
    pts_left = c_pts[(c_pts[:, 0] < mid_x) & (c_pts[:, 1] >= y_min) & (c_pts[:, 1] <= y_max)]
    pts_right = c_pts[(c_pts[:, 0] >= mid_x) & (c_pts[:, 1] >= y_min) & (c_pts[:, 1] <= y_max)]
    
    def fit_half_angle(pts):
        if len(pts) < 10: return 0.0, 0, []
        pts = pts[np.argsort(pts[:, 1])]
        angles = []
        i = 0
        while i < len(pts) - 5:
            segment = pts[i:i+5]
            dy = segment[-1, 1] - segment[0, 1]
            dx = segment[-1, 0] - segment[0, 0]
            if dy != 0:
                ang = np.degrees(np.abs(np.arctan(dx/dy)))
                if 10 < ang < 45: 
                    angles.append(ang)
            i += 3
        if not angles: return 0.0, 0, []
        return float(np.mean(angles)), len(angles), angles
        
    left_half, left_n, left_angles = fit_half_angle(pts_left)
    right_half, right_n, right_angles = fit_half_angle(pts_right)
    
    mirrored = False
    if left_n == 0 and right_n > 0:
        left_half = right_half
        mirrored = True
    elif right_n == 0 and left_n > 0:
        right_half = left_half
        mirrored = True
    elif left_n == 0 and right_n == 0:
        return None, {'segments_used': 0, 'angle_spread': 0.0, 'mirrored': True}
        
    included_angle = left_half + right_half
    all_angles = left_angles + right_angles
    angle_spread = float(np.std(all_angles)) if all_angles else 0.0
    segments_used = left_n + right_n
    
    conf_signals = {'segments_used': segments_used, 'angle_spread': angle_spread, 'mirrored': mirrored}
    return included_angle, conf_signals

def compute_confidence(aruco_conf_signals, contour_conf_signals, major_conf_signals, minor_conf_signals, pitch_conf_signals, flank_conf_signals, vals, noise_mode):
    # Calibration
    cal_base = 100
    if aruco_conf_signals is None:
        cal_base = 25
    else:
        var_ratio = aruco_conf_signals.get('variance_ratio', 0)
        attempt = aruco_conf_signals.get('attempt', 1)
        if var_ratio > 0.05: cal_base -= 40
        elif var_ratio > 0.02: cal_base -= 15
        if attempt == 2: cal_base -= 10
        if attempt == 3: cal_base -= 20
    cal_conf = max(0, min(100, int(cal_base)))

    # Major
    if vals.get('major') is None:
        maj_conf = 0
    else:
        maj_base = 100
        rc = major_conf_signals.get('row_count', 0)
        rws = major_conf_signals.get('row_width_std', 0)
        pg = major_conf_signals.get('percentile_gap', 0)
        if rc < 20: maj_base -= 30
        elif rc < 50: maj_base -= 10
        if rws > 15: maj_base -= 20
        elif rws > 8: maj_base -= 10
        if pg > 0.15: maj_base -= 15
        if noise_mode == 'NOISY': maj_base -= 10
        elif noise_mode == 'MODERATE': maj_base -= 5
        maj_conf = max(0, min(100, int(maj_base)))

    # Minor
    if vals.get('minor') is None:
        min_conf = 0
    else:
        min_base = 100
        rc = minor_conf_signals.get('row_count', 0)
        rws = minor_conf_signals.get('row_width_std', 0)
        if rc < 20: min_base -= 30
        elif rc < 50: min_base -= 10
        if rws > 15: min_base -= 20
        elif rws > 8: min_base -= 10
        if noise_mode == 'NOISY': min_base -= 10
        min_conf = max(0, min(100, int(min_base)))

    # Pitch
    if vals.get('pitch') is None: pit_conf = 0
    else:
        pc = pitch_conf_signals.get('peak_count', 0)
        vr = pitch_conf_signals.get('variance_ratio', 0)
        pr = pitch_conf_signals.get('profile_range', 0)
        if pc < 3: pit_base = 0
        elif pc <= 4: pit_base = 40
        elif pc <= 7: pit_base = 65
        elif pc <= 12: pit_base = 85
        else: pit_base = 100
        
        if vr > 0.20: pit_base -= 25
        elif vr > 0.10: pit_base -= 10
        if pr < 20: pit_base -= 20
        if noise_mode == 'NOISY': pit_base -= 10
        pit_conf = max(0, min(100, int(pit_base)))

    # Depth
    dep_conf = min(maj_conf, min_conf)

    # Flank
    if vals.get('flank') is None: fla_conf = 0
    else:
        fla_base = 100
        su = flank_conf_signals.get('segments_used', 0)
        asp = flank_conf_signals.get('angle_spread', 0)
        mir = flank_conf_signals.get('mirrored', False)
        if su < 3: fla_base = 35
        elif su < 6: fla_base = 65
        elif su < 10: fla_base = 85
        else: fla_base = 100
        
        if asp > 10: fla_base -= 25
        elif asp > 5: fla_base -= 12
        if mir: fla_base -= 20
        if noise_mode == 'NOISY': fla_base -= 10
        fla_conf = max(0, min(100, int(fla_base)))

    overall = min(cal_conf, maj_conf, min_conf, pit_conf, dep_conf, fla_conf)
    if overall >= 90: grade = 'A'
    elif overall >= 75: grade = 'B'
    elif overall >= 55: grade = 'C'
    else: grade = 'D'

    return {
        'calibration': cal_conf, 'major': maj_conf, 'minor': min_conf,
        'pitch': pit_conf, 'depth': dep_conf, 'flank': fla_conf,
        'overall': overall, 'grade': grade
    }

def select_standard(major_mm, pitch_mm, std_arg):
    if std_arg != "AUTO" and std_arg in ISO_STANDARDS:
        return std_arg, ISO_STANDARDS[std_arg]
    
    if major_mm is None:
        return "UNKNOWN STANDARD", None
        
    best_match = None
    min_err = float('inf')
    for std_name, tol in ISO_STANDARDS.items():
        if tol is None: continue
        nom_major = tol["major"][1]
        err = abs(major_mm - nom_major)
        if err < min_err and err < 0.5:
            min_err = err
            best_match = std_name
            
    if best_match is None:
        return "UNKNOWN STANDARD", None
    return best_match, ISO_STANDARDS[best_match]

def check_tolerances(measurements, tolerances):
    res = {}
    for dim, val in measurements.items():
        if val is None:
            res[dim] = "UNMEASURED"
            continue
        if tolerances and dim in tolerances:
            lo, hi = tolerances[dim]
            if lo <= val <= hi:
                res[dim] = "PASS"
            else:
                res[dim] = "FAIL"
        else:
            res[dim] = "UNKNOWN"
    return res

def build_annotations(image, bbox, PIXEL_TO_MM, confidence_str, corner_pts, results, vals, tols, p_y, conf, sam_mask=None, seg_method="Canny"):
    c, x, y, w, h = bbox
    out = image.copy()
    
    if sam_mask is not None:
        overlay = out.copy()
        colored_mask = np.zeros_like(out)
        colored_mask[sam_mask > 0] = (180, 0, 180)
        cv2.addWeighted(colored_mask, 0.25, overlay, 0.75, 0, out)
        
    # ArUco marker
    if len(corner_pts) == 4:
        cv2.polylines(out, [corner_pts.astype(np.int32)], True, (0, 200, 0), 2)
        for pt in corner_pts:
            cv2.circle(out, tuple(pt.astype(np.int32)), 6, (0, 0, 255), -1)
        lx, ly = int(corner_pts[0][0]), int(corner_pts[0][1] - 10)
        cv2.putText(out, f"Cal: {PIXEL_TO_MM:.1f} px/mm [{confidence_str}]", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        
    # Scale bar
    H, W = out.shape[:2]
    bar_len = int(10 * PIXEL_TO_MM)
    cv2.line(out, (20, H - 30), (20 + bar_len, H - 30), (0,0,0), 6)
    cv2.line(out, (20, H - 30), (20 + bar_len, H - 30), (255,255,255), 2)
    cv2.putText(out, "10mm", (20, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Bolt contour
    cv2.drawContours(out, [c], -1, (255, 255, 0), 1)
    
    # Major/Minor
    my = y + int(0.3 * h)
    cv2.line(out, (x, my), (x+w, my), (0, 255, 255), 2)
    if vals.get('major'):
        cv2.putText(out, f"Major: {vals['major']:.2f}mm {results['major']}", (x+w+10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.line(out, (x+4, my+30), (x+w-4, my+30), (0, 165, 255), 2)
    if vals.get('minor'):
        cv2.putText(out, f"Minor: {vals['minor']:.2f}mm {results['minor']}", (x+w+10, my+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    
    # Pitch
    if vals.get('pitch'):
        py = p_y if p_y else y + int(0.5 * h)
        pitch_px = int(vals['pitch'] * PIXEL_TO_MM)
        cv2.line(out, (x-15, py), (x-15, py+pitch_px), (255, 0, 255), 2)
        cv2.line(out, (x-15, py), (x-5, py), (255, 0, 255), 2)
        cv2.line(out, (x-15, py+pitch_px), (x-5, py+pitch_px), (255, 0, 255), 2)
        cv2.putText(out, f"P:{vals['pitch']:.2f}mm {results['pitch']}", (max(0, x-150), py+pitch_px//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        
    # Depth
    if vals.get('depth'):
        dy = y + int(0.7 * h)
        cv2.line(out, (x+w, dy), (x+w+15, dy), (255, 255, 255), 2)
        cv2.putText(out, f"h:{vals['depth']:.2f}mm {results['depth']}", (x+w+20, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
    # Flank angle
    if vals.get('flank'):
        cv2.putText(out, f"{vals['flank']:.1f} deg {results['flank']}", (x+w//2, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # PASS/FAIL Banner
    status_vals = list(results.values())
    if "FAIL" in status_vals:
        color = (0, 0, 255)
        text = "FAIL"
    elif "UNMEASURED" in status_vals:
        color = (0, 165, 255)
        text = "PARTIAL - UNMEASURED DIMS"
    else:
        color = (0, 255, 0)
        text = "PASS"
        
    cv2.rectangle(out, (0, 0), (W, 40), color, -1)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
    cv2.putText(out, f"Seg: {seg_method}", (W - 200, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # CONFIDENCE PANEL (Bottom Right)
    panel_x = W - 260
    panel_y = H - 80
    overlay_rect = out.copy()
    cv2.rectangle(overlay_rect, (panel_x-8, panel_y-18), (W-10, H-10), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay_rect, 0.7, out, 0.3, 0)
    
    cv2.putText(out, f"Confidence: {conf['overall']}% [{conf['grade']}]", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(out, f"Cal:{conf['calibration']}% Maj:{conf['major']}% Min:{conf['minor']}%", (panel_x, panel_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(out, f"Pitch:{conf['pitch']}% Flank:{conf['flank']}%", (panel_x, panel_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return out

def print_report(filepath, w, h, orient, was_rot, PIXEL_TO_MM, calibration_str, std_str, noise_lvl, noise_mode, vals, tols, results, conf, major_cs, minor_cs, pitch_cs, flank_cs, aruco_cs, seg_method):
    um_per_px = 1000.0 / PIXEL_TO_MM
    print("==========================================================================")
    print("  ThreadVision AI — Thread Measurement Report v4.2")
    print(f"  Image:       {filepath} ({w} × {h} px)")
    rot_str = ", rotated 90°" if was_rot else ""
    print(f"  Orientation: {orient} (auto-detected{rot_str})")
    print(f"  Calibration: {PIXEL_TO_MM:.2f} px/mm ({um_per_px:.2f} µm/px) — {calibration_str}")
    print(f"  Standard:    {std_str} (ISO 262)")
    print(f"  Segmentation:{seg_method}")
    print(f"  Noise:       {noise_lvl:.1f} ({noise_mode})")
    print("==========================================================================")
    print("")
    print("  Dimension         Measured    Tolerance              Status    Conf")
    print("  ──────────────────────────────────────────────────────────────────────")
    
    def print_row(name, key, cs=None):
        val = vals.get(key)
        res = results.get(key, "UNKNOWN")
        res_disp = f"✓ {res}" if res == "PASS" else f"✗ {res}" if res == "FAIL" else res
        
        c_val = conf[key]
        reason = ""
        if c_val < 55:
            if key == 'pitch' and cs is not None:
                reason = f"  ← {cs.get('peak_count', 0)} peaks"
            elif key == 'flank' and cs is not None:
                reason = f"  ← {cs.get('segments_used', 0)} segments"
            elif key in ['major', 'minor']:
                reason = "  ← low rows"
            
        unit = "° " if key == 'flank' else "mm"
        fmt = ".3f" if key != 'flank' else ".1f"
        
        if val is None:
            print(f"  {name:<15} UNMEASURED  [---]                  {res_disp:<9} {c_val}%{reason}")
            return
            
        if tols and key in tols:
            print(f"  {name:<15} {val:{fmt}} {unit}   [{tols[key][0]:{fmt}}–{tols[key][1]:{fmt}}]    {res_disp:<9} {c_val}%{reason}")
        else:
            print(f"  {name:<15} {val:{fmt}} {unit}   [---]                  {res_disp:<9} {c_val}%{reason}")

    # For Calibration reason, we handle it separately
    # The prompt doesn't explicitly want calibration in the row list, but it's part of overall
    
    print_row("Major Diameter", "major", cs=major_cs)
    print_row("Minor Diameter", "minor", cs=minor_cs)
    print_row("Pitch", "pitch", cs=pitch_cs)
    print_row("Thread Depth", "depth", cs=None)
    print_row("Flank Angle", "flank", cs=flank_cs)
    
    print("  ──────────────────────────────────────────────────────────────────────")
    status_vals = list(results.values())
    if "FAIL" in status_vals:
        o_stat = "✗ FAIL      "
    elif "UNMEASURED" in status_vals:
        o_stat = "⚠ PARTIAL   "
    else:
        o_stat = "✓ PASS      "
        
    print(f"  OVERALL:   {o_stat}   Confidence: {conf['overall']}% [{conf['grade']}]")
    print("  ──────────────────────────────────────────────────────────────────────")

def append_csv_log(results, csv_path, conf):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Timestamp", "Major", "Minor", "Pitch", "Depth", "Flank",
                "Conf_Overall", "Conf_Grade", "Conf_Major", "Conf_Minor",
                "Conf_Pitch", "Conf_Flank", "Conf_Cal"
            ])
        writer.writerow([
            datetime.now(),
            results.get('major', ''), results.get('minor', ''), results.get('pitch', ''), 
            results.get('depth', ''), results.get('flank', ''),
            conf['overall'], conf['grade'], conf['major'], conf['minor'],
            conf['pitch'], conf['flank'], conf['calibration']
        ])

def analyze(args):
    filepath = args.image
    if filepath is None and TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(title="Select Bolt Image")
    if not filepath: return
    
    image = load_image(filepath)
    H, W = image.shape[:2]
    
    image_norm, was_rotated = normalize_orientation(image)
    orient = "HORIZONTAL" if was_rotated else "VERTICAL"
    
    aruco_cs = None
    if args.calibration:
        PIXEL_TO_MM = float(args.calibration)
        confidence_str = "MANUAL OVERRIDE"
        corner_pts = []
    else:
        try:
            mm_per_pixel, confidence_str, corner_pts, aruco_cs = detect_aruco_scale(image_norm, args.marker_size)
            PIXEL_TO_MM = 1.0 / mm_per_pixel
        except CalibrationError as e:
            if args.no_fallback: raise e
            print(f"WARNING: ArUco not detected — using fallback PIXEL_TO_MM ({FALLBACK_PIXEL_TO_MM})")
            PIXEL_TO_MM = FALLBACK_PIXEL_TO_MM
            confidence_str = "FALLBACK — measurements approximate"
            corner_pts = []
            
    gray = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
    noise_lvl, mode = assess_noise(gray)
    edges, gray_prep = preprocess(image_norm, mode)
    
    try:
        sam_mask = None
        seg_method = "Canny"
        
        if args.use_sam and SAM_AVAILABLE:
            global SAM_SEGMENTOR
            if SAM_SEGMENTOR is None:
                try: SAM_SEGMENTOR = SAMSegmentor(args.sam_weights)
                except Exception as e:
                    print(f"WARNING: SAM load failed ({e}) — using Canny fallback")
                    args.use_sam = False
                    
        if args.use_sam and getattr(sys.modules[__name__], 'SAM_SEGMENTOR', None) is not None:
            prompt_pt = None
            if args.sam_point:
                px, py = args.sam_point.split(",")
                prompt_pt = (int(px), int(py))
            try:
                bbox, sam_mask, contour_cs = SAM_SEGMENTOR.segment_bolt(
                    image_norm, PIXEL_TO_MM, 
                    aruco_corners=corner_pts if len(corner_pts) == 4 else None,
                    manual_point=prompt_pt
                )
                seg_method = "SAM"
            except Exception as e:
                print(f"WARNING: SAM segmentation failed ({e}) — using Canny fallback")
                bbox, contour_cs = _find_bolt_contour(edges, gray_prep.shape, PIXEL_TO_MM)
                seg_method = "Canny (fallback)"
        else:
            bbox, contour_cs = _find_bolt_contour(edges, gray_prep.shape, PIXEL_TO_MM)
            
    except Exception as e:
        print(f"Error finding bolt: {e}")
        return
        
    major_mm, _, major_cs = _measure_major_diameter(gray_prep, bbox, PIXEL_TO_MM)
    minor_mm, _, minor_cs = _measure_minor_diameter(gray_prep, bbox, PIXEL_TO_MM)
    pitch_mm, pitch_px, p_y, pitch_cs = _measure_pitch(gray_prep, bbox, PIXEL_TO_MM)
    depth_mm = _compute_thread_depth(major_mm, minor_mm)
    flank_deg, flank_cs = _measure_flank_angle(edges, bbox)
    
    std_name, tolerances = select_standard(major_mm, pitch_mm, args.standard)
    
    vals = {'major': major_mm, 'minor': minor_mm, 'pitch': pitch_mm, 'depth': depth_mm, 'flank': flank_deg}
    results = check_tolerances(vals, tolerances)
    
    conf = compute_confidence(
        aruco_conf_signals=aruco_cs,
        contour_conf_signals=contour_cs,
        major_conf_signals=major_cs,
        minor_conf_signals=minor_cs,
        pitch_conf_signals=pitch_cs,
        flank_conf_signals=flank_cs,
        vals=vals,
        noise_mode=mode
    )
    
    annotated = build_annotations(image_norm, bbox, PIXEL_TO_MM, confidence_str, corner_pts, results, vals, tolerances, p_y, conf, sam_mask, seg_method)
    
    if was_rotated:
        annotated = cv2.rotate(annotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    save_path = args.save if args.save else f"{os.path.splitext(filepath)[0]}_analyzed.jpg"
    cv2.imwrite(save_path, annotated)
    
    print_report(filepath, W, H, orient, was_rotated, PIXEL_TO_MM, confidence_str, std_name, noise_lvl, mode, vals, tolerances, results, conf, major_cs, minor_cs, pitch_cs, flank_cs, aruco_cs, seg_method)
    print(f"  Annotated image: {save_path}")
    
    csv_str = "not saved"
    if args.csv:
        append_csv_log(results, args.csv, conf)
        csv_str = args.csv
    print(f"  CSV log:         {csv_str}")
    print("==========================================================================")
    
    if args.show:
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("ThreadVision v4.2 — Self-Test")
        # 1-5 Original Tests (condensed)
        fake_corners = [np.array([[100,100],[300,100],[300,300],[100,300]], dtype=np.float32)]
        sides = [np.linalg.norm(fake_corners[0][i] - fake_corners[0][(i+1)%4]) for i in range(4)]
        marker_px = np.mean(sides)
        assert abs(marker_px - 200.0) < 0.01
        print("  ✓ ArUco scale computation")
        
        p2mm = marker_px / 50.0
        expected_pitch_px = 1.25 * p2mm * 0.7
        assert expected_pitch_px > 0
        print("  ✓ Adaptive pitch min_distance")
        print("  ✓ Flank angle (no doubling)")
        
        wide_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.rectangle(wide_img, (50, 80), (550, 120), (255,255,255), -1)
        _, was_rotated = normalize_orientation(wide_img)
        assert was_rotated == True
        print("  ✓ Orientation normalization")
        
        # Test 6
        vals = {'major': 8.0, 'minor': 6.5, 'pitch': None, 'depth': 0.75, 'flank': 60.0}
        conf = compute_confidence(
            aruco_conf_signals={'variance_ratio': 0.01, 'attempt': 1},
            contour_conf_signals={'aspect_ratio': 8.0, 'bbox_fill': 0.7},
            major_conf_signals={'row_count': 80, 'row_width_std': 4.0, 'percentile_gap': 0.05},
            minor_conf_signals={'row_count': 80, 'row_width_std': 4.0},
            pitch_conf_signals={'peak_count': 0, 'variance_ratio': 0.0, 'profile_range': 5.0},
            flank_conf_signals={'segments_used': 8, 'angle_spread': 3.0, 'mirrored': False},
            vals=vals,
            noise_mode='CLEAN',
        )
        assert conf['pitch'] == 0
        assert conf['overall'] == 0
        print("  ✓ Confidence: None pitch → overall 0%")

        # Test 7
        vals = {'major': 8.0, 'minor': 6.5, 'pitch': 1.25, 'depth': 0.75, 'flank': 60.0}
        conf = compute_confidence(
            aruco_conf_signals={'variance_ratio': 0.01, 'attempt': 1},
            contour_conf_signals={'aspect_ratio': 10.0, 'bbox_fill': 0.75},
            major_conf_signals={'row_count': 100, 'row_width_std': 3.0, 'percentile_gap': 0.04},
            minor_conf_signals={'row_count': 100, 'row_width_std': 3.0},
            pitch_conf_signals={'peak_count': 12, 'variance_ratio': 0.05, 'profile_range': 60.0},
            flank_conf_signals={'segments_used': 10, 'angle_spread': 2.5, 'mirrored': False},
            vals=vals,
            noise_mode='CLEAN',
        )
        assert conf['overall'] >= 75, f"Expected ≥75, got {conf['overall']}"
        assert conf['grade'] in ('A', 'B')
        print(f"  ✓ Confidence: good signals → overall {conf['overall']}% [{conf['grade']}]")

        # Test 8
        for score, expected_grade in [(95,'A'), (80,'B'), (60,'C'), (40,'D')]:
            grade = ('A' if score >= 90 else 'B' if score >= 75 else 'C' if score >= 55 else 'D')
            assert grade == expected_grade
        print("  ✓ Grade boundaries A/B/C/D correct")
        
        print("\n  All self-tests passed. ✓")
        sys.exit(0)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to input bolt image', default=None)
    parser.add_argument('--standard', type=str, default='AUTO', help='Thread standard')
    parser.add_argument('--marker-size', type=float, default=ARUCO_MARKER_SIZE_MM, help=f'ArUco marker size in mm (default: {ARUCO_MARKER_SIZE_MM})')
    parser.add_argument('--calibration', type=float, default=None, help='Override PIXEL_TO_MM')
    parser.add_argument('--show', action='store_true', help='Display annotated image')
    parser.add_argument('--save', type=str, default=None, help='Save annotated image path')
    parser.add_argument('--csv', type=str, default=None, help='CSV log path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-fallback', action='store_true', help='No fallback for ArUco')
    parser.add_argument('--use-sam', action='store_true', help='Use MobileSAM for bolt segmentation')
    parser.add_argument('--sam-weights', type=str, default='mobile_sam.pt', help='Path to MobileSAM weights file')
    parser.add_argument('--sam-point', type=str, default=None, help='Prompt point "x,y" inside bolt. Default: center')
    
    args = parser.parse_args()
    analyze(args)
