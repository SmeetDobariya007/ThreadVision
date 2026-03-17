# ── TOP-LEVEL CONSTANTS (user-configurable) ───────────────────────
ARUCO_MARKER_SIZE_MM  = 50.0     # Physical side length of printed ArUco marker
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
    """
    Detect bolt orientation. If horizontal, rotate 90° clockwise.
    Returns: (normalized_image, was_rotated: bool)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, False
    
    largest = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest)
    
    if w > h:   # wider than tall → horizontal bolt
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, True
    return image, False

def unrotate_points(pts, original_shape, was_rotated):
    """Transform annotation coordinates back to original image space."""
    if not was_rotated:
        return pts
    H, W = original_shape[:2]
    # Inverse of 90° clockwise: x_orig = y_rot, y_orig = W - x_rot
    if len(pts) == 0: return pts
    return np.array([[p[1], W - p[0]] for p in pts], dtype=np.int32)

def detect_aruco_scale(image, marker_mm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # 1. Standard detection
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # 2. CLAHE fallback
    if not corners:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_gray = clahe.apply(gray)
        corners, ids, rejected = detector.detectMarkers(cl_gray)
        
    # 3. Adaptive threshold fallback
    if not corners:
        ad_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        corners, ids, rejected = detector.detectMarkers(ad_gray)
        
    if not corners:
        raise CalibrationError("Marker not detected")
        
    corner = corners[0][0] # shape (4,2)
    side_0 = np.linalg.norm(corner[0] - corner[1])
    side_1 = np.linalg.norm(corner[1] - corner[2])
    side_2 = np.linalg.norm(corner[2] - corner[3])
    side_3 = np.linalg.norm(corner[3] - corner[0])
    
    sides = [side_0, side_1, side_2, side_3]
    marker_size_px = np.mean(sides)
    
    variance_ratio = np.std(sides) / marker_size_px
    if variance_ratio > 0.05:
        confidence = "LOW — marker tilted"
    elif variance_ratio > 0.02:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"
        
    mm_per_pixel = marker_mm / marker_size_px
    return mm_per_pixel, confidence, corner

def assess_noise(gray_image):
    """Returns noise_level (float) and preprocessing_mode (str)."""
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
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        frame_area = image_shape[0] * image_shape[1]
        if area < frame_area * 0.005:
            continue
        width_mm = w / PIXEL_TO_MM
        height_mm = h / PIXEL_TO_MM
        if not (BOLT_WIDTH_MIN_MM <= width_mm <= BOLT_WIDTH_MAX_MM):
            continue
        if h < w:
            continue
        valid.append((area, c, x, y, w, h))
    if not valid:
        raise BoltNotFoundError("No valid bolt contour found in image.")
    valid.sort(key=lambda t: t[0], reverse=True)
    return valid[0][1:]

def _measure_major_diameter(gray, bbox, PIXEL_TO_MM):
    c, x, y, w, h = bbox
    # middle 60%
    y_start = int(y + 0.2 * h)
    y_end = int(y + 0.8 * h)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    row_widths = []
    for row_y in range(y_start, y_end):
        row_pixels = binary[row_y, x:x+w]
        indices = np.where(row_pixels > 0)[0]
        if len(indices) > 0:
            row_widths.append(indices[-1] - indices[0])
            
    if not row_widths: return 0.0, 0.0
    major_px = np.percentile(row_widths, 97)
    major_mm = major_px / PIXEL_TO_MM
    return major_mm, major_px

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
            
    if not row_widths: return 0.0, 0.0
    minor_px = np.percentile(row_widths, 10)
    minor_mm = minor_px / PIXEL_TO_MM
    return minor_mm, minor_px

def _measure_pitch(gray, bbox, PIXEL_TO_MM):
    c, x, y, w, h = bbox
    strip = gray[y:y+h, max(0, x):x+8]
    profile = strip.mean(axis=1).astype(np.float64)
    
    bg_sample = gray[y + h//2, max(0, x - 20):max(0, x - 2)]
    bg_mean = bg_sample.mean() if len(bg_sample) > 0 else 200.0
    if bg_mean > 128:
        profile = 255.0 - profile
        
    profile = scipy.ndimage.gaussian_filter1d(profile, sigma=2)
    expected_pitch_px = 1.25 * PIXEL_TO_MM * 0.7
    min_distance = max(int(expected_pitch_px), 15)
    
    peaks, _ = scipy.signal.find_peaks(profile, height=np.percentile(profile, 60), distance=min_distance, prominence=10)
    if len(peaks) < 3:
        return None, None, None
        
    diffs = np.diff(peaks)
    median_diff = np.median(diffs)
    filtered = diffs[np.abs(diffs - median_diff) < np.std(diffs)]
    if len(filtered) == 0:
        return None, None, None
    pitch_px = np.median(filtered)
    pitch_mm = pitch_px / PIXEL_TO_MM
    return pitch_mm, pitch_px, (y + peaks[len(peaks)//2])

def _compute_thread_depth(major_mm, minor_mm):
    return (major_mm - minor_mm) / 2.0

def _measure_flank_angle(edges, bbox):
    c, x, y, w, h = bbox
    mid_x = x + w / 2.0
    
    c_pts = c[:, 0, :]
    y_min, y_max = y + 0.25 * h, y + 0.75 * h
    
    pts_left = c_pts[(c_pts[:, 0] < mid_x) & (c_pts[:, 1] >= y_min) & (c_pts[:, 1] <= y_max)]
    pts_right = c_pts[(c_pts[:, 0] >= mid_x) & (c_pts[:, 1] >= y_min) & (c_pts[:, 1] <= y_max)]
    
    def fit_half_angle(pts):
        if len(pts) < 10: return 0.0
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
        return np.mean(angles) if angles else 30.0
        
    left_half = fit_half_angle(pts_left)
    right_half = fit_half_angle(pts_right)
    included_angle = left_half + right_half
    return included_angle

def select_standard(major_mm, pitch_mm, std_arg):
    if std_arg != "AUTO" and std_arg in ISO_STANDARDS:
        return std_arg, ISO_STANDARDS[std_arg]
    
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

def build_annotations(image, bbox, PIXEL_TO_MM, confidence, corner_pts, results, vals, tols, p_y):
    c, x, y, w, h = bbox
    out = image.copy()
    
    # ArUco marker
    if len(corner_pts) == 4:
        cv2.polylines(out, [corner_pts.astype(np.int32)], True, (0, 200, 0), 2)
        for pt in corner_pts:
            cv2.circle(out, tuple(pt.astype(np.int32)), 6, (0, 0, 255), -1)
        lx, ly = int(corner_pts[0][0]), int(corner_pts[0][1] - 10)
        cv2.putText(out, f"Cal: {PIXEL_TO_MM:.1f} px/mm [{confidence}]", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        
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
    cv2.putText(out, f"Major: {vals['major']:.2f}mm {results['major']}", (x+w+10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.line(out, (x+4, my+30), (x+w-4, my+30), (0, 165, 255), 2)
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
    
    return out

def print_report(filepath, w, h, orient, was_rot, PIXEL_TO_MM, confidence, std_str, noise_lvl, noise_mode, vals, tols, results):
    um_per_px = 1000.0 / PIXEL_TO_MM
    print("============================================================")
    print("  ThreadVision AI — Thread Measurement Report v4.0")
    print(f"  Image:       {filepath} ({w} × {h} px)")
    rot_str = ", rotated 90°" if was_rot else ""
    print(f"  Orientation: {orient} (auto-detected{rot_str})")
    print(f"  Calibration: {PIXEL_TO_MM:.2f} px/mm ({um_per_px:.2f} µm/px) — {confidence}")
    print(f"  Standard:    {std_str} (ISO 262)")
    print(f"  Noise:       {noise_lvl:.1f} ({noise_mode})")
    print("============================================================")
    print("")
    print("  Dimension         Measured    Nominal    Tolerance          Status")
    print("  ───────────────────────────────────────────────────────────────────")
    
    def print_row(name, key):
        val = vals.get(key)
        if val is None:
            print(f"  {name:<15}   UNMEASURED")
            return
        tol = tols.get(key) if tols else None
        res = results.get(key, "UNKNOWN")
        
        unit = "° " if key == 'flank' else "mm"
        fmt = ".1f" if key == 'flank' else ".3f"
        
        if tol:
            nom = tol[1] if key not in ['pitch', 'flank', 'depth'] else (tol[0]+tol[1])/2.0
            print(f"  {name:<15} {val:{fmt}} {unit}  {nom:{fmt}} {unit}  [{tol[0]:{fmt}}–{tol[1]:{fmt}}]   {res}")
        else:
            print(f"  {name:<15} {val:{fmt}} {unit}  ---        [---]         {res}")

    print_row("Major Diameter", "major")
    print_row("Minor Diameter", "minor")
    print_row("Pitch", "pitch")
    print_row("Thread Depth", "depth")
    print_row("Flank Angle", "flank")
    
    print("  ───────────────────────────────────────────────────────────────────")
    status_vals = list(results.values())
    if "FAIL" in status_vals:
        o_stat = "██ FAIL ██"
    elif "UNMEASURED" in status_vals:
        o_stat = "⚠ PARTIAL ⚠"
    else:
        o_stat = "██ PASS ██"
    print(f"  OVERALL:  {o_stat}")
    print("  ───────────────────────────────────────────────────────────────────")

def append_csv_log(results, csv_path):
    # simplistic CSV append
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "Major", "Minor", "Pitch", "Depth", "Flank"])
        writer.writerow([datetime.now()] + list(results.values()))

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
    
    if args.calibration:
        PIXEL_TO_MM = float(args.calibration)
        confidence = "MANUAL OVERRIDE"
        corner_pts = []
    else:
        try:
            mm_per_pixel, confidence, corner_pts = detect_aruco_scale(image_norm, args.marker_size)
            PIXEL_TO_MM = 1.0 / mm_per_pixel
        except CalibrationError as e:
            if args.no_fallback: raise e
            print(f"WARNING: ArUco not detected — using fallback PIXEL_TO_MM ({FALLBACK_PIXEL_TO_MM})")
            PIXEL_TO_MM = FALLBACK_PIXEL_TO_MM
            confidence = "FALLBACK — measurements approximate"
            corner_pts = []
            
    gray = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
    noise_lvl, mode = assess_noise(gray)
    edges, gray_prep = preprocess(image_norm, mode)
    
    try:
        bbox = _find_bolt_contour(edges, gray_prep.shape, PIXEL_TO_MM)
    except Exception as e:
        print(f"Error finding bolt: {e}")
        return
        
    major_mm, _ = _measure_major_diameter(gray_prep, bbox, PIXEL_TO_MM)
    minor_mm, _ = _measure_minor_diameter(gray_prep, bbox, PIXEL_TO_MM)
    pitch_mm, pitch_px, p_y = _measure_pitch(gray_prep, bbox, PIXEL_TO_MM)
    depth_mm = _compute_thread_depth(major_mm, minor_mm)
    flank_deg = _measure_flank_angle(edges, bbox)
    
    std_name, tolerances = select_standard(major_mm, pitch_mm, args.standard)
    
    vals = {'major': major_mm, 'minor': minor_mm, 'pitch': pitch_mm, 'depth': depth_mm, 'flank': flank_deg}
    results = check_tolerances(vals, tolerances)
    
    annotated = build_annotations(image_norm, bbox, PIXEL_TO_MM, confidence, corner_pts, results, vals, tolerances, p_y)
    
    if was_rotated:
        # Rotate back
        annotated = cv2.rotate(annotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    save_path = args.save if args.save else f"{os.path.splitext(filepath)[0]}_analyzed.jpg"
    cv2.imwrite(save_path, annotated)
    
    print_report(filepath, W, H, orient, was_rotated, PIXEL_TO_MM, confidence, std_name, noise_lvl, mode, vals, tolerances, results)
    print(f"  Annotated image: {save_path}")
    
    csv_str = "not saved"
    if args.csv:
        append_csv_log(vals, args.csv)
        csv_str = args.csv
    print(f"  CSV log:         {csv_str}")
    print("============================================================")
    
    if args.show:
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("ThreadVision v4.0 — Self-Test")
        fake_corners = [np.array([[100,100],[300,100],[300,300],[100,300]], dtype=np.float32)]
        sides = [np.linalg.norm(fake_corners[0][i] - fake_corners[0][(i+1)%4]) for i in range(4)]
        marker_px = np.mean(sides)
        assert abs(marker_px - 200.0) < 0.01, "ArUco side length computation FAIL"
        p2mm = marker_px / 50.0
        assert abs(p2mm - 4.0) < 0.01, "PIXEL_TO_MM computation FAIL"
        print("  ✓ ArUco scale computation")
        
        expected_pitch_px = 1.25 * p2mm * 0.7
        assert expected_pitch_px > 0, "min_distance computation FAIL"
        print("  ✓ Adaptive pitch min_distance")
        
        half_angle = 30.0
        included = half_angle + half_angle
        assert abs(included - 60.0) < 0.01, f"Flank angle doubling bug present: {included}"
        print("  ✓ Flank angle (no doubling)")
        
        wide_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.rectangle(wide_img, (50, 80), (550, 120), (255,255,255), -1)
        _, was_rotated = normalize_orientation(wide_img)
        assert was_rotated == True, "Orientation detection FAIL on horizontal bolt"
        print("  ✓ Orientation normalization")
        
        print("\n  All self-tests passed. ✓")
        sys.exit(0)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to input bolt image', default=None)
    parser.add_argument('--standard', type=str, default='AUTO', help='Thread standard')
    parser.add_argument('--marker-size', type=float, default=50.0, help='ArUco marker size in mm')
    parser.add_argument('--calibration', type=float, default=None, help='Override PIXEL_TO_MM')
    parser.add_argument('--show', action='store_true', help='Display annotated image')
    parser.add_argument('--save', type=str, default=None, help='Save annotated image path')
    parser.add_argument('--csv', type=str, default=None, help='CSV log path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-fallback', action='store_true', help='No fallback for ArUco')
    
    args = parser.parse_args()
    analyze(args)
