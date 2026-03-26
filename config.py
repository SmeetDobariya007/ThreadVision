# config.py — ThreadVision AI Configuration
# ============================================
# Single source of truth for all system constants.
# Every other module imports from this file.
# Edit PIXEL_TO_MM after running calibrate.py
#
# Author: ThreadVision AI Team
# Target: Raspberry Pi 4 + Pi Camera v2
# Standard: M8 × 1.25 ISO 262 metric thread

import os

# === BASE DIRECTORY ===
# Auto-detect: use project directory on any platform
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === CALIBRATION ===
PIXEL_TO_MM = 120.0   # pixels per mm — SET THIS WITH calibrate.py BEFORE USE
                       # Approximate starting value only. Must be measured empirically.
                       # Valid range: 80–200 px/mm for Pi Cam v2 at macro distances.
ARUCO_MARKER_SIZE_MM = 50.0  # Physical side length of printed ArUco marker (DICT_4X4_50)

# === THREAD STANDARD: M8 × 1.25 (ISO 262) ===
THREAD_STANDARD = "M8x1.25"
NOMINAL_VALUES = {
    'major_diameter': 8.000,   # mm — outer diameter at thread crests
    'minor_diameter': 6.647,   # mm — inner diameter at thread roots
    'pitch':          1.250,   # mm — distance between adjacent crests
    'thread_depth':   0.677,   # mm — radial depth of one thread tooth
    'flank_angle':    60.0,    # degrees — included angle of thread profile
}
TOLERANCES = {
    # Key: (lower_bound, upper_bound) in mm (degrees for flank_angle)
    # Source: ISO 68-1 / ASME B1.2 for 6g tolerance class
    'major_diameter': (7.972, 8.000),
    'minor_diameter': (6.466, 6.647),
    'pitch':          (1.225, 1.275),
    'thread_depth':   (0.613, 0.677),
    'flank_angle':    (59.0,  61.0),   # degrees, not mm
}

# === GPIO PIN NUMBERS (BCM numbering) ===
PIN_BACKLIGHT = 17    # GPIO 17 → LED backlight relay (OUTPUT)
PIN_LED_PASS  = 27    # GPIO 27 → Green LED (OUTPUT)
PIN_LED_FAIL  = 22    # GPIO 22 → Red LED (OUTPUT)
PIN_TRIGGER   = 18    # GPIO 18 → Manual trigger button (INPUT, active LOW, pull-up)

# === CAMERA SETTINGS ===
CAPTURE_WIDTH   = 3280   # Full-resolution capture for maximum measurement accuracy
CAPTURE_HEIGHT  = 2464   # Sony IMX219 sensor maximum
PREVIEW_WIDTH   = 640    # Live preview in GUI
PREVIEW_HEIGHT  = 480    # Scaled for performance

# === IMAGE PROCESSING THRESHOLDS ===
CANNY_LOW       = 50     # Canny edge detector lower threshold
CANNY_HIGH      = 150    # Canny edge detector upper threshold
BLUR_KERNEL     = (5, 5) # Gaussian blur kernel size (must be odd)
CLAHE_CLIP      = 2.0    # CLAHE contrast limiting clip value
CLAHE_GRID      = (8, 8) # CLAHE tile grid size

# === PEAK DETECTION PARAMETERS (for pitch measurement) ===
PEAK_MIN_HEIGHT_RATIO = 0.4   # Fraction of max intensity to qualify as a crest peak
PEAK_MIN_DISTANCE_PX  = 50    # Minimum pixels between adjacent crests

# === DATABASE ===
LOG_DIR  = os.path.join(BASE_DIR, "logs")
DB_PATH  = os.path.join(LOG_DIR, "inspection_log.db")
CSV_PATH = os.path.join(LOG_DIR, "export.csv")

# === TIMING ===
LED_ON_DURATION_MS = 1500     # How long to hold pass/fail LEDs after result (ms)

# === PLATFORM DETECTION ===
IS_RASPBERRY_PI = os.path.exists("/proc/device-tree/model")


# ---------------------------------------------------------------------------
# Self-test: run `python config.py` to verify all constants load correctly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ThreadVision AI — Configuration Self-Test")
    print("=" * 60)
    print(f"  Base directory   : {BASE_DIR}")
    print(f"  Thread standard  : {THREAD_STANDARD}")
    print(f"  PIXEL_TO_MM      : {PIXEL_TO_MM} px/mm")
    print(f"  µm per pixel     : {1e3 / PIXEL_TO_MM:.2f} µm/px")
    print(f"  Camera resolution: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
    print(f"  DB path          : {DB_PATH}")
    print(f"  Is Raspberry Pi  : {IS_RASPBERRY_PI}")
    print()
    print("  Nominal values:")
    for k, v in NOMINAL_VALUES.items():
        unit = "°" if k == "flank_angle" else "mm"
        lo, hi = TOLERANCES[k]
        print(f"    {k:20s} = {v:7.3f} {unit}  [tol: {lo:.3f} – {hi:.3f}]")
    print()
    print("  GPIO pins:")
    print(f"    Backlight : GPIO {PIN_BACKLIGHT}")
    print(f"    Pass LED  : GPIO {PIN_LED_PASS}")
    print(f"    Fail LED  : GPIO {PIN_LED_FAIL}")
    print(f"    Trigger   : GPIO {PIN_TRIGGER}")
    print()
    print("✓ All constants loaded successfully.")
