# ThreadVision AI — Image Analysis Mode
## Master Prompt v2.0 — Upload-Based Thread Measurement with Ruler Scale
### Optimized for Extended Thinking AI (Claude Sonnet 4.6 Thinking / Gemini 2.5 Pro)
### Use Case: Single image upload → auto-calibrate from ruler → measure all 5 dimensions

---

## INSTRUCTION TO THE AI MODEL

You are a senior computer vision engineer and precision metrology specialist. You have deep expertise in:
- Sub-pixel edge detection and geometric measurement from 2D images
- Automatic ruler/scale detection in macro photography
- ISO 68-1 / ASME B1.2 thread metrology standards
- Python 3.10+ with OpenCV 4.x, NumPy, SciPy
- Robust algorithm design for real-world noisy image data

Your task is to write a **single, self-contained Python script** called `analyze_thread_image.py` that:
1. Accepts a user-supplied image file (JPEG/PNG) as a command-line argument
2. Automatically detects a millimetre ruler scale anywhere in the image
3. Uses the detected ruler to calibrate pixel-to-mm ratio (no manual input needed)
4. Measures all 5 critical thread dimensions from the bolt silhouette
5. Compares measurements to ISO M8×1.25 tolerances
6. Outputs a clean PASS/FAIL report with all values and a visual annotated result image

**THINKING REQUIREMENT — reason through ALL of these before writing any code:**

1. How do you robustly detect a ruler scale in an image that may have varying contrast, partial occlusion, or different orientations?
2. How do you distinguish ruler tick marks from thread edges when both are dark on light background?
3. What is the correct approach when the ruler is vertical vs horizontal in the image?
4. The bolt and ruler may be at different depths — how does this affect pixel-to-mm accuracy?
5. What happens if the ruler detection fails — how do you fall back gracefully?
6. The thread edge profile (left edge vs right edge vs center column) — which gives the most reliable pitch signal and why?
7. How do you handle partial images where only some thread periods are visible?
8. What is the mathematically correct way to compute flank angle from contour edge points without the 2× doubling error?
9. How do you validate that a computed measurement is physically reasonable before reporting it?
10. What visual annotations on the output image best communicate confidence to the user?

---

## PART 1 — PROBLEM DEFINITION

### What this script solves

A user photographs a threaded bolt or screw next to a millimetre ruler. They run:

```bash
python analyze_thread_image.py --image bolt_photo.jpg
```

The script automatically:
- Finds the ruler in the image
- Reads the scale from it (pixels per mm)
- Finds the bolt/thread silhouette
- Measures 5 thread dimensions in mm
- Outputs PASS/FAIL against ISO M8×1.25 tolerances
- Saves an annotated image showing all measurements visually

### Target image characteristics
- Bolt photographed against a bright background (backlit or white paper behind)
- Millimetre ruler visible somewhere in the frame (left, right, top, or bottom)
- Ruler shows at least 5mm of graduated markings
- Bolt thread region has at least 5 visible thread periods
- Image resolution: minimum 800×600, typically 3000×2000+
- Bolt orientation: primarily vertical (thread axis top-to-bottom)

---

## PART 2 — SCRIPT ARCHITECTURE

### Single file: `analyze_thread_image.py`

The script is ONE self-contained file with no imports from other project modules.
All constants are defined at the top of the file.
All functions are defined in the file.
The script runs end-to-end from a single command.

### Required structure:

```
analyze_thread_image.py
│
├── CONSTANTS BLOCK
│   ├── ISO M8×1.25 tolerances
│   ├── Algorithm parameters
│   └── Output settings
│
├── STEP 1: load_and_validate_image(path)
│   └── Load image, validate it's usable, return BGR array
│
├── STEP 2: detect_ruler_and_calibrate(image)
│   ├── find_ruler_region(image) → (x, y, w, h) of ruler
│   ├── extract_tick_marks(ruler_region) → list of tick positions in pixels
│   ├── compute_pixel_to_mm(tick_positions) → float (px/mm)
│   └── Returns: PIXEL_TO_MM float, ruler_bbox, confidence score
│
├── STEP 3: find_bolt_silhouette(image, ruler_bbox)
│   ├── Exclude ruler region from analysis
│   ├── Find largest contour = bolt outline
│   └── Returns: contour, bounding_box (x, y, w, h)
│
├── STEP 4: preprocess_for_measurement(image, bolt_bbox)
│   ├── Crop to bolt region with padding
│   ├── Grayscale → GaussianBlur → CLAHE → Canny
│   └── Returns: edges, gray (cropped to bolt region)
│
├── STEP 5: measure_all_dimensions(edges, gray, bolt_bbox, px_per_mm)
│   ├── measure_major_diameter()
│   ├── measure_pitch()          ← CRITICAL: use edge strip, not center column
│   ├── measure_minor_diameter()
│   ├── derive_thread_depth()
│   └── measure_flank_angle()    ← CRITICAL: no 2× doubling error
│
├── STEP 6: check_tolerances(measurements)
│   └── Returns: overall PASS/FAIL, per-dimension results
│
├── STEP 7: generate_annotated_image(image, results, calibration)
│   ├── Draw bolt bounding box
│   ├── Draw major diameter arrow + label
│   ├── Draw minor diameter arrow + label
│   ├── Draw pitch bracket showing one period
│   ├── Draw ruler detection overlay
│   ├── Draw PASS (green) or FAIL (red) result banner
│   └── Returns: annotated BGR image
│
├── STEP 8: print_report(measurements, tolerance_results, calibration)
│   └── Clean formatted terminal output
│
└── main() — CLI entry point with argparse
```

---

## PART 3 — RULER DETECTION ALGORITHM (MOST CRITICAL)

This is the hardest algorithmic problem in the script. The ruler detection determines accuracy of every subsequent measurement.

### Strategy: Multi-method detection with confidence scoring

```
Method 1 — Hough Line Detection (primary):
  - Convert image to grayscale
  - Apply Canny edge detection with high thresholds (100, 200)
  - Run cv2.HoughLinesP to find straight line segments
  - Cluster collinear lines → find the longest straight edge
  - This is likely the ruler baseline

Method 2 — Periodic tick mark detection (secondary + calibration):
  - Once ruler baseline is found, scan perpendicular to it
  - Find regularly-spaced high-contrast edges = tick marks
  - The spacing between tick marks = 1mm (small ticks) or 5mm / 10mm (large ticks)
  - Median spacing of detected ticks → most reliable px/mm estimate

Method 3 — Template approach (fallback):
  - Look for a region with high density of parallel short line segments
  - Characteristic of ruler tick mark patterns
  - Less accurate but works when Methods 1+2 fail

Confidence scoring:
  - HIGH confidence: >10 evenly-spaced ticks detected, variance <5%
  - MEDIUM confidence: 5–10 ticks, variance <15%
  - LOW confidence: <5 ticks or variance >15% — warn user
  - FAILED: no ruler detected — prompt user for manual mm input
```

### Tick mark spacing to PIXEL_TO_MM conversion:

```python
# For a ruler with 1mm graduations:
# median_tick_spacing_px = median distance between adjacent small ticks
# PIXEL_TO_MM = median_tick_spacing_px / 1.0

# For a ruler with mixed 1mm/5mm/10mm ticks:
# First classify ticks by size (small=1mm, medium=5mm, large=10mm)
# Use the most numerous class for median computation
# This is more robust than using large ticks only

# Validation:
# Physical constraint: PIXEL_TO_MM must be 20–500 px/mm for macro photography
# Below 20: image is too far away for thread measurement
# Above 500: image resolution exceeds typical camera capability
# If outside range: raise CalibrationError with user-friendly message
```

### Ruler orientation handling:

```python
# Detect ruler orientation:
# - Measure aspect ratio of ruler bounding region
# - If width >> height: horizontal ruler → tick marks are vertical lines
# - If height >> width: vertical ruler → tick marks are horizontal lines
# - Apply HoughLinesP perpendicular to the detected baseline direction

# Critical: ruler may be at a slight angle (< 5° tilt)
# Use the angle of the detected baseline for rotation correction
# Apply cv2.warpAffine to deskew the ruler strip before tick detection
```

---

## PART 4 — MEASUREMENT ALGORITHMS (CORRECTED SPECIFICATIONS)

### PIXEL_TO_MM convention
```python
mm_per_px = 1.0 / PIXEL_TO_MM
# All measurements: distance_mm = distance_px * mm_per_px
```

### Dimension 1: Major Diameter

```python
def measure_major_diameter(contour, bbox, mm_per_px):
    """
    Major diameter = maximum width across thread crests.
    
    Algorithm:
    1. Get contour points in the middle 60% of bolt height (avoid head/tip)
    2. For each row y in the middle region:
       - Find min_x and max_x of contour points at that y
       - row_width = max_x - min_x
    3. major_diameter_px = np.percentile(row_widths, 97)
       WHY 97th not 90th: ISO tolerance is ±0.028mm = ±3.4px at 120px/mm
       Using 97th percentile is more accurate than 90th for tight tolerances
    4. major_diameter_mm = major_diameter_px * mm_per_px
    
    Validation:
    - Must be between 4mm and 20mm (reasonable bolt range)
    - If outside range: raise MeasurementError
    """
```

### Dimension 2: Pitch — CORRECTED (edge strip method)

```python
def measure_pitch(gray, bbox, mm_per_px):
    """
    Pitch = distance between adjacent thread crests.
    
    CRITICAL DESIGN DECISION:
    Do NOT scan the center column of the bolt.
    The center column is inside the dark silhouette — uniform dark,
    no periodic variation. find_peaks() will find nothing or noise.
    
    CORRECT approach: scan the LEFT EDGE STRIP of the bolt.
    Thread crests at the bolt edge create intensity peaks where
    background light bleeds in. Thread roots are darker.
    The edge strip has a clear periodic intensity signal.
    
    Algorithm:
    1. Define edge strip: x_strip = bolt_bbox_x to bolt_bbox_x + 8
       (8-pixel-wide strip along the LEFT edge of the bolt)
    2. Extract strip from grayscale: strip = gray[y:y+h, x:x+8]
    3. Average horizontally: profile = strip.mean(axis=1)
       (averaging 8 pixels reduces noise vs single column)
    4. Determine background brightness:
       bg_sample = gray[y + h//2, max(0, x-20):x-2]
       bg_mean = bg_sample.mean() if len(bg_sample) > 0 else 200.0
    5. Invert if bolt is dark on light background:
       if bg_mean > 128: profile = 255.0 - profile
       (thread crests become peaks after inversion)
    6. Smooth: profile_smooth = uniform_filter1d(profile, size=5)
    7. Peak detection:
       height_threshold = profile_smooth.min() + 
                          PEAK_MIN_HEIGHT_RATIO * profile_smooth.ptp()
       peaks, _ = find_peaks(profile_smooth,
                              height=height_threshold,
                              distance=PEAK_MIN_DISTANCE_PX,
                              prominence=profile_smooth.ptp() * 0.1)
    8. Require minimum 3 peaks (need at least 2 intervals)
       If < 3 peaks: return None, flag as UNMEASURED
    9. Inter-peak distances: diffs = np.diff(peaks).astype(float)
   10. Outlier removal: keep diffs within 1.5*IQR of median
       (IQR method is more robust than std dev for small samples)
   11. pitch_px = np.median(filtered_diffs)
   12. pitch_mm = pitch_px * mm_per_px
   13. Variance check: if std/median > 0.2: flag "HIGH VARIANCE — POSSIBLE DAMAGE"
    
    Returns: (pitch_mm, peaks_array, profile_array)
    pitch_mm is None if unmeasurable.
    """
```

### Dimension 3: Minor Diameter

```python
def measure_minor_diameter(edges, bbox, mm_per_px):
    """
    Minor diameter = width at thread roots (minimum functional width).
    
    Algorithm:
    1. Scan rows in the middle 60% of bolt height
    2. For each row:
       - Find edge pixels: positions = np.nonzero(edges[row, x:x+w])[0]
       - If >= 2 edge pixels: width_px = positions[-1] - positions[0]
    3. Collect all valid row widths
    4. Trim top/bottom 10% of width list (remove outliers)
    5. minor_diameter_px = np.percentile(trimmed_widths, 10)
       (10th percentile = thread root width, not minimum which is noise-prone)
    6. minor_diameter_mm = minor_diameter_px * mm_per_px
    
    Physical validation:
    - minor_diameter must be < major_diameter
    - minor_diameter must be > 0.5 * major_diameter
    - If violated: return None with warning
    """
```

### Dimension 4: Thread Depth (derived)

```python
# Simple derivation — no direct measurement needed:
thread_depth_mm = (major_diameter_mm - minor_diameter_mm) / 2.0

# Validation:
# Must be positive (minor < major)
# For M8: expected ~0.677mm
# If negative: measurement error, report as None with warning
```

### Dimension 5: Flank Angle — CORRECTED (no 2× doubling error)

```python
def measure_flank_angle(edges, bbox, mm_per_px):
    """
    Flank angle = included angle of thread tooth (60° for ISO metric).
    
    CRITICAL CORRECTION: The previous implementation had a 2× doubling error.
    It computed angle_from_vertical for each flank segment, then multiplied by 2.
    This gave ~120-140° instead of ~60°.
    
    CORRECT algorithm:
    1. Find contour points in the FULL bolt ROI (not split left/right)
    2. Focus on middle 40% of bolt height (avoid head and tail effects)
    3. For each point on the LEFT edge (x < bolt_center_x):
       - These points trace the FULL tooth profile: rising flank + falling flank
    4. Segment the left-edge profile into individual tooth flanks:
       a. Sort points by Y coordinate (top to bottom)
       b. For each consecutive pair of rows, compute dX/dY (local slope)
       c. Sign change in dX/dY = transition from rising to falling flank
       d. Group consecutive same-sign slopes into segments
    5. Fit a line to each flank segment using np.polyfit(y_pts, x_pts, 1)
       (fit x as function of y — avoids vertical line instability)
    6. For each segment:
       slope = fit_coefficients[0]  # dx/dy
       half_angle = degrees(arctan(abs(slope)))  # angle from vertical
    7. Separate rising and falling half-angles:
       rising_half_angles = [angles from rising flank segments]
       falling_half_angles = [angles from falling flank segments]
    8. CORRECT computation:
       avg_rising = np.median(rising_half_angles)   # e.g. ~30° for ISO
       avg_falling = np.median(falling_half_angles) # e.g. ~30° for ISO
       flank_angle = avg_rising + avg_falling        # = 60° for ISO
       
       NOTE: The included angle = sum of the two half-angles, NOT 2× one half-angle.
       For symmetric threads: avg_rising ≈ avg_falling ≈ 30°
       So included angle = 30 + 30 = 60° ✓
       
       Previous bug: 2 × avg_half_angle = 2 × 69° = 138° ✗
    
    9. Require minimum 3 rising + 3 falling segments for confidence
   10. Validate: flank_angle must be between 45° and 75°
       If outside range: return value but add SUSPECT warning
    
    Returns: (flank_angle_degrees, rising_angles_list, falling_angles_list)
    """
```

---

## PART 5 — ISO TOLERANCE CONSTANTS

```python
# M8 × 1.25 — ISO 262 / ISO 68-1 Tolerances
THREAD_STANDARD = "M8x1.25"

TOLERANCES = {
    'major_diameter': (7.972, 8.000),   # mm
    'minor_diameter': (6.466, 6.647),   # mm
    'pitch':          (1.225, 1.275),   # mm
    'thread_depth':   (0.613, 0.677),   # mm
    'flank_angle':    (59.0,  61.0),    # degrees
}

NOMINAL = {
    'major_diameter': 8.000,
    'minor_diameter': 6.647,
    'pitch':          1.250,
    'thread_depth':   0.677,
    'flank_angle':    60.0,
}

# Algorithm parameters
PEAK_MIN_HEIGHT_RATIO = 0.4   # fraction of profile range
PEAK_MIN_DISTANCE_PX  = 40    # minimum pixels between adjacent crest peaks
CANNY_LOW             = 50
CANNY_HIGH            = 150
BLUR_KERNEL           = (5, 5)
CLAHE_CLIP            = 2.0
CLAHE_GRID            = (8, 8)

# Calibration validation bounds
PIXEL_TO_MM_MIN = 20    # px/mm — below this, image too far away
PIXEL_TO_MM_MAX = 500   # px/mm — above this, physically unreasonable

# Ruler detection parameters
RULER_MIN_TICKS       = 5     # minimum tick marks needed for calibration
RULER_TICK_VARIANCE   = 0.15  # max allowed variance in tick spacing (15%)
HOUGH_THRESHOLD       = 50    # minimum votes for Hough line detection
HOUGH_MIN_LINE_LEN    = 100   # minimum ruler line length in pixels
```

---

## PART 6 — TOLERANCE CHECK AND REPORT

```python
def check_tolerances(measurements):
    """
    Compare each measured dimension against ISO M8×1.25 tolerance bands.
    
    Rules:
    - If measurement is None (UNMEASURED): status = "UNMEASURED", 
      does NOT contribute to FAIL. Only measured-but-out-of-tolerance = FAIL.
    - If measurement is within [low, high]: status = "PASS"
    - If measurement is outside [low, high]: status = "FAIL"
    - overall = "PASS" only if ALL measured dimensions pass
    - overall = "FAIL" if ANY measured dimension fails
    - overall = "PARTIAL" if some dimensions are UNMEASURED (no FAIL)
    
    Returns:
        dict with keys:
            'overall': "PASS" | "FAIL" | "PARTIAL"
            'dimensions': {
                dim_name: {
                    'value': float or None,
                    'status': "PASS" | "FAIL" | "UNMEASURED",
                    'deviation': float (mm from nominal) or None,
                    'tolerance_band': (low, high),
                }
            }
    """
```

### Terminal report format:

```
============================================================
  ThreadVision AI — Thread Measurement Report
  Image: bolt_photo.jpg
  Calibration: 127.3 px/mm (7.85 µm/px) — HIGH confidence (14 ticks)
  Thread Standard: M8×1.25 (ISO 262)
============================================================

  Dimension         Measured    Nominal    Tolerance        Status
  ─────────────────────────────────────────────────────────────────
  Major Diameter    7.994 mm    8.000 mm   [7.972–8.000]    ✓ PASS
  Minor Diameter    6.521 mm    6.647 mm   [6.466–6.647]    ✓ PASS
  Pitch             1.247 mm    1.250 mm   [1.225–1.275]    ✓ PASS
  Thread Depth      0.637 mm    0.677 mm   [0.613–0.677]    ✓ PASS
  Flank Angle       59.8°       60.0°      [59.0°–61.0°]    ✓ PASS

  ─────────────────────────────────────────────────────────────────
  OVERALL RESULT:  ██  PASS  ██
  ─────────────────────────────────────────────────────────────────

  Annotated image saved: bolt_photo_analyzed.jpg
============================================================
```

---

## PART 7 — ANNOTATED OUTPUT IMAGE

The script must save a visual result image that communicates the analysis clearly. Annotations must include:

```python
def generate_annotated_image(original_image, measurements, calibration, tolerance_results):
    """
    Draw the following on a copy of the original image:
    
    1. RULER DETECTION OVERLAY:
       - Semi-transparent colored rectangle over detected ruler region
       - Label: "Scale: X.X px/mm (confidence: HIGH/MED/LOW)"
    
    2. BOLT BOUNDING BOX:
       - Rectangle around detected bolt
       - Color: green if PASS, red if FAIL, yellow if PARTIAL
    
    3. MAJOR DIAMETER ANNOTATION:
       - Horizontal double-headed arrow at widest crest point
       - Label: "Major Ø: X.XXX mm [PASS/FAIL]"
    
    4. MINOR DIAMETER ANNOTATION:
       - Horizontal double-headed arrow at narrowest root point
       - Label: "Minor Ø: X.XXX mm [PASS/FAIL]"
    
    5. PITCH ANNOTATION:
       - Vertical bracket spanning exactly one thread period
       - Label: "Pitch: X.XXX mm [PASS/FAIL]"
       - Arrow pointing to edge strip scan region
    
    6. PEAK MARKERS (if pitch detected):
       - Small circles at each detected crest peak position
       - Color: cyan
    
    7. RESULT BANNER:
       - Semi-transparent rectangle at top of image
       - Large text: "PASS" (green) or "FAIL" (red)
       - Thread standard label: "ISO M8×1.25"
    
    Font: cv2.FONT_HERSHEY_SIMPLEX, size scaled to image resolution
    Line thickness: scaled to image resolution (max 3px for 4K, 1px for 800px)
    
    Save as: {original_filename}_analyzed.{extension}
    Also display using cv2.imshow() if --show flag is set
    """
```

---

## PART 8 — CLI INTERFACE

```bash
# Basic usage:
python analyze_thread_image.py --image bolt_photo.jpg

# With options:
python analyze_thread_image.py \
    --image bolt_photo.jpg \
    --standard M8x1.25 \
    --show \
    --output analyzed_result.jpg \
    --verbose

# Arguments:
# --image PATH      Required. Path to input image file.
# --standard STR    Thread standard. Default: M8x1.25. 
#                   Future: M6x1.0, M10x1.5 etc.
# --show            Display annotated image in OpenCV window after analysis.
# --output PATH     Custom output path for annotated image.
# --verbose         Show DEBUG-level logging (algorithm internals).
# --no-ruler        Skip ruler detection, prompt for manual px/mm input.
# --calibration F   Manually specify PIXEL_TO_MM value (skips ruler detection).
```

---

## PART 9 — ERROR HANDLING

Every failure must produce a clear, actionable user message. No silent exceptions.

```python
class ThreadVisionError(Exception):
    """Base error with user-friendly message."""
    pass

class ImageLoadError(ThreadVisionError):
    """File not found, corrupt, or unsupported format."""
    pass

class CalibrationError(ThreadVisionError):
    """Ruler not detected or calibration out of physical bounds."""
    # User message example:
    # "Could not detect a ruler scale in this image.
    #  Options:
    #  1. Ensure the ruler is clearly visible with at least 5mm of markings
    #  2. Use --calibration 120.0 to manually specify pixels-per-mm
    #  3. Use --no-ruler to be prompted for the value interactively"
    pass

class BoltNotFoundError(ThreadVisionError):
    """No bolt/thread silhouette detected in the image."""
    # User message example:
    # "Could not detect a bolt silhouette.
    #  Ensure the bolt is photographed against a bright background
    #  and occupies at least 5% of the image area."
    pass

class MeasurementError(ThreadVisionError):
    """Specific dimension cannot be computed from this image."""
    pass
```

---

## PART 10 — QUALITY REQUIREMENTS

The generated script must meet all of these:

1. **Single file** — zero imports from other ThreadVision modules. Only stdlib + opencv + numpy + scipy.
2. **Fully runnable** — `python analyze_thread_image.py --image test.jpg` works with no additional setup beyond pip installs.
3. **Handles real photos** — not just synthetic images. Must work on images like: a bolt next to a metal ruler, shot on a white paper background, with natural lighting variation.
4. **Graceful degradation** — if ruler detection fails, falls back to `--calibration` flag. If pitch is unmeasurable, reports UNMEASURED (not FAIL). If flank angle segments are insufficient, reports UNMEASURED.
5. **Verbose mode** — `--verbose` flag enables DEBUG logging showing every intermediate value: peak positions, tick spacings, contour area, bbox dimensions, per-row widths etc.
6. **Standalone test** — when run with `--image` pointing to a synthetic bolt (generated internally if file not found), produces reasonable output without crashing.
7. **No magic numbers** — all thresholds are named constants at the top of the file.
8. **Correct flank angle** — must NOT have the 2× doubling error. Expected output for ISO metric thread: ~60°, not ~120° or ~138°.
9. **Correct pitch** — must scan LEFT EDGE STRIP (x to x+8), not center column. Expected output for M8×1.25: ~1.25mm.
10. **Annotated output** — always saves annotated image alongside terminal report.

---

## PART 11 — KNOWN BUGS TO AVOID (from previous implementation)

These specific bugs were identified in an earlier version. Do NOT repeat them:

### Bug 1 — Pitch detection: center column scanning (WRONG)
```python
# WRONG — do not do this:
mid_col_x = x + w // 2
profile = gray[y:y_end, mid_col_x]  # center column = uniform dark, no peaks
```
```python
# CORRECT — edge strip averaging:
strip = gray[y:y+h, x:x+8]
profile = strip.mean(axis=1)  # 8px wide strip at LEFT edge of bolt
```

### Bug 2 — Flank angle: 2× doubling error (WRONG)
```python
# WRONG — do not do this:
avg_half_angle = np.median(all_angles)  # ~69° from one-side analysis
included_angle = 2.0 * avg_half_angle  # = 138° — WRONG

# Also WRONG:
# Calling _fit_flank_angle() separately for left and right sides,
# where each call already returns a "doubled" angle,
# then averaging two doubled angles together.
```
```python
# CORRECT — sum of rising + falling half-angles:
avg_rising  = np.median(rising_half_angles)   # ~30° for ISO
avg_falling = np.median(falling_half_angles)  # ~30° for ISO
flank_angle = avg_rising + avg_falling         # = 60° ✓
```

### Bug 3 — Major diameter: 90th percentile underestimation
```python
# WRONG (underestimates for tight tolerances):
major_px = np.percentile(max_widths, 90)

# CORRECT:
major_px = np.percentile(max_widths, 97)
```

### Bug 4 — Background inversion: using profile mean (unreliable)
```python
# WRONG:
if profile.mean() < 128: profile = 255 - profile

# CORRECT — sample actual background:
bg_sample = gray[y + h//2, max(0, x-20):x-2]
bg_mean = bg_sample.mean() if len(bg_sample) > 5 else 200.0
if bg_mean > 128: profile = 255.0 - profile
```

---

## PART 12 — DELIVERABLE

Write the **complete, production-grade `analyze_thread_image.py`** as a single file.

Requirements:
- Minimum 400 lines
- Every function has a complete docstring
- All 4 known bugs from Part 11 are fixed
- Ruler detection implemented with multi-method + confidence scoring
- Correct flank angle algorithm (rising + falling half-angles summed)
- Correct pitch algorithm (edge strip, not center column)
- Full CLI with argparse
- Annotated output image generation
- Clean formatted terminal report

Start with the imports and constants block, then implement each STEP function in order (1 through 8), then write main().

Do not write placeholder implementations. Every function must be complete and runnable.
