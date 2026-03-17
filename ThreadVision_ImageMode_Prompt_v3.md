# ThreadVision AI — Image Analysis Mode
## Master Prompt v3.0 — Multi-Orientation, Multi-Quality Thread Measurement
### Optimized for Extended Thinking AI (Claude Sonnet 4.6 Thinking / Gemini 2.5 Pro)
### Handles: Professional shadowgrams, real-world noisy photos, vertical AND horizontal bolts

---

## INSTRUCTION TO THE AI MODEL

You are a senior computer vision engineer and precision metrology specialist. You have deep expertise in:
- Sub-pixel edge detection and geometric measurement from 2D images
- Automatic ruler/scale detection in macro photography (any position, any orientation)
- Bolt orientation detection (vertical and horizontal thread axis)
- Noise-robust image preprocessing for real-world photographs
- ISO 68-1 / ASME B1.2 thread metrology standards
- Python 3.10+ with OpenCV 4.x, NumPy, SciPy signal processing

Your task is to write a **single, complete, production-grade Python script** called `analyze_thread_image.py` that works reliably on ALL of the following real image types:

### Reference image types the script MUST handle:

**Type A — Professional shadowgram, vertical bolt, side ruler**
- Clean dark silhouette on bright white background
- Vertical metal ruler on the right side (dark background)
- Ruler shows 0–4mm or 0–8mm range
- High contrast, low noise
- Example: M34 bolt, 1.5mm pitch

**Type B — Professional shadowgram, vertical bolt, fine pitch**
- Similar to Type A but finer thread pitch (~0.5mm visible)
- 30+ thread periods visible
- Ruler inverted (0 at top, increasing downward)

**Type C — Real-world photograph, HORIZONTAL bolt, ruler below**
- THIS IS THE HARDEST AND MOST IMPORTANT CASE
- Bolt lies horizontally (thread axis runs LEFT to RIGHT)
- Background is gray/mottled, NOT clean white
- Image is GRAINY and NOISY
- Ruler is positioned below the bolt, also horizontal
- Ruler has fine subdivision tick marks (0.5mm visible)
- Low overall contrast
- The script must auto-detect horizontal orientation and rotate analysis 90°

**THINKING REQUIREMENT — reason through ALL of these before writing any code:**

1. How do you detect whether the bolt is vertical or horizontal? What image features reliably indicate orientation?
2. For a horizontal bolt: how do all 5 measurement algorithms change? Major diameter is now the HEIGHT of the bolt region, not its width. The edge strip for pitch is now the TOP edge, not the left edge.
3. For Type C (noisy, gray background): Canny thresholds of (50, 150) will produce noise-dominated edge maps. What preprocessing chain produces clean edges on a grainy gray background?
4. The ruler in Type C is below the bolt, not to the side. How does your ruler detection handle this? How do you separate ruler tick marks from thread tooth edges when both are periodic horizontal lines?
5. For Type C, the ruler shows fine 0.5mm graduations AND coarser 1mm/5mm marks. How do you correctly identify which class of tick is which, and how does this affect PIXEL_TO_MM computation?
6. For Type A/B: the ruler background is DARK (the ruler itself is black metal). The tick marks are LIGHTER than the ruler background. How does your tick detection handle inverted contrast vs Type C's ruler?
7. When the ruler is on the right side (Type A/B) vs below (Type C), the ruler's long axis is vertical vs horizontal. How does your Hough line detection correctly identify the ruler baseline in both cases?
8. For fine-pitch threads (Type B, ~0.5mm pitch), PEAK_MIN_DISTANCE_PX must be much smaller. How do you auto-adapt this parameter based on the detected ruler calibration?
9. The flank angle for all ISO metric threads is 60°. How does the sign convention of the contour slope change between vertical and horizontal bolt orientations?
10. What is the complete list of visual annotations needed on the output image to clearly communicate: calibration quality, detected orientation, all 5 measurements, and PASS/FAIL for each?

---

## PART 1 — ORIENTATION DETECTION (NEW — critical for Type C)

### Auto-detect bolt orientation before any measurement

```python
def detect_bolt_orientation(image):
    """
    Determine if the bolt axis is vertical or horizontal.
    
    Algorithm:
    1. Convert to grayscale, apply threshold to isolate dark bolt
    2. Find the largest dark connected region (= bolt silhouette)
    3. Compute the bounding box aspect ratio: w/h
       - If h > w (taller than wide): VERTICAL orientation
       - If w > h (wider than tall): HORIZONTAL orientation
    4. Additionally: compute PCA on the contour points
       - First principal component = bolt axis direction
       - angle = degrees(arctan2(eigvec[1], eigvec[0]))
       - angle near 0° or 180° = HORIZONTAL
       - angle near 90° = VERTICAL
    5. Return: orientation ('vertical' | 'horizontal'), angle_degrees
    
    IMPORTANT:
    For HORIZONTAL bolts, all subsequent algorithms must be
    transposed. The simplest approach: rotate the entire image
    90° clockwise before processing, run all vertical algorithms,
    then rotate annotations back 90° counter-clockwise for output.
    
    Implementation:
        if orientation == 'horizontal':
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated = True
    All subsequent processing uses the (possibly rotated) image.
    All pixel coordinates in the final output are un-rotated before
    drawing annotations.
    """
```

---

## PART 2 — NOISE-ROBUST PREPROCESSING (NEW — critical for Type C)

### Standard preprocessing (Type A/B — clean images)
```python
def preprocess_clean(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 50, 150)
    return edges, enhanced
```

### Noise-robust preprocessing (Type C — grainy real photos)
```python
def preprocess_noisy(image):
    """
    For grainy, low-contrast images (Type C).
    
    The key problem: Gaussian blur at kernel (5,5) is insufficient
    for high-grain images. Standard Canny at (50,150) produces
    noise-dominated edge maps where grain edges outnumber thread edges.
    
    Solution chain:
    1. Convert to grayscale
    2. Non-local means denoising (better than Gaussian for grain):
       denoised = cv2.fastNlMeansDenoising(gray, h=10, 
                      templateWindowSize=7, searchWindowSize=21)
       WHY: fastNlMeans preserves edges while removing grain noise.
            Gaussian blur smears both grain AND thread edges equally.
    3. Bilateral filter pass (edge-preserving smoothing):
       smoothed = cv2.bilateralFilter(denoised, d=9, 
                      sigmaColor=75, sigmaSpace=75)
    4. CLAHE with stronger clip limit for low-contrast images:
       clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
       enhanced = clahe.apply(smoothed)
    5. Canny with AUTO-COMPUTED thresholds (Otsu method):
       _, thresh = cv2.threshold(enhanced, 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
       otsu_val = thresh  # This is Otsu's optimal threshold
       canny_low = 0.5 * otsu_val
       canny_high = otsu_val
       edges = cv2.Canny(enhanced, canny_low, canny_high)
    6. Morphological cleanup:
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
       edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    Returns: edges, enhanced
    """
```

### Auto-select preprocessing mode
```python
def preprocess(image, bolt_bbox=None):
    """
    Automatically select preprocessing mode based on image quality.
    
    Quality assessment:
    1. Compute local noise estimate in a background region:
       - Sample a 50×50 region known to be background (top-left corner)
       - noise_level = np.std(gray_region)
    2. If noise_level > 15: use preprocess_noisy()
       If noise_level <= 15: use preprocess_clean()
    3. Log which mode was selected and the noise estimate
    
    The threshold of 15 (out of 255) separates:
    - Professional backlit images: noise ~ 3–8 (clean white background)
    - Real-world photos on gray: noise ~ 20–40 (grain + ambient variation)
    """
```

---

## PART 3 — RULER DETECTION (UPDATED for all 4 image types)

### The ruler detection challenge across image types

```
Type A/B ruler: RIGHT side of image, vertical orientation
               DARK background (black metal ruler body)
               LIGHTER tick marks on dark background
               Numbers also visible but not needed
               
Type C ruler:   BELOW the bolt, horizontal orientation  
               LIGHTER background (gray paper/surface)
               DARKER tick marks on light background
               Fine 0.5mm + coarser 1mm + coarser 5mm graduations
```

### Ruler detection algorithm — handles both contrast polarities

```python
def detect_ruler_and_calibrate(image, bolt_bbox=None):
    """
    Detect millimetre ruler in image and compute PIXEL_TO_MM.
    Works for rulers on any side, any contrast polarity.
    
    STEP 1 — Locate ruler region:
    
    a) Exclude the bolt region from ruler search:
       If bolt_bbox is known, mask it out before ruler detection.
       The bolt has periodic edges that could confuse ruler detection.
    
    b) Divide image into quadrant strips:
       - Right strip: x > 70% of image width
       - Left strip:  x < 30% of image width  
       - Bottom strip: y > 70% of image height
       - Top strip:    y < 30% of image height
    
    c) For each strip, compute a "ruler score":
       ruler_score = density of parallel, evenly-spaced line segments
       Use HoughLinesP to find line segments in each strip.
       Count segments that are:
         - Short (tick marks are short): length between 5px and 80px
         - Parallel to each other (within 10° angle tolerance)
         - Evenly distributed perpendicular to their direction
       The strip with the highest score = ruler location.
    
    d) Detect ruler contrast polarity:
       Sample the mean brightness of the identified ruler strip.
       If mean < 100: DARK ruler background (Type A/B) → ticks are LIGHT
       If mean >= 100: LIGHT ruler background (Type C) → ticks are DARK
    
    STEP 2 — Extract tick positions:
    
    a) Crop the ruler strip with padding
    b) Apply Canny edge detection (auto-threshold)
    c) Run HoughLinesP perpendicular to ruler baseline direction
       For VERTICAL ruler (Type A/B): look for HORIZONTAL line segments
       For HORIZONTAL ruler (Type C): look for VERTICAL line segments
    d) Cluster detected lines by position:
       Lines within 3px of each other = same tick mark
       Use centroid of cluster as tick position
    e) Sort tick positions along the ruler axis
    
    STEP 3 — Classify tick sizes:
    
    For a ruler with 1mm small ticks + 5mm medium ticks + 10mm large ticks:
    a) Measure the LENGTH of each detected tick segment
    b) Cluster tick lengths into 2–3 groups (KMeans or simple percentile split)
       Shortest group = 1mm ticks (most numerous)
       Medium group = 5mm ticks
       Longest group = 10mm ticks
    c) Use the 1mm tick group for calibration (most data points, best statistics)
    d) Validate: spacing of 5mm ticks should be ~5× spacing of 1mm ticks
       If this ratio is not approximately 5, classification is wrong → retry
    
    STEP 4 — Compute PIXEL_TO_MM:
    
    a) From the 1mm tick positions: diffs = np.diff(sorted_1mm_ticks)
    b) Remove outliers: keep diffs within 1.5×IQR of median
    c) PIXEL_TO_MM = np.median(filtered_diffs) / 1.0
       (Each 1mm tick interval spans PIXEL_TO_MM pixels)
    
    STEP 5 — Compute confidence score:
    
    HIGH: >= 10 valid 1mm ticks, variance < 5%
    MEDIUM: 5–9 ticks, variance < 15%
    LOW: 3–4 ticks, variance < 25% → warn user
    FAILED: < 3 ticks or variance >= 25% → raise CalibrationError
    
    STEP 6 — Physical validation:
    
    PIXEL_TO_MM must be between 20 and 500 px/mm:
    - < 20: image too far, threads not measurable
    - > 500: physically impossible for standard cameras
    If outside range: raise CalibrationError with explanation.
    
    STEP 7 — Auto-adapt PEAK_MIN_DISTANCE_PX based on calibration:
    
    # For pitch detection, the minimum distance between peaks must be
    # at least 80% of the expected minimum pitch at this magnification.
    # Minimum ISO metric pitch is 0.5mm (M3). Maximum we care about = 3.0mm.
    PEAK_MIN_DISTANCE_PX = max(10, int(0.4 * PIXEL_TO_MM))
    # At 120 px/mm: min_distance = 48px (correct for 1.25mm pitch spacing ~150px)
    # At 300 px/mm: min_distance = 120px (fine pitch at high magnification)
    
    Returns:
        PIXEL_TO_MM: float
        ruler_bbox: (x, y, w, h)
        confidence: 'HIGH' | 'MEDIUM' | 'LOW'
        tick_positions: list of pixel positions (for annotation)
        peak_min_dist: int (auto-adapted for pitch detection)
    """
```

---

## PART 4 — BOLT SILHOUETTE DETECTION (UPDATED)

```python
def find_bolt_silhouette(image, ruler_bbox, orientation):
    """
    Find the bolt contour, excluding the ruler region.
    
    STEP 1 — Create search mask:
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    # Zero out ruler region (expanded by 20px padding to be safe):
    rx, ry, rw, rh = ruler_bbox
    mask[max(0,ry-20):ry+rh+20, max(0,rx-20):rx+rw+20] = 0
    
    STEP 2 — Threshold to find dark regions (bolt silhouette):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Otsu threshold — works for both clean and noisy images:
    _, binary = cv2.threshold(gray, 0, 255, 
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, mask)
    
    STEP 3 — Morphological cleanup:
    # Close small gaps in bolt silhouette:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    STEP 4 — Find contours and select bolt:
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise BoltNotFoundError("No bolt silhouette detected.")
    
    # Filter contours: must have reasonable aspect ratio for bolt
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = max(w,h) / max(min(w,h), 1)
        area = cv2.contourArea(c)
        # Bolt should be elongated (ratio > 2) and large enough
        if ratio > 1.5 and area > 0.005 * image.shape[0] * image.shape[1]:
            valid.append(c)
    
    if not valid:
        raise BoltNotFoundError(
            "No elongated object found (expected bolt with aspect ratio > 1.5)."
        )
    
    # Select the largest valid contour = bolt
    bolt_contour = max(valid, key=cv2.contourArea)
    bbox = cv2.boundingRect(bolt_contour)
    
    # Physical size validation:
    x, y, w, h = bbox
    mm_per_px = 1.0 / PIXEL_TO_MM
    thread_dim_mm = w * mm_per_px  # width if vertical, or height if horizontal
    if thread_dim_mm < 1.0 or thread_dim_mm > 50.0:
        raise BoltNotFoundError(
            f"Detected object is {thread_dim_mm:.1f}mm wide — "
            "outside expected bolt range (1–50mm)."
        )
    
    return bolt_contour, bbox
```

---

## PART 5 — MEASUREMENT ALGORITHMS (ORIENTATION-AWARE)

All measurement functions must accept an `orientation` parameter and adjust accordingly.

### Convention:
```
VERTICAL bolt:   measure width  → diameter, scan LEFT EDGE strip → pitch
HORIZONTAL bolt: measure height → diameter, scan TOP EDGE strip → pitch

After rotating horizontal image 90° clockwise:
  The bolt becomes vertical.
  ALL measurement algorithms run identically on the rotated image.
  This means: rotate once → run vertical algorithms → rotate annotations back.
  
The orientation rotation approach is STRONGLY PREFERRED over
writing two versions of every algorithm.
```

### Dimension 1: Major Diameter
```python
def measure_major_diameter(contour, bbox, mm_per_px, warnings):
    """
    Major diameter = widest extent perpendicular to bolt axis.
    For vertical bolt: this is the horizontal extent (width).
    After rotation normalization, always computes WIDTH.
    
    Algorithm (same as before, but using 97th percentile):
    1. Get contour points in middle 60% of bolt length
    2. Per-row: max_x - min_x = row width
    3. major_px = np.percentile(row_widths, 97)  ← 97th, not 90th
    4. major_mm = major_px * mm_per_px
    
    Validation: 1mm < major_mm < 50mm
    """
```

### Dimension 2: Pitch — CORRECTED EDGE STRIP METHOD
```python
def measure_pitch(gray, bbox, mm_per_px, peak_min_dist, warnings):
    """
    Pitch = periodic spacing of thread crests.
    After rotation normalization: always scans the LEFT EDGE strip.
    
    MANDATORY IMPLEMENTATION (do not deviate):
    
    x, y, w, h = bbox
    
    # 8-pixel-wide strip at the LEFT edge of the bolt
    x_strip_end = min(x + 8, gray.shape[1])
    strip = gray[y:y+h, x:x_strip_end]           # shape: (h, 8)
    profile = strip.mean(axis=1)                   # shape: (h,)
    
    # Determine background brightness from left of bolt
    bg_x_start = max(0, x - 30)
    bg_x_end   = max(0, x - 2)
    if bg_x_end > bg_x_start:
        bg_sample = gray[y + h//2, bg_x_start:bg_x_end]
        bg_mean = float(bg_sample.mean())
    else:
        bg_mean = 200.0  # assume light background
    
    # Invert for dark bolt on light background
    if bg_mean > 128:
        profile = 255.0 - profile
    
    # Smooth (size=5 for clean images, size=9 for noisy)
    from scipy.ndimage import uniform_filter1d
    smooth_size = 9 if noise_level > 15 else 5
    profile_smooth = uniform_filter1d(profile, size=smooth_size)
    
    # Peak detection with auto-adapted distance
    p_range = profile_smooth.ptp()
    height_threshold = profile_smooth.min() + PEAK_MIN_HEIGHT_RATIO * p_range
    
    peaks, properties = find_peaks(
        profile_smooth,
        height=height_threshold,
        distance=peak_min_dist,          # auto-adapted from calibration
        prominence=p_range * 0.08,       # 8% prominence (relaxed from 10%)
    )
    
    if len(peaks) < 3:
        warnings.append(
            f"Only {len(peaks)} peaks found in edge strip "
            f"(need ≥3). Image: check focus and bolt alignment."
        )
        return None, peaks, profile_smooth
    
    # Inter-peak distances — IQR outlier removal (more robust for small N)
    diffs = np.diff(peaks).astype(float)
    q1, q3 = np.percentile(diffs, [25, 75])
    iqr = q3 - q1
    mask = (diffs >= q1 - 1.5*iqr) & (diffs <= q3 + 1.5*iqr)
    filtered = diffs[mask] if mask.sum() > 0 else diffs
    
    pitch_px = np.median(filtered)
    pitch_mm = pitch_px * mm_per_px
    
    # Thread damage warning
    cv = np.std(filtered) / np.median(filtered) if np.median(filtered) > 0 else 0
    if cv > 0.2:
        warnings.append(
            f"Pitch coefficient of variation = {cv*100:.1f}% "
            "(>20%). Possible thread damage or irregular spacing."
        )
    
    return pitch_mm, peaks, profile_smooth
```

### Dimension 3: Minor Diameter
```python
def measure_minor_diameter(edges, bbox, mm_per_px, warnings):
    """
    Minor diameter = thread root width (minimum functional width).
    Uses 10th percentile of per-row widths in middle 60% of bolt.
    
    Additional robustness for noisy images:
    Before scanning rows, apply morphological dilation to edge map
    to connect fragmented edge pixels caused by noise:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    Then scan edges_dilated instead of edges.
    This prevents gaps in thread root edges from giving wrong widths.
    """
```

### Dimension 4: Thread Depth (derived, unchanged)
```python
thread_depth_mm = (major_diameter_mm - minor_diameter_mm) / 2.0
```

### Dimension 5: Flank Angle — CORRECTED ALGORITHM
```python
def measure_flank_angle(edges, bbox, mm_per_px, warnings):
    """
    Flank angle = included angle between rising and falling thread flanks.
    Expected: ~60° for ISO metric threads.
    
    CORRECTED ALGORITHM — NO 2× DOUBLING ERROR:
    
    STEP 1 — Extract left-edge contour points in middle 40% of bolt:
    x, y, w, h = bbox
    h_start = y + int(h * 0.30)
    h_end   = y + int(h * 0.70)
    
    # Get all contour points in the middle-height band
    # Use left half of bolt (x to x + w//2) only
    # Left edge traces the complete tooth profile (both flanks)
    
    STEP 2 — Identify rising and falling segments:
    Sort all left-edge points by Y coordinate.
    Compute local slope: for each point i, slope = (x[i+2] - x[i]) / (y[i+2] - y[i])
    Sign of slope indicates flank direction:
      Negative slope (x decreasing as y increases) = rising flank (into the crest)
      Positive slope (x increasing as y increases) = falling flank (away from crest)
    
    Group consecutive same-sign slope points into segments.
    Require minimum 5 points per segment to be valid.
    
    STEP 3 — Fit line to each segment:
    For each segment, fit: x = m*y + b using np.polyfit(y_pts, x_pts, 1)
    slope m = dx/dy
    half_angle = degrees(arctan(abs(m)))
    
    STEP 4 — Compute included angle (CORRECT method):
    rising_half_angles  = [half_angle for rising segments]
    falling_half_angles = [half_angle for falling segments]
    
    avg_rising  = np.median(rising_half_angles)   # typically ~30° for ISO
    avg_falling = np.median(falling_half_angles)  # typically ~30° for ISO
    
    # CORRECT: sum of two half-angles = included angle
    flank_angle = avg_rising + avg_falling         # should give ~60° for ISO
    
    # NOT: flank_angle = 2 * avg_one_side  ← this was the previous bug
    # NOT: flank_angle = 2 * avg_half_angle ← this gave 138° instead of 60°
    
    STEP 5 — Minimum segment requirements:
    Require >= 2 rising segments AND >= 2 falling segments.
    If not met: return None, UNMEASURED.
    
    STEP 6 — Physical validation:
    if flank_angle < 45 or flank_angle > 75:
        warnings.append(f"Flank angle {flank_angle:.1f}° outside 45–75°. SUSPECT.")
    
    Returns: (flank_angle_degrees, rising_angles, falling_angles)
    """
```

---

## PART 6 — ISO TOLERANCE CONSTANTS

```python
# Supported thread standards — extend as needed
THREAD_STANDARDS = {
    'M3x0.5':  {'major': (2.874, 3.000), 'minor': (2.387, 2.459),
                'pitch': (0.490, 0.510), 'depth': (0.307, 0.307),
                'angle': (59.0, 61.0)},
    'M4x0.7':  {'major': (3.838, 4.000), 'minor': (3.141, 3.242),
                'pitch': (0.686, 0.714), 'depth': (0.429, 0.429),
                'angle': (59.0, 61.0)},
    'M6x1.0':  {'major': (5.794, 6.000), 'minor': (4.773, 4.917),
                'pitch': (0.980, 1.020), 'depth': (0.613, 0.613),
                'angle': (59.0, 61.0)},
    'M8x1.25': {'major': (7.972, 8.000), 'minor': (6.466, 6.647),
                'pitch': (1.225, 1.275), 'depth': (0.613, 0.677),
                'angle': (59.0, 61.0)},
    'M10x1.5': {'major': (9.968, 10.000), 'minor': (8.160, 8.376),
                'pitch': (1.470, 1.530), 'depth': (0.920, 0.920),
                'angle': (59.0, 61.0)},
    'M12x1.75':{'major': (11.966, 12.000), 'minor': (9.853, 10.106),
                'pitch': (1.715, 1.785), 'depth': (1.074, 1.074),
                'angle': (59.0, 61.0)},
    'M34x1.5': {'major': (33.962, 34.000), 'minor': (32.376, 32.752),
                'pitch': (1.470, 1.530), 'depth': (0.920, 0.920),
                'angle': (59.0, 61.0)},
}
DEFAULT_STANDARD = 'M8x1.25'
```

---

## PART 7 — ANNOTATED OUTPUT IMAGE (UPDATED)

```python
def generate_annotated_image(original, results, calibration, tolerance_results, 
                              orientation, rotated_image=None):
    """
    Draw comprehensive measurement annotations on the original image.
    All annotations use coordinates in the ORIGINAL image space
    (un-rotate if the image was rotated for processing).
    
    Scale all text and line thickness to image resolution:
        scale = max(image.shape) / 1400.0
        font_scale = max(0.4, 0.6 * scale)
        thickness = max(1, int(2 * scale))
    
    ANNOTATIONS TO DRAW:
    
    1. CALIBRATION OVERLAY:
       - Semi-transparent rectangle over detected ruler region (blue tint)
       - Label: "Cal: X.X px/mm (Xµm/px) [HIGH/MED/LOW]"
       - Mark detected tick positions as small cyan dots on ruler
    
    2. BOLT BOUNDING BOX:
       - Dashed rectangle around bolt
       - Color: green=PASS, red=FAIL, yellow=PARTIAL/UNMEASURED
       - Label in corner: orientation detected ("VERTICAL"/"HORIZONTAL")
    
    3. MAJOR DIAMETER:
       - Double-headed horizontal arrow at y = bolt_center_y
       - Arrow spans from left to right edge of bolt
       - Label above arrow: "Major Ø: X.XXX mm [✓/✗]"
       - Color: green if PASS, red if FAIL
    
    4. MINOR DIAMETER:
       - Double-headed horizontal arrow at y = deepest root row
       - Shorter than major diameter arrow
       - Label below arrow: "Minor Ø: X.XXX mm [✓/✗]"
    
    5. PITCH BRACKET:
       - Vertical bracket on the LEFT side of bolt spanning exactly 1 pitch
       - Small horizontal bars at top and bottom of bracket
       - Label left of bracket: "P: X.XXX mm [✓/✗]"
       - Mark detected peaks as small cyan circles on the edge strip
    
    6. THREAD DEPTH LABEL:
       - Small arrow from major edge to minor edge (radial direction)
       - Label: "Depth: X.XXX mm [✓/✗]"
    
    7. FLANK ANGLE ARC:
       - Small arc at one thread tooth tip showing the included angle
       - Label: "α: XX.X° [✓/✗]"
    
    8. RESULT BANNER (top of image, semi-transparent):
       - Large text: "██ PASS ██" (green) or "██ FAIL ██" (red)
       - Subtitle: "ISO M8×1.25 | Cal: X.X px/mm | X dims measured"
    
    9. EDGE STRIP INDICATOR:
       - Thin vertical rectangle on left bolt edge showing where
         the pitch profile was scanned (x to x+8 pixels)
       - Color: magenta, semi-transparent
       - Label: "Edge strip"
    
    Save as: {input_filename}_analyzed.jpg (JPEG quality=95)
    """
```

---

## PART 8 — CLI INTERFACE

```bash
# Basic usage:
python analyze_thread_image.py --image bolt_photo.jpg

# With all options:
python analyze_thread_image.py \
    --image bolt_photo.jpg \
    --standard M8x1.25 \
    --show \
    --output result.jpg \
    --verbose \
    --save-profile \
    --calibration 127.3

# Arguments:
# --image PATH        Required. Input image path (JPEG/PNG/BMP/TIFF)
# --standard STR      Thread standard. Default: auto-detect from pitch.
#                     Options: M3x0.5, M4x0.7, M6x1.0, M8x1.25, M10x1.5,
#                              M12x1.75, M34x1.5
# --show              Display result image in OpenCV window
# --output PATH       Custom output path for annotated image
# --verbose           DEBUG logging — shows all intermediate values
# --save-profile      Save the pitch intensity profile as a separate plot
# --calibration F     Manual PIXEL_TO_MM value — skips ruler detection
# --no-ruler          Skip ruler detection, prompt for manual input
# --force-orientation [vertical|horizontal]  Override auto-detection
```

---

## PART 9 — TERMINAL REPORT FORMAT

```
============================================================
  ThreadVision AI — Thread Measurement Report v3.0
  Image:       bolt_photo.jpg (1412 × 784 px)
  Orientation: HORIZONTAL (auto-detected, rotated 90° for processing)
  Calibration: 134.2 px/mm (7.45 µm/px) — HIGH confidence (16 ticks)
  Standard:    M8×1.25 (ISO 262)
  Noise level: 28.4 (NOISY — used noise-robust preprocessing)
============================================================

  Dimension         Measured    Nominal    Tolerance        Status
  ─────────────────────────────────────────────────────────────────
  Major Diameter    7.994 mm    8.000 mm   [7.972–8.000]    ✓ PASS
  Minor Diameter    6.521 mm    6.647 mm   [6.466–6.647]    ✓ PASS
  Pitch             1.247 mm    1.250 mm   [1.225–1.275]    ✓ PASS
  Thread Depth      0.737 mm    0.677 mm   [0.613–0.677]    ✗ FAIL
  Flank Angle       59.8°       60.0°      [59.0°–61.0°]    ✓ PASS

  ─────────────────────────────────────────────────────────────────
  OVERALL RESULT:  ██  FAIL  ██  (Thread Depth out of tolerance)
  ─────────────────────────────────────────────────────────────────

  Warnings:
    ⚠ Thread Depth derived (not directly measured)
    ⚠ Pitch detection used noisy preprocessing — verify with clean image

  Annotated image saved: bolt_photo_analyzed.jpg
============================================================
```

---

## PART 10 — ERROR HANDLING

```python
class ThreadVisionError(Exception): pass
class ImageLoadError(ThreadVisionError): pass

class CalibrationError(ThreadVisionError):
    """Raised when ruler cannot be detected or calibration is invalid."""
    # Message must include actionable advice:
    # "Ruler not detected. Options:
    #  1. Ensure ruler has ≥5mm of visible tick marks
    #  2. Use --calibration 120.0 to specify manually
    #  3. Use --no-ruler for interactive prompt"

class BoltNotFoundError(ThreadVisionError):
    """Raised when no bolt silhouette can be found."""
    # "No bolt detected. Ensure:
    #  1. Bolt is photographed against contrasting background
    #  2. Bolt occupies at least 5% of image area
    #  3. Use --force-orientation to override auto-detection"

class MeasurementError(ThreadVisionError):
    """Raised for unrecoverable measurement failures."""
    # Individual dimension failures return None (UNMEASURED),
    # not exceptions. Only raise if entire image is unmeasurable.
```

---

## PART 11 — KNOWN BUGS — DO NOT REPEAT

### Bug 1 — Pitch: center column scanning ✗
```python
# WRONG — center column is uniform dark, no periodic signal:
profile = gray[y:y+h, x + w//2]

# CORRECT — 8px LEFT EDGE STRIP averaged horizontally:
strip = gray[y:y+h, x:x+8]
profile = strip.mean(axis=1)
```

### Bug 2 — Flank angle: 2× doubling ✗
```python
# WRONG — one-sided computation doubled:
included_angle = 2.0 * avg_half_angle  # gives ~138° instead of ~60°

# CORRECT — sum of two independently measured half-angles:
flank_angle = avg_rising + avg_falling  # ~30° + ~30° = ~60° ✓
```

### Bug 3 — Major diameter: 90th percentile ✗
```python
# WRONG: major_px = np.percentile(row_widths, 90)
# CORRECT: major_px = np.percentile(row_widths, 97)
```

### Bug 4 — Background inversion: profile mean ✗
```python
# WRONG: if profile.mean() < 128: profile = 255 - profile
# CORRECT: sample actual background region to the LEFT of bolt
bg_sample = gray[y + h//2, max(0, x-30):max(0, x-2)]
bg_mean = bg_sample.mean() if len(bg_sample) > 5 else 200.0
if bg_mean > 128: profile = 255.0 - profile
```

### Bug 5 — Noisy images: standard Canny thresholds (50,150) ✗
```python
# WRONG for grainy Type C images: cv2.Canny(enhanced, 50, 150)
# Produces noise-dominated edge map.
# CORRECT: auto-compute thresholds using Otsu + fastNlMeansDenoising first
```

### Bug 6 — Ruler detection: ignoring contrast polarity ✗
```python
# WRONG: assume all rulers have dark ticks on light background
# CORRECT: detect ruler background brightness and adapt accordingly.
# Type A/B: dark ruler background → ticks are LIGHTER than background
# Type C: light surface → ticks are DARKER than background
```

---

## PART 12 — DELIVERABLE REQUIREMENTS

Write the **complete, production-grade `analyze_thread_image.py`** as a single file.

Mandatory requirements:
- Handles all 4 image types described in the INSTRUCTION section
- Auto-detects bolt orientation (vertical/horizontal)
- Auto-adapts preprocessing mode based on image noise level
- Ruler detection works for any ruler position (left/right/top/bottom)
- Ruler detection handles both dark-on-light and light-on-dark tick marks
- Correct pitch algorithm: edge strip, not center column
- Correct flank angle: rising + falling half-angle sum, not 2× one side
- 97th percentile for major diameter (not 90th)
- All 6 known bugs from Part 11 are explicitly avoided
- Auto-adapts PEAK_MIN_DISTANCE_PX from detected PIXEL_TO_MM
- Multiple thread standards supported (M3 through M34)
- Full CLI via argparse
- Annotated output image with all 9 annotation types
- Clean formatted terminal report
- Zero silent exception swallowing — all errors visible to user

Start with: imports → constants → orientation detection → preprocessing → 
ruler detection → bolt detection → measurement functions → 
tolerance check → annotation → report → main()

Every function must be complete and runnable. No placeholder implementations.
Minimum 500 lines.
