# ThreadVision AI — Master Project Prompt
### Optimized for Extended Thinking AI (Claude Sonnet 4.6 Thinking / Gemini 2.5 Pro)
### Version 1.0 | SSIP / Hackathon 2026 | Deadline: April 6, 2026

---

## INSTRUCTION TO THE AI MODEL

You are a senior embedded systems engineer and computer vision specialist. You have deep expertise in:
- Raspberry Pi hardware (GPIO, CSI camera, DSI display)
- Python 3.10+ and OpenCV 4.x image processing pipelines
- Sub-pixel edge detection and geometric measurement from silhouette images
- ISO 68-1 / ASME B1.2 thread metrology standards
- CustomTkinter GUI development for touchscreen industrial interfaces
- SQLite3 database design for quality control logging
- systemd service configuration for embedded Linux auto-start

Your task is to help build **ThreadVision AI** — a fully working, standalone, automated thread inspection system running on Raspberry Pi 4. You will reason carefully before generating any code, ensuring every function, variable, and module is production-grade, well-commented, and handles real hardware edge cases.

**THINKING REQUIREMENT**: Before writing any code, think through:
1. What could go wrong with this approach on real hardware?
2. What are the edge cases in the image data?
3. What is the mathematically correct way to compute each measurement?
4. How does each module interact with the others?
5. What will fail first on demo day and how do we prevent it?

---

## PART 1 — PROJECT OVERVIEW

**Project Name:** ThreadVision AI
**Full Name:** AI-Powered Non-Contact Thread Inspection System
**Domain:** Manufacturing Technology / Machine Vision / Embedded AI
**Deadline:** April 6, 2026 (SSIP Hackathon, Chennai)
**Budget:** Under ₹8,000 total BOM
**Target Part:** M8 × 1.25 bolt (ISO 262 standard metric thread)

### What the system does
A Raspberry Pi 4 captures a backlit silhouette image of a threaded bolt, runs OpenCV computer vision algorithms to extract thread geometry, measures 5 critical dimensions in millimetres, compares them to ISO tolerances, and outputs PASS or FAIL — all in under 2 seconds, with no human measurement involved.

### Why this is hard
1. Thread features are tiny (pitch = 1.25mm). Small errors in pixel location = large measurement errors.
2. The pixel-to-mm calibration ratio is a single float that ALL measurements depend on. Wrong calibration = all 5 measurements are systematically wrong.
3. The pitch measurement requires detecting periodic peaks (thread crests) in noisy pixel data — this is a signal processing problem, not just image processing.
4. The flank angle calculation requires fitting lines to noisy edge points — requires robust least-squares regression, not simple slope calculation.
5. The system must run unattended on a Pi with no keyboard — code must be defensive against hardware errors, camera failures, and bad images.

---

## PART 2 — HARDWARE SPECIFICATION

### Chosen Hardware Stack (Final Decision)

```
PHYSICAL LAYER
├── Raspberry Pi 4 Model B (4GB RAM)     ← main processor
├── Pi Camera Module v2 (8MP, CSI)       ← image capture
├── 7" Official DSI Touchscreen          ← operator interface  
├── LED Backlight Panel (150×150mm)      ← controlled via GPIO
├── V-Block Fixture (90° steel)          ← holds bolt precisely
├── 5V/3A USB-C Power Adapter            ← powers everything
├── 32GB MicroSD Card (Class 10/U3)      ← OS + code + logs
└── Green + Red LED (5mm, 3.3V)         ← physical PASS/FAIL signal

GPIO PIN ASSIGNMENTS
├── GPIO 17 → LED Backlight relay (OUTPUT)
├── GPIO 27 → Green LED / PASS (OUTPUT)
├── GPIO 22 → Red LED / FAIL (OUTPUT)
└── GPIO 18 → Trigger button (INPUT, pull-up)
```

### Camera Details
- **Model:** Raspberry Pi Camera Module v2
- **Sensor:** Sony IMX219, 8 megapixels
- **Resolution used for inspection:** 3280 × 2464 (max, for best accuracy)
- **Connection:** CSI ribbon cable (not USB)
- **Python library:** `picamera2` (official RPi Foundation library)
- **Frame format returned:** NumPy ndarray, BGR color

### Critical Hardware Constraint
The Pi Camera v2 with a standard M12 macro lens (1:1 magnification, ~50mm working distance) placed at the correct distance from the bolt produces approximately **120 pixels per millimetre** at the sensor. However, **this value MUST be measured empirically** using the calibration procedure — do not hardcode 120 as a constant. Store it as `PIXEL_TO_MM` in `config.py` and load it at runtime.

---

## PART 3 — SOFTWARE ARCHITECTURE

### File Structure (complete)
```
/home/pi/threadvision/
├── main.py                  ← entry point, called by systemd
├── config.py                ← all constants, tolerances, GPIO pins
├── capture.py               ← picamera2 wrapper
├── preprocessor.py          ← OpenCV edge detection pipeline  
├── measurement.py           ← 5 dimension extraction (the hard module)
├── inspector.py             ← ISO tolerance check + GPIO output
├── logger.py                ← SQLite3 + CSV export + SPC stats
├── gui.py                   ← CustomTkinter touchscreen interface
├── calibrate.py             ← standalone pixel-to-mm calibration tool
├── requirements.txt         ← pip dependencies
├── threadvision.service     ← systemd auto-start unit file
└── logs/
    └── inspection_log.db    ← SQLite3 database (auto-created)
```

### Technology Stack (exact versions)
```
Python          3.10+
picamera2       0.3.x          (camera interface)
OpenCV          4.8.x          (cv2 — image processing)
NumPy           1.24.x         (array math)
SciPy           1.11.x         (scipy.signal.find_peaks — pitch detection)
CustomTkinter   5.2.x          (modern Tkinter GUI)
Pillow (PIL)    10.x           (image format conversion for Tkinter)
pandas          2.0.x          (SPC stats + CSV export)
RPi.GPIO        0.7.x          (GPIO control for LEDs)
SQLite3         built-in       (inspection log database)
```

### Module Interaction Diagram
```
main.py
  │
  ├─► gui.py (creates window, owns the main loop)
  │     │
  │     ├─► capture.py         → returns np.ndarray (raw frame)
  │     ├─► preprocessor.py    → returns (edges, gray)
  │     ├─► measurement.py     → returns dict of 5 measurements
  │     ├─► inspector.py       → returns (overall, per_dim_results)
  │     └─► logger.py          → saves to DB, exports CSV on demand
  │
  └─► config.py (imported by ALL modules — single source of truth)
```

---

## PART 4 — CONFIG.PY SPECIFICATION

`config.py` is the single source of truth for all constants. Every other module imports from it. Here is exactly what it must contain:

```python
# config.py — ThreadVision AI Configuration
# Edit PIXEL_TO_MM after running calibrate.py

# === CALIBRATION ===
PIXEL_TO_MM = 120.0   # pixels per mm — SET THIS WITH calibrate.py BEFORE USE
                       # Approximate starting value only. Must be measured.

# === THREAD STANDARD: M8 × 1.25 (ISO 262) ===
THREAD_STANDARD = "M8x1.25"
NOMINAL_VALUES = {
    'major_diameter': 8.000,   # mm
    'minor_diameter': 6.647,   # mm  
    'pitch':          1.250,   # mm
    'thread_depth':   0.677,   # mm
    'flank_angle':    60.0,    # degrees
}
TOLERANCES = {
    # Key: (lower_bound_mm, upper_bound_mm)
    'major_diameter': (7.972, 8.000),
    'minor_diameter': (6.466, 6.647),
    'pitch':          (1.225, 1.275),
    'thread_depth':   (0.613, 0.677),
    'flank_angle':    (59.0,  61.0),   # degrees, not mm
}

# === GPIO PIN NUMBERS (BCM numbering) ===
PIN_BACKLIGHT = 17    # GPIO 17 → LED backlight relay
PIN_LED_PASS  = 27    # GPIO 27 → Green LED
PIN_LED_FAIL  = 22    # GPIO 22 → Red LED
PIN_TRIGGER   = 18    # GPIO 18 → Manual trigger button (active LOW)

# === CAMERA SETTINGS ===
CAPTURE_WIDTH   = 3280
CAPTURE_HEIGHT  = 2464
PREVIEW_WIDTH   = 640
PREVIEW_HEIGHT  = 480

# === IMAGE PROCESSING THRESHOLDS ===
CANNY_LOW       = 50
CANNY_HIGH      = 150
BLUR_KERNEL     = (5, 5)
CLAHE_CLIP      = 2.0
CLAHE_GRID      = (8, 8)

# === PEAK DETECTION PARAMETERS (for pitch measurement) ===
PEAK_MIN_HEIGHT_RATIO = 0.4   # fraction of max intensity
PEAK_MIN_DISTANCE_PX  = 50    # minimum pixels between crests

# === DATABASE ===
DB_PATH = "/home/pi/threadvision/logs/inspection_log.db"
CSV_PATH = "/home/pi/threadvision/logs/export.csv"

# === TIMING ===
LED_ON_DURATION_MS = 1500     # how long to hold LEDs after result
```

---

## PART 5 — PIXEL-TO-MM CALIBRATION (HIGHEST PRIORITY)

**This is the most critical step in the entire project. Build this first.**

### Why it is critical
Every measurement (major diameter, minor diameter, pitch, thread depth, flank angle) is computed by:
1. Finding a distance in pixels
2. Multiplying by `(1 / PIXEL_TO_MM)` to get millimetres

If `PIXEL_TO_MM` is wrong by even 5%, all 5 measurements will be 5% off. For M8 bolt major diameter (nominal 8.000mm), a 5% error = 0.4mm error — which is far outside the ISO tolerance of ±0.028mm. The system would give wrong PASS/FAIL for every single part.

### How calibration works
1. Place a reference object of **known physical width** in exactly the same position as the bolt in the V-block fixture.
   - Best reference: a precision gauge block (e.g. 10mm width, traceable to NIST/NPL)
   - Acceptable for hackathon: a standard M8 bolt of known shank diameter measured with a micrometer
2. Capture an image of the reference object under the same backlight conditions as inspection.
3. Identify the reference object's pixel width in the image.
4. Compute: `PIXEL_TO_MM = pixel_width_of_reference / known_mm_width_of_reference`
5. Save this value to `config.py`.

### calibrate.py — What it must do
- Capture a still frame using picamera2
- Display the image on screen with OpenCV window
- Allow user to click two points (left edge and right edge of reference object)
- Calculate the pixel distance between the two points (Euclidean distance)
- Prompt user to enter the known physical width in mm
- Calculate `PIXEL_TO_MM = pixel_distance / known_mm`
- Display the result and ask for confirmation
- Write the value directly into `config.py` using file I/O
- Print a validation summary: "At this ratio, 1 pixel = X µm. M8 bolt major diameter (8mm) would span Y pixels."

### calibrate.py — Complete implementation requirements

```python
# The script must:
# 1. Use picamera2 to capture at full resolution (3280x2464)
# 2. Scale image down to 1280x960 for display (preserve aspect ratio)
# 3. Track scale_factor = 1280/3280 so clicks on display image are
#    converted back to full-resolution coordinates before calculating
# 4. Show crosshairs at click points
# 5. Show live pixel distance as user moves mouse after first click
# 6. On second click, compute: 
#      full_res_px = euclidean(pt1, pt2) / scale_factor
#      PIXEL_TO_MM = full_res_px / user_entered_mm
# 7. Rewrite ONLY the PIXEL_TO_MM line in config.py, preserve all else
# 8. Log calibration event to DB: timestamp, reference_mm, pixel_count, ratio
```

### Mathematical validation after calibration
After computing `PIXEL_TO_MM`, validate it is physically reasonable:
- Should be between 80 and 200 px/mm for a Pi Cam v2 at typical macro distances
- If outside this range, warn the user: "Value seems unusual — re-check camera distance"
- Convert to microns/pixel for display: `1e3 / PIXEL_TO_MM` → should be 5–12 µm/pixel

---

## PART 6 — MEASUREMENT MODULE SPECIFICATION

### measurement.py — The 5 dimensions

**Input:** `edges` (np.ndarray, binary edge map), `gray` (np.ndarray, grayscale image)
**Output:** `dict` with keys: `major_diameter`, `minor_diameter`, `pitch`, `thread_depth`, `flank_angle`
**Units:** millimetres for all linear measurements, degrees for angle

#### Dimension 1: Major Diameter
```
Algorithm:
1. Find all contours in edges using cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
2. Select largest contour by area (this is the bolt outline)
3. Get bounding rect: x, y, w, h = cv2.boundingRect(bolt_contour)
4. major_diameter_mm = w * (1.0 / PIXEL_TO_MM)

Edge case: If no contours found, or contour area < threshold, raise ValueError("No bolt detected in frame")
Edge case: If bounding box width is unreasonably large (> 20mm) or small (< 4mm), raise ValueError("Bolt position error")
```

#### Dimension 2: Pitch
```
Algorithm:
1. Extract the vertical intensity profile from the middle column of the bolt region:
   mid_col_x = x + w // 2
   profile = gray[y:y+h, mid_col_x]  # 1D array of pixel intensities
2. Invert if needed: if bolt is dark on light background, profile = 255 - profile
3. Use scipy.signal.find_peaks(profile, height=threshold, distance=min_peak_dist)
   where threshold = PEAK_MIN_HEIGHT_RATIO * profile.max()
   and min_peak_dist = PEAK_MIN_DISTANCE_PX
4. Extract peak positions (indices in the array)
5. Compute inter-peak distances: diffs = np.diff(peaks)
6. Remove outliers: keep only diffs within 1 std dev of median
7. pitch_px = np.median(filtered_diffs)  # median is more robust than mean
8. pitch_mm = pitch_px * (1.0 / PIXEL_TO_MM)

Edge case: If fewer than 3 peaks found, return None and flag as UNMEASURED (not FAIL)
Edge case: If inter-peak variance is > 20%, flag as "THREAD DAMAGED"
```

#### Dimension 3: Minor Diameter
```
Algorithm:
1. For each row in the bolt bounding box (row = y to y+h):
   a. Extract the edge pixels in that row: row_edges = edges[row, x:x+w]
   b. Find nonzero (edge) pixel positions: positions = np.nonzero(row_edges)[0]
   c. If at least 2 edge pixels found, width_px = positions[-1] - positions[0]
   d. Store width_px in a list
2. row_widths = sorted list of all widths
3. Remove top and bottom 10% (outliers from bolt ends)
4. minor_diameter_px = np.percentile(row_widths, 10)  # 10th percentile = thread root
5. minor_diameter_mm = minor_diameter_px * (1.0 / PIXEL_TO_MM)

Note: The minor diameter is the MINIMUM width across the threaded region.
Using percentile(10) instead of min() prevents single-pixel noise from dominating.
```

#### Dimension 4: Thread Depth
```
Algorithm (derived, not directly measured):
thread_depth_mm = (major_diameter_mm - minor_diameter_mm) / 2.0

This is geometrically correct: depth = half the difference of diameters.
No additional computation needed beyond dimensions 1 and 3.
```

#### Dimension 5: Flank Angle
```
Algorithm:
1. Find contour points on the RISING flanks of threads (left-side edges of each tooth)
2. Isolate a 3-period window in the middle of the thread region (avoids end effects)
3. For each rising flank: collect (x, y) pixel coordinates of the edge points
4. Fit a line to these points using np.linalg.lstsq (least-squares regression)
5. Compute angle of fitted line from vertical: angle = degrees(arctan(slope))
6. Repeat for falling flanks (right-side edges)
7. flank_angle = (rising_angle + falling_angle) / 2.0

Expected result: ~60° for ISO metric threads
Edge case: If less than 5 points per flank, flag as UNMEASURED
Edge case: If computed angle is outside 45°–75°, flag as "SUSPECT — VERIFY"
```

---

## PART 7 — GUI SPECIFICATION

### gui.py must implement exactly this layout

```
┌─────────────────────────────────────────────────────┐
│  ThreadVision AI                    [⚙ Settings]    │
├──────────────────────┬──────────────────────────────┤
│                      │  Dimension      Value  Status │
│   LIVE CAMERA FEED   │  Major Dia.   8.001mm  ✓ PASS│
│   (640×480 preview)  │  Minor Dia.   6.512mm  ✓ PASS│
│                      │  Pitch        1.249mm  ✓ PASS│
│                      │  Thread Depth 0.644mm  ✓ PASS│
│                      │  Flank Angle  60.2°    ✓ PASS│
├──────────────────────┴──────────────────────────────┤
│                                                      │
│              ██ PASS ██   (or  ██ FAIL ██)          │
│                                                      │
│   [  INSPECT  ]    [  EXPORT CSV  ]   [  HISTORY ]  │
│                                                      │
│   Parts today: 47 | Pass: 44 | Fail: 3 | Rate: 93.6%│
└─────────────────────────────────────────────────────┘
```

### GUI technical requirements
- Framework: `customtkinter` (NOT standard Tkinter — must use CTk widgets)
- The camera preview must run in a **separate thread** so the GUI never freezes
- The INSPECT button triggers the full pipeline: capture → preprocess → measure → inspect → log
- PASS result: result label text is "PASS", background color is green (#1DB954 or similar)
- FAIL result: result label text is "FAIL", background color is red (#E22134 or similar), failing rows in table are highlighted red
- The stats bar at bottom updates after every inspection
- Touch-optimized: all buttons minimum 60px tall for finger press

---

## PART 8 — STANDALONE BOOT CONFIGURATION

### systemd service file: `/etc/systemd/system/threadvision.service`
```ini
[Unit]
Description=ThreadVision AI Inspection System
After=graphical.target
Wants=graphical.target

[Service]
User=pi
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority
WorkingDirectory=/home/pi/threadvision
ExecStart=/usr/bin/python3 /home/pi/threadvision/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
```

Enable with:
```bash
sudo systemctl daemon-reload
sudo systemctl enable threadvision.service
```

---

## PART 9 — WHAT I WANT YOU TO BUILD (TASK LIST)

**IMMEDIATE PRIORITY — Build in this exact order:**

### Task 1 (DO FIRST): `calibrate.py`
Write the complete, runnable calibrate.py script. It must:
- Use picamera2 to capture a still frame
- Display in an OpenCV window at 1280×960 (scaled from full res)
- Handle mouse clicks to select two calibration points
- Show live feedback: crosshairs, pixel distance, running estimate in µm/pixel
- Ask user for physical dimension in mm (typed input)
- Validate the resulting PIXEL_TO_MM is within physical bounds (80–200 px/mm)
- Rewrite the PIXEL_TO_MM line in config.py with the new value
- Print a clear confirmation and validation summary

### Task 2: `config.py`
Write the complete config.py with all constants exactly as specified in Part 4.

### Task 3: `capture.py`
Write capture_frame() using picamera2. Must handle: camera not found, timeout, bad frame.

### Task 4: `preprocessor.py`
Write preprocess(frame) — full pipeline: grayscale → blur → CLAHE → Canny → return (edges, gray)

### Task 5: `measurement.py`
Write measure(edges, gray) — all 5 dimensions using algorithms in Part 6 exactly.
Use scipy.signal.find_peaks for pitch. Handle all edge cases. Return structured dict.

### Task 6: `inspector.py`
Write check(measurements) — load tolerances from config, return per-dimension + overall result.
Also write trigger_gpio(result) to control LEDs.

### Task 7: `logger.py`
Write save_inspection(measurements, result), export_csv(), get_stats().
Auto-create SQLite DB and table on first run.

### Task 8: `gui.py`
Write full CustomTkinter GUI as specified in Part 7.

### Task 9: `main.py`
Write entry point that imports and initializes all modules, starts the GUI main loop.

### Task 10: `requirements.txt` + `threadvision.service`
Write pip requirements file and systemd unit file.

---

## PART 10 — QUALITY STANDARDS

Every file you write must meet these standards:

1. **Fully runnable** — no placeholder functions, no `pass` bodies, no `# TODO` comments
2. **Defensive** — all hardware calls wrapped in try/except with meaningful error messages
3. **Commented** — every function has a docstring explaining inputs, outputs, and algorithm
4. **Testable** — each module can be run standalone (use `if __name__ == "__main__":` test block)
5. **Config-driven** — no magic numbers inline; all constants imported from config.py
6. **Logged** — important events (calibration, inspection, errors) written to console with timestamp

---

## PART 11 — DEMO DAY CHECKLIST

Your code must enable this exact demo sequence without failure:
1. Judge walks up. System is powered on, running automatically.
2. Touchscreen shows live camera feed + "READY" status.
3. Smeet places an M8 bolt in the V-block.
4. Smeet taps "INSPECT" on the touchscreen.
5. Within 2 seconds: 5 measurements appear in the table. PASS/FAIL shown in large text. Green or Red LED lights up.
6. Smeet places a deliberately defective bolt (wrong pitch or diameter). System shows FAIL with the failing dimension highlighted in red.
7. Smeet taps "EXPORT CSV" to show the log file.
8. Judges are impressed.

---

## START HERE

Begin with **Task 1: calibrate.py**. Think carefully before writing:
- What could go wrong with mouse click detection in OpenCV?
- How do you handle the coordinate scaling between display resolution and capture resolution?
- What is the correct formula given potential lens distortion at the edges?
- How do you write back to config.py safely without corrupting the rest of the file?

Write the complete, production-grade calibrate.py now.
