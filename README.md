# ThreadVision AI

ThreadVision AI is an automated dimensional inspection system for measuring the physical features of metric threaded bolts (M8–M34) using computer vision. Built specifically for high-precision metrology, it captures images of fasteners and evaluates them against standard ISO 262 thread tolerances to provide instant PASS/FAIL quality control grading.

Designed to be lightweight and rigorous, this system can be deployed on a standard laptop or an embedded edge device (e.g., Raspberry Pi 4 with a Pi Camera V2 and a backlit V-block fixture).

---

## 🎯 Features

*   **ArUco Marker Calibration**: Automatically calculates the exact physical scale (pixels-per-mm) dynamically by detecting a standardized 50mm ArUco marker in the frame, adapting perfectly to different camera focal lengths and distances.
*   **Auto-Orientation**: Detects if the bolt is placed horizontally or vertically, automatically rotating horizontal inputs before processing and unrotating annotation coordinates for the final output.
*   **Noise-Adaptive Preprocessing**: Analyzes the Laplacian variance of the image to categorize the noise level (`CLEAN`, `MODERATE`, `NOISY`). Automatically applies the appropriate pipeline of Gaussian blurs, FastNlMeansDenoising, Otsu thresholding, and morphological operations to extract crisp edges.
*   **5-Axis Dimension Measurements**:
    *   **Major Diameter**: Uses the 97th percentile of the bounding widths for robust crest detection.
    *   **Minor Diameter**: Uses the 10th percentile of the bounding widths to accurately identify thread roots.
    *   **Pitch**: Employs an isolated edge-strip scan and 1D Gaussian signal processing to evaluate crest-to-crest distances, applying adaptive minimum-distance constraints to prevent false peaks.
    *   **Thread Depth**: Derived dynamically based on major and minor diameters.
    *   **Flank Angle**: Computes the half-angles of the thread flanks through geometric line fitting of the rising and falling edges of the thread contour.
*   **ISO Multi-Standard Compliance**: Actively matches the measured geometry against an internal ISO tolerance library, automatically selecting the correct standard (e.g., `M8x1.25`, `M24x3.0`) and reporting strict dimension-by-dimension PASS/FAIL results.
*   **Rich Image Annotation**: Generates visual measurement overlays, including dimension lines, bounding contours, scale bars, ArUco bounds, and a visual PASS/FAIL banner.
*   **CSV Logging**: Optionally logs detailed timestamps and geometry outputs to a CSV file for analytical tracking.

---

## 🛠 System Architecture & Methodology

The entire computer-vision pipeline is housed in `analyze_thread_image.py`. When an image is supplied, the script follows a strict 15-step execution order:

1.  **Image Ingestion**: Loads the raw BGR photo.
2.  **Orientation Normalization**: Assesses the bounding box aspect ratio and corrects rotation. 
3.  **Scale Calibration (`detect_aruco_scale`)**: Scans for a 4x4_50 ArUco marker. If poor lighting prevents standard detection, the system cascades into CLAHE and Adaptive Threshold algorithms to find the marker. Computes `PIXEL_TO_MM` based on marker corner geometry and handles perspective tilting mathematically.
4.  **Noise Assessment**: Characterizes image quality and prescribes a processing mode.
5.  **Preprocessing**: Applies Canny edge detection uniquely tailored to the noise profile.
6.  **Bolt Contour Targeting**: Isolates the largest valid contour mimicking bolt dimensions.
7.  **Diameter Measurement**: Aggregates point-to-point widths across the central 60% of the bolt body to eliminate tip tapering.
8.  **Pitch Threading**: Isolates the left edge of the bolt and converts pixel intensities into a 1D waveform, locating peaks (threads) dynamically and calculating median spacing.
9.  **Angle Calculation**: Splits the thread boundaries and performs least-squares line segments to find reliable flank half-angles before deriving the full included angle.
10. **ISO Validation**: Compares raw dimensions to ISO 262 engineering databases.
11. **Output Generation**: Composes annotated graphics and a formatted terminal report mapping exact millimeter outputs.

---

## 🚀 Quick Start / Usage

### Prerequisites
Developed with Python 3.10. Install the following core libraries:
```bash
pip install opencv-contrib-python numpy scipy
```

### Running the Measurement Script

The script is entirely standalone. You can invoke it via the command line with various optional parameters.

**Basic Run (Opens UI File Dialog):**
```bash
python analyze_thread_image.py
```

**Run on a specific image and show the annotated output popup:**
```bash
python analyze_thread_image.py --image bolt_photo.jpg --show
```

**Run with logging, a specified standard, and save the image:**
```bash
python analyze_thread_image.py --image images/M24.jpeg --standard M24x3.0 --csv measurements.csv --save output_m24.jpg
```

### Available CLI Flags
*   `--image PATH`: Direct path to the image to analyze.
*   `--standard STD`: Impose a specific thread standard (e.g., `M10x1.5`). Default is `AUTO`.
*   `--marker-size MM`: Imposes the physical side length (in mm) of the ArUco marker. Default is `50.0`.
*   `--calibration PX_MM`: Skips ArUco detection entirely and forces a hardcoded `PIXEL_TO_MM` scale.
*   `--show`: Pops up an integrated OpenCV window displaying the final annotated image. 
*   `--save PATH`: Custom file path to save the annotated output image. 
*   `--csv PATH`: Appends the extracted dimensional data to a CSV document.
*   `--verbose`: Outputs detailed debug information across every algorithm step.
*   `--selftest`: Executes a 4-part programmatic self-test verifying internal mathematics (Pitch limits, Flank doubling logic, ArUco bounds, Orientation mocks).

---

## 📷 How to Calibrate Using the ArUco Marker

For accurate millimeters, print an ArUco marker.
1. Generate an ArUco marker from the `4x4_50` dictionary (using [https://chev.me/arucogen/](https://chev.me/arucogen/)).
2. Set the marker size to exactly `50mm`. **Turn off "Scale to Fit" when printing.**
3. Place the printed marker flat beside the bolt ensuring it is on the **same focal plane** (distance from the camera lens).
4. Take a photo. The system will auto-detect the marker and establish the sub-pixel dimensional layout.

If no marker is detected, the script gracefully rolls back to a hardcoded `FALLBACK_PIXEL_TO_MM` variable to prevent unexpected crashes in production lines.

---

## 🧪 Hardware Testing Rig (Optional)
If deploying on a factory floor or hardware implementation:
*   Use a back-illuminated LED light panel behind the bolt. High-contast silhouettes provide vastly superior thread edge clarity over top-down lighting.
*   Place the bolt on a measured V-block fixture.
*   Secure the camera perfectly parallel to the light panel to nullify perspective lens distortion.
