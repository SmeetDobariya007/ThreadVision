# calibrate.py — ThreadVision AI Pixel-to-MM Calibration Tool
# =============================================================
# Standalone tool to calibrate the pixel-to-millimetre conversion
# ratio. This is the MOST CRITICAL step — all measurements depend
# on this value being accurate.
#
# Usage:
#   python calibrate.py
#   python calibrate.py --image path/to/reference.png   (use saved image)
#
# Procedure:
#   1. Place a reference object of KNOWN physical width in the V-block.
#   2. Run this script. It captures (or loads) an image.
#   3. Click the LEFT edge and RIGHT edge of the reference object.
#   4. Enter the known physical width in mm when prompted.
#   5. The script computes PIXEL_TO_MM, validates it, and writes to config.py.

import os
import re
import sys
import math
import logging
import argparse
import datetime
import numpy as np
import cv2

from config import BASE_DIR, CAPTURE_WIDTH, CAPTURE_HEIGHT

logger = logging.getLogger("ThreadVision.calibrate")

# ---------------------------------------------------------------------------
# Constants for this module
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = 1280        # Display window width (scaled for screen)
DISPLAY_HEIGHT = 960        # Display window height (scaled for screen)
VALID_RATIO_MIN = 80.0      # Minimum plausible px/mm
VALID_RATIO_MAX = 200.0     # Maximum plausible px/mm
CROSSHAIR_SIZE = 20         # Size of crosshair lines in pixels
CROSSHAIR_COLOR = (0, 255, 0)     # Green
LINE_COLOR = (0, 200, 255)        # Orange-yellow
TEXT_COLOR = (255, 255, 255)      # White
INFO_BG_COLOR = (40, 40, 40)     # Dark background for text overlay


class CalibrationTool:
    """
    Interactive calibration tool for measuring pixel-to-mm ratio.

    The user clicks two points on a reference object of known physical
    width. The pixel distance between clicks is divided by the known
    mm width to produce the PIXEL_TO_MM calibration constant.

    All calculations are performed in FULL RESOLUTION coordinates,
    even though the display image is scaled down for the screen.
    """

    def __init__(self, image=None):
        """
        Initialize calibration tool.

        Args:
            image: Optional np.ndarray (BGR). If None, captures from camera.
        """
        self.full_res_image = image         # Full-resolution image
        self.display_image = None           # Scaled copy for display
        self.scale_factor = 1.0             # display_px / full_res_px

        self.point1 = None                  # First click (display coords)
        self.point2 = None                  # Second click (display coords)
        self.mouse_pos = None               # Current mouse position
        self.calibration_done = False

    def run(self):
        """
        Execute the full calibration workflow.

        Returns:
            float: The computed PIXEL_TO_MM ratio, or None if cancelled.
        """
        # Step 1: Get image
        if self.full_res_image is None:
            self.full_res_image = self._capture_image()

        if self.full_res_image is None:
            print("✗ No image available for calibration.")
            return None

        # Step 2: Prepare display image
        self._prepare_display()

        # Step 3: Interactive point selection
        print("\n" + "=" * 60)
        print("CALIBRATION MODE")
        print("=" * 60)
        print("Click the LEFT edge of the reference object, then the RIGHT edge.")
        print("Press 'R' to reset points. Press 'Q' or ESC to cancel.")
        print("=" * 60)

        cv2.namedWindow("ThreadVision Calibration", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ThreadVision Calibration", self._mouse_callback)

        while not self.calibration_done:
            display = self._draw_overlay()
            cv2.imshow("ThreadVision Calibration", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nCalibration cancelled by user.")
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):  # Reset
                self.point1 = None
                self.point2 = None
                print("Points reset.")

        cv2.destroyAllWindows()

        # Step 4: Compute pixel distance in full resolution
        pt1_full = self._display_to_full_res(self.point1)
        pt2_full = self._display_to_full_res(self.point2)
        pixel_distance = math.sqrt(
            (pt2_full[0] - pt1_full[0]) ** 2 +
            (pt2_full[1] - pt1_full[1]) ** 2
        )

        print(f"\nPixel distance (full res): {pixel_distance:.1f} px")
        print(f"  Point 1 (full res): ({pt1_full[0]:.0f}, {pt1_full[1]:.0f})")
        print(f"  Point 2 (full res): ({pt2_full[0]:.0f}, {pt2_full[1]:.0f})")

        # Step 5: Get known physical width
        known_mm = self._get_physical_width()
        if known_mm is None:
            return None

        # Step 6: Compute ratio
        pixel_to_mm = pixel_distance / known_mm
        um_per_pixel = 1e3 / pixel_to_mm

        # Step 7: Validate
        print("\n" + "=" * 60)
        print("CALIBRATION RESULT")
        print("=" * 60)
        print(f"  PIXEL_TO_MM    = {pixel_to_mm:.4f} px/mm")
        print(f"  Resolution     = {um_per_pixel:.2f} µm/pixel")
        print(f"  M8 major dia.  = {8.0 * pixel_to_mm:.0f} px (expected span)")
        print(f"  Pitch (1.25mm) = {1.25 * pixel_to_mm:.0f} px (expected span)")

        if pixel_to_mm < VALID_RATIO_MIN or pixel_to_mm > VALID_RATIO_MAX:
            print(f"\n⚠ WARNING: Value {pixel_to_mm:.2f} is outside the expected ")
            print(f"  range of {VALID_RATIO_MIN}–{VALID_RATIO_MAX} px/mm.")
            print("  Re-check camera distance and reference object dimensions.")
            confirm = input("\nSave anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Calibration discarded.")
                return None
        else:
            print(f"\n✓ Value is within expected range ({VALID_RATIO_MIN}–{VALID_RATIO_MAX} px/mm)")

        # Step 8: Write to config.py
        self._write_config(pixel_to_mm)

        # Step 9: Log calibration event
        self._log_calibration(known_mm, pixel_distance, pixel_to_mm)

        print(f"\n✓ Calibration complete. PIXEL_TO_MM = {pixel_to_mm:.4f}")
        return pixel_to_mm

    # ------------------------------------------------------------------
    # Image acquisition
    # ------------------------------------------------------------------

    def _capture_image(self):
        """
        Capture an image from the camera for calibration.

        Returns:
            np.ndarray: BGR image at full resolution, or None on failure.
        """
        try:
            from capture import CameraManager, CameraError
            cam = CameraManager()
            print("Capturing calibration image...")
            frame = cam.capture_frame()
            cam.release()
            print(f"  Captured: {frame.shape[1]}×{frame.shape[0]}")
            return frame
        except Exception as e:
            logger.warning(f"Camera capture failed: {e}")
            print(f"⚠ Camera not available: {e}")
            print("  Use --image flag to load a saved image instead.")
            return None

    def _prepare_display(self):
        """Scale the full-resolution image to display size, computing scale factor."""
        h, w = self.full_res_image.shape[:2]

        # Compute scale to fit within DISPLAY_WIDTH × DISPLAY_HEIGHT
        scale_x = DISPLAY_WIDTH / w
        scale_y = DISPLAY_HEIGHT / h
        self.scale_factor = min(scale_x, scale_y)

        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)

        self.display_image = cv2.resize(
            self.full_res_image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        logger.info(
            f"Display: {new_w}×{new_h} (scale={self.scale_factor:.4f} "
            f"from {w}×{h})"
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _display_to_full_res(self, display_point):
        """
        Convert display coordinates to full-resolution coordinates.

        Args:
            display_point: (x, y) tuple in display image coords.

        Returns:
            (x, y) tuple in full-resolution image coords.
        """
        x = display_point[0] / self.scale_factor
        y = display_point[1] / self.scale_factor
        return (x, y)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        """
        OpenCV mouse callback for point selection.

        First click sets point1, second click sets point2 and
        marks calibration as ready for measurement input.
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.point1 is None:
                self.point1 = (x, y)
                logger.info(f"Point 1 set: ({x}, {y}) display coords")
            elif self.point2 is None:
                self.point2 = (x, y)
                self.calibration_done = True
                logger.info(f"Point 2 set: ({x}, {y}) display coords")

    # ------------------------------------------------------------------
    # Display overlay rendering
    # ------------------------------------------------------------------

    def _draw_overlay(self):
        """
        Draw crosshairs, measurements, and instructions on the display image.

        Returns:
            np.ndarray: Annotated display image (copy of original).
        """
        display = self.display_image.copy()
        h, w = display.shape[:2]

        # Draw crosshair at point 1
        if self.point1 is not None:
            self._draw_crosshair(display, self.point1, CROSSHAIR_COLOR)
            cv2.putText(
                display, "P1", (self.point1[0] + 12, self.point1[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CROSSHAIR_COLOR, 2,
            )

        # Draw crosshair at point 2
        if self.point2 is not None:
            self._draw_crosshair(display, self.point2, CROSSHAIR_COLOR)
            cv2.putText(
                display, "P2", (self.point2[0] + 12, self.point2[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CROSSHAIR_COLOR, 2,
            )

        # Draw line between points (or to mouse cursor if only P1 set)
        if self.point1 is not None:
            end_point = self.point2 if self.point2 else self.mouse_pos
            if end_point is not None:
                cv2.line(display, self.point1, end_point, LINE_COLOR, 2)

                # Compute and display live pixel distance
                dx = end_point[0] - self.point1[0]
                dy = end_point[1] - self.point1[1]
                dist_display = math.sqrt(dx*dx + dy*dy)
                dist_full = dist_display / self.scale_factor

                # Distance label at midpoint
                mx = (self.point1[0] + end_point[0]) // 2
                my = (self.point1[1] + end_point[1]) // 2
                label = f"{dist_full:.1f} px (full res)"
                cv2.putText(
                    display, label, (mx + 10, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2,
                )

        # Info bar at top
        self._draw_info_bar(display)

        return display

    def _draw_crosshair(self, image, center, color):
        """Draw a crosshair marker at the specified center point."""
        x, y = center
        cv2.line(image, (x - CROSSHAIR_SIZE, y), (x + CROSSHAIR_SIZE, y), color, 2)
        cv2.line(image, (x, y - CROSSHAIR_SIZE), (x, y + CROSSHAIR_SIZE), color, 2)
        cv2.circle(image, center, 4, color, -1)

    def _draw_info_bar(self, display):
        """Draw instruction text overlay at top of image."""
        h, w = display.shape[:2]

        # Semi-transparent background bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), INFO_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        if self.point1 is None:
            text = "Click LEFT edge of reference object"
        elif self.point2 is None:
            text = "Click RIGHT edge of reference object | R=Reset | Q=Cancel"
        else:
            text = "Both points selected. Processing..."

        cv2.putText(
            display, text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2,
        )

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------

    def _get_physical_width(self):
        """
        Prompt the user to enter the known physical width of the reference.

        Returns:
            float: Width in mm, or None if cancelled.
        """
        print("\nEnter the known physical width of the reference object.")
        print("(Use a micrometer or calipers for best accuracy.)")

        while True:
            try:
                raw = input("  Width in mm: ").strip()
                if raw.lower() in ('q', 'quit', 'cancel', ''):
                    print("Calibration cancelled.")
                    return None
                mm = float(raw)
                if mm <= 0:
                    print("  ✗ Width must be positive.")
                    continue
                if mm > 50:
                    print(f"  ⚠ {mm} mm seems large. Are you sure? (y/N)")
                    if input("  ").strip().lower() != 'y':
                        continue
                return mm
            except ValueError:
                print("  ✗ Invalid number. Enter a decimal value (e.g. 8.000)")

    # ------------------------------------------------------------------
    # Config file update
    # ------------------------------------------------------------------

    def _write_config(self, pixel_to_mm):
        """
        Rewrite ONLY the PIXEL_TO_MM line in config.py.

        Uses regex to find and replace the line while preserving
        all other content. Creates a backup before writing.

        Args:
            pixel_to_mm: The new calibration ratio to write.
        """
        config_path = os.path.join(BASE_DIR, "config.py")

        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return

        # Read current content
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create backup
        backup_path = config_path + ".bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Config backup saved: {backup_path}")

        # Replace PIXEL_TO_MM line using regex
        # Matches: PIXEL_TO_MM = <any number> followed by optional comment
        pattern = r'^(PIXEL_TO_MM\s*=\s*)[\d.]+(.*)$'
        replacement = rf'\g<1>{pixel_to_mm:.4f}\2'
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

        if count == 0:
            print("✗ Could not find PIXEL_TO_MM line in config.py!")
            print("  Add this line manually: PIXEL_TO_MM = {:.4f}".format(pixel_to_mm))
            return

        # Write updated content
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  ✓ Updated config.py: PIXEL_TO_MM = {pixel_to_mm:.4f}")
        print(f"  ✓ Backup saved: {backup_path}")

    # ------------------------------------------------------------------
    # Calibration logging
    # ------------------------------------------------------------------

    def _log_calibration(self, reference_mm, pixel_count, ratio):
        """
        Log the calibration event to the database.

        Args:
            reference_mm: Known physical width of the reference.
            pixel_count: Measured pixel width at full resolution.
            ratio: Computed PIXEL_TO_MM value.
        """
        try:
            import sqlite3
            from config import DB_PATH, LOG_DIR

            os.makedirs(LOG_DIR, exist_ok=True)

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    reference_mm REAL NOT NULL,
                    pixel_count REAL NOT NULL,
                    pixel_to_mm REAL NOT NULL,
                    um_per_pixel REAL NOT NULL
                )
            """)

            cursor.execute("""
                INSERT INTO calibration_log
                    (timestamp, reference_mm, pixel_count, pixel_to_mm, um_per_pixel)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.datetime.now().isoformat(),
                reference_mm,
                pixel_count,
                ratio,
                1e3 / ratio,
            ))

            conn.commit()
            conn.close()
            print("  ✓ Calibration logged to database.")
        except Exception as e:
            logger.warning(f"Failed to log calibration: {e}")
            print(f"  ⚠ Could not log calibration: {e}")


class CalibrationError(Exception): pass

def detect_aruco_scale(image, marker_size_mm):
    from cv2 import aruco
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        raise CalibrationError("ArUco marker not found in image.")

    marker_idx = np.where(ids == 0)[0]
    if len(marker_idx) > 0:
        pts = corners[marker_idx[0]][0]
    else:
        pts = corners[0][0]

    side_0 = np.linalg.norm(pts[0] - pts[1])
    side_1 = np.linalg.norm(pts[1] - pts[2])
    side_2 = np.linalg.norm(pts[2] - pts[3])
    side_3 = np.linalg.norm(pts[3] - pts[0])

    sides = np.sort([side_0, side_1, side_2, side_3])
    size_px = np.mean(sides[1:3])

    mm_per_pixel = marker_size_mm / size_px

    # Calculate confidence based on perspective skew
    max_skew = abs(side_0 - side_2) + abs(side_1 - side_3)
    if max_skew < 5:
        confidence = "HIGH"
    elif max_skew < 15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW (High perspective distortion)"

    return mm_per_pixel, confidence, pts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """CLI entry point for the calibration tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ThreadVision AI — Pixel-to-MM Calibration Tool"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to a saved reference image (skip camera capture)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║       ThreadVision AI — Calibration Tool         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    image = None
    if args.image:
        if not os.path.exists(args.image):
            print(f"✗ Image file not found: {args.image}")
            sys.exit(1)
        image = cv2.imread(args.image)
        if image is None:
            print(f"✗ Failed to load image: {args.image}")
            sys.exit(1)
        print(f"  Loaded image: {args.image} ({image.shape[1]}×{image.shape[0]})")

    tool = CalibrationTool(image=image)
    result = tool.run()

    if result is not None:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  PIXEL_TO_MM      : {result:.4f} px/mm")
        print(f"  1 pixel          : {1e3 / result:.2f} µm")
        print(f"  M8 major (8mm)   : {8.0 * result:.0f} pixels wide")
        print(f"  Pitch (1.25mm)   : {1.25 * result:.0f} pixels between crests")
        print(f"  Thread depth     : {0.677 * result:.0f} pixels deep")
        print("=" * 60)
        print("\n✓ System is ready for inspection. Run: python main.py")
    else:
        print("\n✗ Calibration was not completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
