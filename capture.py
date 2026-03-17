# capture.py — ThreadVision AI Camera Capture Module
# ====================================================
# Provides a unified camera interface that works on both
# Raspberry Pi (picamera2) and desktop (OpenCV VideoCapture).
#
# Usage:
#   from capture import CameraManager
#   cam = CameraManager()
#   frame = cam.capture_frame()       # full-res still for measurement
#   preview = cam.capture_preview()   # scaled preview for GUI
#   cam.release()

import time
import logging
import numpy as np
import cv2

from config import (
    CAPTURE_WIDTH, CAPTURE_HEIGHT,
    PREVIEW_WIDTH, PREVIEW_HEIGHT,
    IS_RASPBERRY_PI,
    PIXEL_TO_MM, NOMINAL_VALUES,
)

logger = logging.getLogger("ThreadVision.capture")

# ---------------------------------------------------------------------------
# Try to import picamera2 (only available on Raspberry Pi OS)
# ---------------------------------------------------------------------------
_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
    logger.info("picamera2 found — using Pi Camera interface.")
except ImportError:
    logger.info("picamera2 not found — falling back to OpenCV camera / test image.")


class CameraError(Exception):
    """Raised when camera hardware is unavailable or returns a bad frame."""
    pass


class CameraManager:
    """
    Unified camera manager for ThreadVision AI.

    On Raspberry Pi:
        Uses picamera2 with the CSI Pi Camera Module v2.
        Captures at full sensor resolution (3280×2464) for measurement,
        and at preview resolution (640×480) for the GUI feed.

    On Desktop (Windows/Linux without picamera2):
        Falls back to OpenCV VideoCapture (USB webcam) or generates
        a synthetic test pattern if no camera is available.
    """

    def __init__(self):
        """Initialize the camera. Raises CameraError if no camera found."""
        self._camera = None
        self._is_picamera = False
        self._is_open = False

        if _PICAMERA2_AVAILABLE and IS_RASPBERRY_PI:
            self._init_picamera2()
        else:
            self._init_opencv()

    # ------------------------------------------------------------------
    # Initialization backends
    # ------------------------------------------------------------------

    def _init_picamera2(self):
        """Initialize Pi Camera Module v2 via picamera2."""
        try:
            self._camera = Picamera2()

            # Configure for still capture at full resolution
            config = self._camera.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "BGR888"},
                lores={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "BGR888"},
                display="lores",
            )
            self._camera.configure(config)
            self._camera.start()

            # Allow auto-exposure to settle
            time.sleep(2.0)

            self._is_picamera = True
            self._is_open = True
            logger.info(
                f"Pi Camera initialized: main={CAPTURE_WIDTH}×{CAPTURE_HEIGHT}, "
                f"lores={PREVIEW_WIDTH}×{PREVIEW_HEIGHT}"
            )
        except Exception as e:
            raise CameraError(f"Failed to initialize Pi Camera: {e}") from e

    def _init_opencv(self):
        """Initialize fallback camera via OpenCV VideoCapture."""
        try:
            self._camera = cv2.VideoCapture(0)
            if not self._camera.isOpened():
                logger.warning(
                    "No USB camera detected. Will generate synthetic test frames."
                )
                self._camera = None
                self._is_open = True  # Still "open" — we'll generate test images
                return

            # Try to set resolution (USB webcams may not support full res)
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

            actual_w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._is_picamera = False
            self._is_open = True
            logger.info(f"OpenCV camera initialized: {actual_w}×{actual_h}")
        except Exception as e:
            logger.warning(f"OpenCV camera init failed: {e}. Using synthetic frames.")
            self._camera = None
            self._is_open = True

    # ------------------------------------------------------------------
    # Capture methods
    # ------------------------------------------------------------------

    def capture_frame(self):
        """
        Capture a full-resolution frame for measurement.

        Returns:
            np.ndarray: BGR image at CAPTURE_WIDTH × CAPTURE_HEIGHT
                        (or best available resolution on fallback cameras).

        Raises:
            CameraError: If camera returns an invalid frame.
        """
        if not self._is_open:
            raise CameraError("Camera is not initialized.")

        if self._is_picamera:
            return self._capture_picamera_full()
        elif self._camera is not None:
            return self._capture_opencv()
        else:
            return self._generate_test_frame(CAPTURE_WIDTH, CAPTURE_HEIGHT)

    def capture_preview(self):
        """
        Capture a preview-resolution frame for the GUI live feed.

        Returns:
            np.ndarray: BGR image at PREVIEW_WIDTH × PREVIEW_HEIGHT.
        """
        if not self._is_open:
            raise CameraError("Camera is not initialized.")

        if self._is_picamera:
            return self._capture_picamera_preview()
        elif self._camera is not None:
            frame = self._capture_opencv()
            return cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        else:
            return self._generate_test_frame(PREVIEW_WIDTH, PREVIEW_HEIGHT)

    # ------------------------------------------------------------------
    # Backend capture implementations
    # ------------------------------------------------------------------

    def _capture_picamera_full(self):
        """Capture full-resolution still from Pi Camera."""
        try:
            frame = self._camera.capture_array("main")
            if frame is None or frame.size == 0:
                raise CameraError("Pi Camera returned an empty frame.")
            return frame
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Pi Camera capture failed: {e}") from e

    def _capture_picamera_preview(self):
        """Capture low-resolution preview from Pi Camera."""
        try:
            frame = self._camera.capture_array("lores")
            if frame is None or frame.size == 0:
                raise CameraError("Pi Camera preview returned an empty frame.")
            return frame
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Pi Camera preview failed: {e}") from e

    def _capture_opencv(self):
        """Capture frame from OpenCV VideoCapture (USB webcam)."""
        try:
            ret, frame = self._camera.read()
            if not ret or frame is None or frame.size == 0:
                raise CameraError(
                    "OpenCV camera returned no frame. Check USB connection."
                )
            return frame
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"OpenCV capture failed: {e}") from e

    def _generate_test_frame(self, width, height):
        """
        Generate a synthetic bolt silhouette for pipeline testing.

        The bolt is sized relative to the frame so that:
        - width / PIXEL_TO_MM lands in the 4–20mm valid range
        - At least 15 thread teeth are drawn for reliable peak detection
        - The bolt fills ~60% of the frame for realistic contour detection

        This is NOT dimensionally accurate — it's a test pattern to
        verify the pipeline runs end-to-end. Real accuracy comes from
        real camera images after calibration.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            np.ndarray: Synthetic BGR test frame.
        """
        # Bright background (backlit)
        frame = np.full((height, width, 3), 240, dtype=np.uint8)

        cx, cy = width // 2, height // 2

        # Size the bolt to fill ~60% of the frame and land in valid mm range
        # We want bolt_w / PIXEL_TO_MM to be ~8mm (M8 bolt)
        target_major_mm = NOMINAL_VALUES['major_diameter']  # 8.0 mm
        bolt_w = int(target_major_mm * PIXEL_TO_MM)

        # If that's too wide for the frame, scale to fit 60% of width
        max_bolt_w = int(width * 0.6)
        if bolt_w > max_bolt_w:
            bolt_w = max_bolt_w

        # Bolt height: fill ~70% of frame height
        bolt_h = int(height * 0.7)

        # Thread geometry proportional to bolt width
        # Minor diameter ~83% of major (M8: 6.647/8.0 = 0.831)
        minor_w = int(bolt_w * 0.83)
        # Thread depth = (major - minor) / 2 in pixels
        depth_px = (bolt_w - minor_w) // 2

        # Pitch: enough to get 15+ teeth in bolt_h
        # Target ~20 teeth for robust peak detection
        num_teeth = 20
        pitch_px = bolt_h // num_teeth

        # Ensure minimum pitch of 8px for peak detection
        pitch_px = max(pitch_px, 8)
        depth_px = max(depth_px, 4)

        # Bolt shank rectangle (minor diameter = root width)
        x1 = cx - minor_w // 2
        x2 = cx + minor_w // 2
        y1 = cy - bolt_h // 2
        y2 = cy + bolt_h // 2

        # Clamp to frame
        y1 = max(10, y1)
        y2 = min(height - 10, y2)

        # Draw bolt core (minor diameter shank — dark)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)

        # Draw thread teeth — triangular profile on both sides
        actual_teeth = (y2 - y1) // pitch_px

        for i in range(actual_teeth):
            ty = y1 + i * pitch_px

            # Left side tooth (triangle pointing left)
            pts_left = np.array([
                [x1, ty],
                [x1 - depth_px, ty + pitch_px // 2],
                [x1, ty + pitch_px],
            ], np.int32)
            cv2.fillPoly(frame, [pts_left], (25, 25, 25))

            # Right side tooth (triangle pointing right)
            pts_right = np.array([
                [x2, ty],
                [x2 + depth_px, ty + pitch_px // 2],
                [x2, ty + pitch_px],
            ], np.int32)
            cv2.fillPoly(frame, [pts_right], (25, 25, 25))

        # Bolt head (wider block at top)
        head_w = int(bolt_w * 1.5)
        head_h = max(int(bolt_h * 0.08), 15)
        cv2.rectangle(
            frame,
            (cx - head_w // 2, max(5, y1 - head_h)),
            (cx + head_w // 2, y1),
            (25, 25, 25),
            -1,
        )

        # Add slight Gaussian noise for realism
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        logger.debug(
            f"Synthetic bolt: {bolt_w}px wide ({bolt_w/PIXEL_TO_MM:.1f}mm), "
            f"{actual_teeth} teeth, pitch={pitch_px}px, depth={depth_px}px"
        )

        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self):
        """Release camera resources. Safe to call multiple times."""
        if not self._is_open:
            return

        try:
            if self._is_picamera and self._camera is not None:
                self._camera.stop()
                self._camera.close()
                logger.info("Pi Camera released.")
            elif self._camera is not None:
                self._camera.release()
                logger.info("OpenCV camera released.")
        except Exception as e:
            logger.warning(f"Error releasing camera: {e}")
        finally:
            self._camera = None
            self._is_open = False

    def __del__(self):
        """Ensure camera is released on garbage collection."""
        self.release()

    @property
    def is_available(self):
        """Check if a real camera is available (not synthetic)."""
        return self._is_open and self._camera is not None


# ---------------------------------------------------------------------------
# Self-test: run `python capture.py` to verify camera capture works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Camera Capture Self-Test")
    print("=" * 60)

    try:
        cam = CameraManager()
        print(f"  Camera type: {'Pi Camera' if cam._is_picamera else 'OpenCV/Synthetic'}")
        print(f"  Real camera: {cam.is_available}")

        # Capture full-resolution frame
        frame = cam.capture_frame()
        print(f"  Full frame : {frame.shape[1]}×{frame.shape[0]} ({frame.dtype})")

        # Capture preview frame
        preview = cam.capture_preview()
        print(f"  Preview    : {preview.shape[1]}×{preview.shape[0]} ({preview.dtype})")

        # Save test captures
        cv2.imwrite("test_full_capture.png", frame)
        cv2.imwrite("test_preview.png", preview)
        print("  Saved: test_full_capture.png, test_preview.png")

        cam.release()
        print("\n✓ Camera capture test passed.")

    except CameraError as e:
        print(f"\n✗ Camera error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
