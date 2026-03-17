# main.py — ThreadVision AI Entry Point
# ========================================
# Initializes all subsystems and launches the GUI.
# Called by systemd on Raspberry Pi boot, or run manually.
#
# Usage:
#   python main.py              ← normal launch
#   python main.py --no-gui     ← headless single inspection (for testing)
#   python main.py --calibrate  ← launch calibration tool instead

import os
import sys
import time
import signal
import logging
import argparse
import datetime

# ---------------------------------------------------------------------------
# Logging setup — must happen before any other imports
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose=False):
    """Configure logging for the entire application."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


logger = logging.getLogger("ThreadVision.main")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ThreadVisionApp:
    """
    Main application controller for ThreadVision AI.

    Initializes hardware (GPIO, camera, backlight), starts the GUI,
    and handles graceful shutdown on SIGINT/SIGTERM.
    """

    def __init__(self):
        self._running = False

    def start(self, headless=False):
        """
        Start the ThreadVision application.

        Args:
            headless (bool): If True, run a single inspection without GUI.
        """
        logger.info("=" * 60)
        logger.info("  ThreadVision AI — Starting")
        logger.info(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Working dir: {os.getcwd()}")
        logger.info("=" * 60)

        # Import config after logging is set up
        from config import (
            THREAD_STANDARD, PIXEL_TO_MM,
            IS_RASPBERRY_PI, BASE_DIR, LOG_DIR,
        )

        logger.info(f"  Thread standard : {THREAD_STANDARD}")
        logger.info(f"  PIXEL_TO_MM     : {PIXEL_TO_MM:.4f} px/mm")
        logger.info(f"  Platform        : {'Raspberry Pi' if IS_RASPBERRY_PI else 'Desktop'}")
        logger.info(f"  Base directory  : {BASE_DIR}")

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # Initialize GPIO
        self._init_gpio()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True

        if headless:
            self._run_headless()
        else:
            self._run_gui()

    def _init_gpio(self):
        """Initialize GPIO pins for LEDs and backlight."""
        try:
            from inspector import setup_gpio, set_backlight
            setup_gpio()
            set_backlight(True)  # Turn on backlight for inspection
            logger.info("GPIO and backlight initialized.")
        except Exception as e:
            logger.warning(f"GPIO init skipped: {e}")

    def _run_gui(self):
        """Launch the CustomTkinter GUI interface."""
        try:
            from gui import ThreadVisionGUI

            logger.info("Launching GUI...")
            app = ThreadVisionGUI()
            app.run()

        except ImportError as e:
            logger.error(f"GUI dependency missing: {e}")
            logger.error("Install with: pip install customtkinter Pillow")
            sys.exit(1)
        except Exception as e:
            logger.error(f"GUI failed: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self._shutdown()

    def _run_headless(self):
        """
        Run a single inspection without GUI.

        Useful for testing the pipeline from the command line.
        """
        logger.info("Running headless inspection...")

        try:
            from capture import CameraManager
            from preprocessor import preprocess
            from measurement import measure
            from inspector import check, trigger_gpio
            from logger import save_inspection

            # Capture
            logger.info("Step 1: Capturing frame...")
            cam = CameraManager()
            frame = cam.capture_frame()
            cam.release()
            logger.info(f"  Frame: {frame.shape[1]}×{frame.shape[0]}")

            # Preprocess
            logger.info("Step 2: Preprocessing...")
            edges, gray = preprocess(frame)

            # Measure
            logger.info("Step 3: Measuring dimensions...")
            measurements = measure(edges, gray)

            # Inspect
            logger.info("Step 4: Checking tolerances...")
            result, overall_str = check(measurements)

            # GPIO
            logger.info("Step 5: Triggering GPIO...")
            trigger_gpio(result.overall_pass)

            # Log
            logger.info("Step 6: Saving to database...")
            row_id = save_inspection(measurements, result, overall_str)

            # Print results
            print()
            print("=" * 60)
            print(f"  RESULT: {overall_str}")
            print("=" * 60)
            for dim_key in ['major_diameter', 'minor_diameter', 'pitch',
                             'thread_depth', 'flank_angle']:
                info = result.per_dimension.get(dim_key, {})
                val = info.get('value')
                status = info.get('status', 'N/A')
                unit = info.get('unit', 'mm')
                if val is not None:
                    print(f"  {info.get('name', dim_key):16s} : "
                          f"{val:7.3f} {unit}  [{status}]")
                else:
                    print(f"  {info.get('name', dim_key):16s} : "
                          f"{'N/A':>7s}      [{status}]")
            print(f"\n  Record saved as #{row_id}")
            print("=" * 60)

            if result.per_dimension:
                warnings = measurements.get('warnings', [])
                if warnings:
                    print("\n  Warnings:")
                    for w in warnings:
                        print(f"    ⚠ {w}")

        except Exception as e:
            logger.error(f"Headless inspection failed: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self._shutdown()

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Signal {signum} received — shutting down...")
        self._running = False
        self._shutdown()
        sys.exit(0)

    def _shutdown(self):
        """Clean up all resources."""
        logger.info("Shutting down ThreadVision AI...")

        try:
            from inspector import cleanup_gpio, set_backlight
            set_backlight(False)
            cleanup_gpio()
        except Exception as e:
            logger.debug(f"GPIO cleanup: {e}")

        logger.info("ThreadVision AI stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Parse CLI arguments and launch the application."""
    parser = argparse.ArgumentParser(
        description="ThreadVision AI — Automated Thread Inspection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                   Launch GUI (normal operation)
  python main.py --no-gui          Headless single inspection
  python main.py --calibrate       Run calibration tool
  python main.py --verbose         Enable debug logging
        """,
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run a single headless inspection (no GUI)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Launch the calibration tool instead",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    # Setup logging first
    setup_logging(verbose=args.verbose)

    # Banner
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         ThreadVision AI — Thread Inspection         ║")
    print("║         AI-Powered Non-Contact Measurement          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Target: M8 × 1.25 ISO 262 Metric Thread           ║")
    print("║  Platform: Raspberry Pi 4 + Pi Camera v2            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    if args.calibrate:
        # Launch calibration tool
        from calibrate import main as calibrate_main
        calibrate_main()
    else:
        # Launch main application
        app = ThreadVisionApp()
        app.start(headless=args.no_gui)


if __name__ == "__main__":
    main()
