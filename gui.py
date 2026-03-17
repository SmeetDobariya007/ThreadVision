# gui.py — ThreadVision AI Touchscreen Interface
# =================================================
# Full CustomTkinter GUI for the thread inspection system.
# Designed for a 7" DSI touchscreen (800×480) with finger-friendly
# buttons and live camera preview.
#
# Layout matches the Part 7 specification exactly.
#
# Usage:
#   from gui import ThreadVisionGUI
#   app = ThreadVisionGUI()
#   app.run()

import os
import sys
import time
import logging
import threading
import datetime
import numpy as np
import cv2
from PIL import Image, ImageTk

import customtkinter as ctk

from config import (
    PREVIEW_WIDTH, PREVIEW_HEIGHT,
    THREAD_STANDARD, NOMINAL_VALUES, TOLERANCES,
    PIXEL_TO_MM, BASE_DIR,
)

logger = logging.getLogger("ThreadVision.gui")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLOR_BG          = "#1a1a2e"     # Deep navy background
COLOR_SURFACE     = "#16213e"     # Card/panel background
COLOR_SURFACE_ALT = "#0f3460"     # Alternate surface
COLOR_PRIMARY     = "#533483"     # Primary accent (purple)
COLOR_PASS        = "#1DB954"     # Spotify green — PASS
COLOR_FAIL        = "#E22134"     # Red — FAIL
COLOR_WARNING     = "#FFA726"     # Amber — UNMEASURED
COLOR_TEXT        = "#e4e4e4"     # Primary text
COLOR_TEXT_DIM    = "#8d99ae"     # Secondary text
COLOR_BORDER      = "#2b2d42"     # Border/separator
COLOR_BUTTON      = "#533483"     # Button fill
COLOR_BUTTON_HOVER = "#6c44a2"   # Button hover


class ThreadVisionGUI:
    """
    Main GUI application for ThreadVision AI.

    Creates a full-screen touchscreen interface with:
    - Live camera preview (640×480)
    - Measurement results table
    - Large PASS/FAIL result banner
    - Action buttons (INSPECT, EXPORT, HISTORY)
    - Daily statistics bar
    """

    def __init__(self):
        """Initialize the GUI and all sub-components."""
        self._setup_window()
        self._create_layout()
        self._create_camera_panel()
        self._create_results_panel()
        self._create_result_banner()
        self._create_action_bar()
        self._create_stats_bar()

        # State
        self._camera = None
        self._camera_thread = None
        self._camera_running = False
        self._current_frame = None
        self._inspection_running = False

        # Start camera preview
        self._start_camera()

    # ==================================================================
    # Window setup
    # ==================================================================

    def _setup_window(self):
        """Configure the main window."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("ThreadVision AI — Thread Inspection System")
        self.root.configure(fg_color=COLOR_BG)

        # Get screen size for responsive layout
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Use 1024×600 for desktop, fullscreen on Pi (800×480)
        win_w = min(1024, screen_w)
        win_h = min(650, screen_h)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(800, 480)

        # Bind cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # Layout creation
    # ==================================================================

    def _create_layout(self):
        """Create the main layout grid."""
        # Top bar
        self.top_bar = ctk.CTkFrame(self.root, fg_color=COLOR_SURFACE, height=50)
        self.top_bar.pack(fill="x", padx=5, pady=(5, 2))
        self.top_bar.pack_propagate(False)

        # Title
        title_label = ctk.CTkLabel(
            self.top_bar,
            text="⚙  ThreadVision AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXT,
        )
        title_label.pack(side="left", padx=15, pady=8)

        # Standard badge
        std_label = ctk.CTkLabel(
            self.top_bar,
            text=f"ISO {THREAD_STANDARD}",
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT_DIM,
            fg_color=COLOR_SURFACE_ALT,
            corner_radius=12,
            width=100,
        )
        std_label.pack(side="left", padx=10, pady=10)

        # Status indicator
        self.status_label = ctk.CTkLabel(
            self.top_bar,
            text="● READY",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PASS,
        )
        self.status_label.pack(side="right", padx=15, pady=10)

        # Calibration value display
        um_per_px = 1e3 / PIXEL_TO_MM
        cal_label = ctk.CTkLabel(
            self.top_bar,
            text=f"Cal: {PIXEL_TO_MM:.1f} px/mm ({um_per_px:.1f} µm/px)",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_DIM,
        )
        cal_label.pack(side="right", padx=10, pady=10)

        # Middle content area
        self.content_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        self.content_frame.pack(fill="both", expand=True, padx=5, pady=2)

        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

    def _create_camera_panel(self):
        """Create the live camera preview panel."""
        camera_frame = ctk.CTkFrame(
            self.content_frame, fg_color=COLOR_SURFACE, corner_radius=10,
        )
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3), pady=0)

        # Camera label header
        cam_header = ctk.CTkLabel(
            camera_frame,
            text="LIVE CAMERA FEED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        cam_header.pack(pady=(8, 4))

        # Camera preview canvas (using CTkLabel with image)
        self.camera_label = ctk.CTkLabel(
            camera_frame,
            text="Initializing camera...",
            font=ctk.CTkFont(size=14),
            text_color=COLOR_TEXT_DIM,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
        )
        self.camera_label.pack(padx=10, pady=(0, 10), expand=True)

    def _create_results_panel(self):
        """Create the measurement results table."""
        results_frame = ctk.CTkFrame(
            self.content_frame, fg_color=COLOR_SURFACE, corner_radius=10,
        )
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0), pady=0)

        # Table header
        header_frame = ctk.CTkFrame(results_frame, fg_color=COLOR_SURFACE_ALT)
        header_frame.pack(fill="x", padx=8, pady=(8, 4))

        for text, w in [("Dimension", 140), ("Value", 90), ("Status", 80)]:
            ctk.CTkLabel(
                header_frame,
                text=text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLOR_TEXT_DIM,
                width=w,
                anchor="w",
            ).pack(side="left", padx=6, pady=6)

        # Dimension rows
        self.dim_rows = {}
        dimensions = [
            ('major_diameter', 'Major Dia.'),
            ('minor_diameter', 'Minor Dia.'),
            ('pitch',          'Pitch'),
            ('thread_depth',   'Thread Depth'),
            ('flank_angle',    'Flank Angle'),
        ]

        for dim_key, dim_label in dimensions:
            row_frame = ctk.CTkFrame(results_frame, fg_color=COLOR_SURFACE, height=42)
            row_frame.pack(fill="x", padx=8, pady=1)
            row_frame.pack_propagate(False)

            name_lbl = ctk.CTkLabel(
                row_frame,
                text=dim_label,
                font=ctk.CTkFont(size=13),
                text_color=COLOR_TEXT,
                width=140,
                anchor="w",
            )
            name_lbl.pack(side="left", padx=6, pady=4)

            value_lbl = ctk.CTkLabel(
                row_frame,
                text="—",
                font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
                text_color=COLOR_TEXT,
                width=90,
                anchor="e",
            )
            value_lbl.pack(side="left", padx=6, pady=4)

            status_lbl = ctk.CTkLabel(
                row_frame,
                text="READY",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLOR_TEXT_DIM,
                width=80,
            )
            status_lbl.pack(side="left", padx=6, pady=4)

            self.dim_rows[dim_key] = {
                'frame': row_frame,
                'value': value_lbl,
                'status': status_lbl,
            }

    def _create_result_banner(self):
        """Create the large PASS/FAIL result banner."""
        self.banner_frame = ctk.CTkFrame(
            self.root, fg_color=COLOR_SURFACE, height=70, corner_radius=10,
        )
        self.banner_frame.pack(fill="x", padx=5, pady=3)
        self.banner_frame.pack_propagate(False)

        self.result_label = ctk.CTkLabel(
            self.banner_frame,
            text="READY",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        self.result_label.pack(expand=True)

    def _create_action_bar(self):
        """Create the action buttons bar."""
        action_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG, height=70)
        action_frame.pack(fill="x", padx=5, pady=2)

        # INSPECT button (primary action)
        self.inspect_btn = ctk.CTkButton(
            action_frame,
            text="🔍  INSPECT",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_BUTTON_HOVER,
            height=60,
            corner_radius=12,
            command=self._on_inspect,
        )
        self.inspect_btn.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        # EXPORT CSV button
        self.export_btn = ctk.CTkButton(
            action_frame,
            text="📄  EXPORT CSV",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_SURFACE_ALT,
            hover_color=COLOR_BUTTON_HOVER,
            height=60,
            corner_radius=12,
            command=self._on_export,
        )
        self.export_btn.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        # HISTORY button
        self.history_btn = ctk.CTkButton(
            action_frame,
            text="📋  HISTORY",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLOR_SURFACE_ALT,
            hover_color=COLOR_BUTTON_HOVER,
            height=60,
            corner_radius=12,
            command=self._on_history,
        )
        self.history_btn.pack(side="left", padx=5, pady=5, expand=True, fill="x")

    def _create_stats_bar(self):
        """Create the bottom statistics bar."""
        self.stats_frame = ctk.CTkFrame(
            self.root, fg_color=COLOR_SURFACE, height=35, corner_radius=8,
        )
        self.stats_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.stats_frame.pack_propagate(False)

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Parts today: 0 | Pass: 0 | Fail: 0 | Rate: —",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM,
        )
        self.stats_label.pack(expand=True)

        # Load initial stats
        self._update_stats()

    # ==================================================================
    # Camera preview
    # ==================================================================

    def _start_camera(self):
        """Start the live camera preview in a background thread."""
        self._camera_running = True
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True
        )
        self._camera_thread.start()

    def _camera_loop(self):
        """Background thread: continuously capture and display preview frames."""
        try:
            from capture import CameraManager, CameraError
            self._camera = CameraManager()
        except Exception as e:
            logger.warning(f"Camera init failed in preview thread: {e}")
            self.root.after(0, lambda: self.camera_label.configure(
                text=f"Camera unavailable:\n{e}",
                image=None,
            ))
            return

        while self._camera_running:
            try:
                frame = self._camera.capture_preview()
                self._current_frame = frame

                # Convert BGR (OpenCV) to RGB (PIL/Tkinter)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                ctk_img = ctk.CTkImage(light_image=pil_img, size=(PREVIEW_WIDTH, PREVIEW_HEIGHT))

                # Update on main thread
                self.root.after(0, lambda img=ctk_img: self.camera_label.configure(
                    text="", image=img,
                ))

            except Exception as e:
                logger.debug(f"Preview frame error: {e}")

            # ~20 FPS
            time.sleep(0.05)

    # ==================================================================
    # Inspection pipeline
    # ==================================================================

    def _on_inspect(self):
        """Handle INSPECT button press — run the full measurement pipeline."""
        if self._inspection_running:
            return

        self._inspection_running = True
        self.inspect_btn.configure(state="disabled", text="⏳ INSPECTING...")
        self.status_label.configure(text="● INSPECTING", text_color=COLOR_WARNING)
        self._reset_results()

        # Run pipeline in background thread
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        """Execute the full inspection pipeline in a background thread."""
        try:
            from capture import CameraManager, CameraError
            from preprocessor import preprocess
            from measurement import measure, MeasurementError
            from inspector import check, trigger_gpio
            from logger import save_inspection

            # Step 1: Capture full-resolution frame
            if self._camera is not None:
                frame = self._camera.capture_frame()
            else:
                cam = CameraManager()
                frame = cam.capture_frame()
                cam.release()

            # Step 2: Preprocess
            edges, gray = preprocess(frame)

            # Step 3: Measure
            measurements = measure(edges, gray)

            # Step 4: Inspect
            result, overall_str = check(measurements)

            # Step 5: GPIO
            trigger_gpio(result.overall_pass)

            # Step 6: Log
            save_inspection(measurements, result, overall_str)

            # Step 7: Update GUI on main thread
            self.root.after(0, lambda: self._display_results(
                measurements, result, overall_str
            ))

        except (MeasurementError, Exception) as e:
            import traceback
            logger.error(f"Pipeline error: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: self._display_error(str(e)))

        finally:
            self.root.after(0, self._inspection_complete)

    def _inspection_complete(self):
        """Re-enable the inspect button after pipeline completes."""
        self._inspection_running = False
        self.inspect_btn.configure(state="normal", text="🔍  INSPECT")

    # ==================================================================
    # Results display
    # ==================================================================

    def _reset_results(self):
        """Reset all result displays to default state."""
        for dim_key, widgets in self.dim_rows.items():
            widgets['value'].configure(text="...", text_color=COLOR_TEXT_DIM)
            widgets['status'].configure(text="...", text_color=COLOR_TEXT_DIM)
            widgets['frame'].configure(fg_color=COLOR_SURFACE)

        self.result_label.configure(text="INSPECTING...", text_color=COLOR_WARNING)
        self.banner_frame.configure(fg_color=COLOR_SURFACE)

    def _display_results(self, measurements, result, overall_str):
        """
        Update the GUI with inspection results.

        Args:
            measurements: dict from measure()
            result: InspectionResult from check()
            overall_str: 'PASS', 'FAIL', or 'UNMEASURED'
        """
        per_dim = result.per_dimension

        for dim_key, widgets in self.dim_rows.items():
            info = per_dim.get(dim_key, {})
            value = info.get('value')
            status = info.get('status', 'N/A')
            unit = info.get('unit', 'mm')

            # Format value
            if value is not None:
                if unit == '°':
                    val_text = f"{value:.1f}°"
                else:
                    val_text = f"{value:.3f}mm"
            else:
                val_text = "N/A"

            # Status color
            if status == 'PASS':
                status_text = "✓ PASS"
                status_color = COLOR_PASS
                row_color = COLOR_SURFACE
            elif status == 'FAIL':
                status_text = "✗ FAIL"
                status_color = COLOR_FAIL
                row_color = "#3d1111"  # Dark red tint
            else:
                status_text = "? N/A"
                status_color = COLOR_WARNING
                row_color = COLOR_SURFACE

            widgets['value'].configure(text=val_text, text_color=COLOR_TEXT)
            widgets['status'].configure(text=status_text, text_color=status_color)
            widgets['frame'].configure(fg_color=row_color)

        # Update banner
        if overall_str == 'PASS':
            self.result_label.configure(text="██  PASS  ██", text_color="#FFFFFF")
            self.banner_frame.configure(fg_color=COLOR_PASS)
            self.status_label.configure(text="● PASS", text_color=COLOR_PASS)
        elif overall_str == 'FAIL':
            self.result_label.configure(text="██  FAIL  ██", text_color="#FFFFFF")
            self.banner_frame.configure(fg_color=COLOR_FAIL)
            self.status_label.configure(text="● FAIL", text_color=COLOR_FAIL)
        else:
            self.result_label.configure(text="UNMEASURED", text_color=COLOR_WARNING)
            self.banner_frame.configure(fg_color=COLOR_SURFACE)
            self.status_label.configure(text="● UNMEASURED", text_color=COLOR_WARNING)

        # Update stats
        self._update_stats()

    def _display_error(self, error_msg):
        """Display an error state in the results."""
        self.result_label.configure(text=f"ERROR", text_color=COLOR_FAIL)
        self.banner_frame.configure(fg_color="#3d1111")
        self.status_label.configure(text="● ERROR", text_color=COLOR_FAIL)
        logger.error(f"Pipeline error displayed: {error_msg}")

    # ==================================================================
    # Action handlers
    # ==================================================================

    def _on_export(self):
        """Handle EXPORT CSV button press."""
        try:
            from logger import export_csv
            csv_path = export_csv()
            self.status_label.configure(
                text=f"● Exported CSV", text_color=COLOR_PASS,
            )
            logger.info(f"CSV exported: {csv_path}")

            # Show brief confirmation, then reset status after 3s
            self.root.after(3000, lambda: self.status_label.configure(
                text="● READY", text_color=COLOR_PASS,
            ))
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            self.status_label.configure(
                text="● Export failed", text_color=COLOR_FAIL,
            )

    def _on_history(self):
        """Handle HISTORY button press — show a popup with recent inspections."""
        try:
            from logger import get_history
            records = get_history(limit=20)
        except Exception as e:
            logger.error(f"History load failed: {e}")
            return

        # Create a popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title("Inspection History")
        popup.geometry("700x450")
        popup.configure(fg_color=COLOR_BG)
        popup.transient(self.root)
        popup.grab_set()

        # Header
        ctk.CTkLabel(
            popup,
            text="📋 Recent Inspections",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLOR_TEXT,
        ).pack(pady=(15, 10))

        # Scrollable frame for records
        scroll = ctk.CTkScrollableFrame(
            popup, fg_color=COLOR_SURFACE, corner_radius=10,
        )
        scroll.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        if not records:
            ctk.CTkLabel(
                scroll, text="No inspection records yet.",
                text_color=COLOR_TEXT_DIM,
            ).pack(pady=20)
        else:
            # Table header row
            hdr = ctk.CTkFrame(scroll, fg_color=COLOR_SURFACE_ALT)
            hdr.pack(fill="x", pady=(5, 3))
            for text, w in [("#", 40), ("Time", 160), ("Result", 80),
                            ("Major Ø", 80), ("Pitch", 80)]:
                ctk.CTkLabel(
                    hdr, text=text, font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLOR_TEXT_DIM, width=w, anchor="w",
                ).pack(side="left", padx=4, pady=4)

            for rec in records:
                row = ctk.CTkFrame(scroll, fg_color=COLOR_SURFACE, height=32)
                row.pack(fill="x", pady=1)
                row.pack_propagate(False)

                res_color = (
                    COLOR_PASS if rec['overall_result'] == 'PASS'
                    else COLOR_FAIL if rec['overall_result'] == 'FAIL'
                    else COLOR_WARNING
                )

                for text, w in [
                    (f"#{rec['id']}", 40),
                    (rec['timestamp'][:19], 160),
                    (rec['overall_result'], 80),
                    (f"{rec['major_diameter']:.3f}" if rec['major_diameter'] else "—", 80),
                    (f"{rec['pitch']:.3f}" if rec['pitch'] else "—", 80),
                ]:
                    color = res_color if text in ('PASS', 'FAIL', 'UNMEASURED') else COLOR_TEXT
                    ctk.CTkLabel(
                        row, text=text, font=ctk.CTkFont(size=11),
                        text_color=color, width=w, anchor="w",
                    ).pack(side="left", padx=4, pady=2)

        # Close button
        ctk.CTkButton(
            popup, text="Close", command=popup.destroy,
            fg_color=COLOR_SURFACE_ALT, hover_color=COLOR_BUTTON_HOVER,
            height=40, corner_radius=10,
        ).pack(pady=(5, 15))

    # ==================================================================
    # Stats
    # ==================================================================

    def _update_stats(self):
        """Update the bottom stats bar with today's counts."""
        try:
            from logger import get_stats
            stats = get_stats()
            rate = f"{stats['pass_rate']}%" if stats['total'] > 0 else "—"
            self.stats_label.configure(
                text=(
                    f"Parts today: {stats['total']} | "
                    f"Pass: {stats['pass']} | "
                    f"Fail: {stats['fail']} | "
                    f"Rate: {rate}"
                )
            )
        except Exception as e:
            logger.debug(f"Stats update failed: {e}")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def _on_close(self):
        """Cleanly shut down the application."""
        logger.info("Shutting down ThreadVision GUI...")
        self._camera_running = False

        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass

        self.root.destroy()

    def run(self):
        """Start the GUI main loop."""
        logger.info("ThreadVision GUI starting...")
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Self-test: run `python gui.py` to launch the GUI standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("ThreadVision AI — GUI Launch (Standalone)")
    print("=" * 60)

    app = ThreadVisionGUI()
    app.run()
