# sam_segmentor.py — MobileSAM bolt segmentation module
# =======================================================
# Replaces _find_bolt_contour with a pixel-accurate SAM mask.
# Imported optionally by analyze_thread_image.py — if import
# fails, the system falls back to Canny contour detection.
#
# Usage:
#   from sam_segmentor import SAMSegmentor
#   seg = SAMSegmentor("mobile_sam.pt")          # load once at startup
#   bbox, mask, cs = seg.segment_bolt(image, p2mm)   # call per frame

import numpy as np
import cv2
import os
import logging

logger = logging.getLogger("ThreadVision.sam")

# ── Try importing MobileSAM ───────────────────────────────────────
SAM_AVAILABLE = False
try:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("MobileSAM not installed — SAM segmentation unavailable.")

class SAMSegmentationError(Exception):
    pass

class SAMSegmentor:
    """
    Wraps MobileSAM for single-point bolt segmentation.

    Load once at application startup with SAMSegmentor(weights_path).
    Call segment_bolt(image, p2mm) per frame — returns same format
    as _find_bolt_contour so it drops in as a direct replacement.
    """
    def __init__(self, weights_path="mobile_sam.pt"):
        if not SAM_AVAILABLE:
            raise SAMSegmentationError(
                "MobileSAM not installed. "
                "Run: pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )
        if not os.path.exists(weights_path):
            raise SAMSegmentationError(
                f"Model weights not found: {weights_path}\n"
                "Download from: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        logger.info(f"Loading MobileSAM from {weights_path}...")
        self._device = "cpu"   # target device Pi 4
        model = sam_model_registry["vit_t"](checkpoint=weights_path)
        model.to(self._device)
        model.eval()
        self._predictor = SamPredictor(model)
        logger.info("MobileSAM loaded.")

    def segment_bolt(self, image, p2mm, prompt_point=None, fallback_fn=None):
        """
        Segment the bolt in the image and return bbox + contour.
        """
        H, W = image.shape[:2]

        if prompt_point is None:
            prompt_point = (W // 2, H // 2)

        # Encode image (MobileSAM internally converts BGR→RGB if needed, but it's safe to feed RGB)
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        self._predictor.set_image(image_rgb)

        point_coords = np.array([[prompt_point[0], prompt_point[1]]])
        point_labels = np.array([1])

        try:
            masks, scores, logits = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except Exception as e:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(f"Prediction inference failed: {e}")

        best_idx  = int(np.argmax(scores))
        mask      = masks[best_idx].astype(np.uint8) * 255
        conf      = float(scores[best_idx])

        mask_px    = np.count_nonzero(mask)
        frame_px   = H * W
        coverage   = mask_px / frame_px

        if coverage < 0.005 or coverage > 0.85:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(
                f"SAM mask invalid (coverage={coverage:.1%}). Ensure bolt is centred in frame."
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError("No contours in SAM mask.")

        best_contour = max(contours, key=cv2.contourArea)
        x, y, w, h   = cv2.boundingRect(best_contour)

        width_mm = w / p2mm
        if not (4.0 <= width_mm <= 50.0):
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(
                f"SAM segmented object is {width_mm:.1f}mm wide — not a valid bolt size."
            )

        contour_cs = self.get_confidence_signals(mask, scores)
        contour_cs['aspect_ratio'] = float(h / w) if w > 0 else 0.0

        return (best_contour, x, y, w, h), mask, contour_cs

    def get_confidence_signals(self, mask, scores):
        """
        Return confidence signals for the confidence scoring layer.
        """
        best_score = float(np.max(scores))
        coverage   = float(np.count_nonzero(mask)) / mask.size
        return {
            'sam_score':    best_score,
            'coverage':     coverage,
            'bbox_fill':    best_score,  # Proxy mapping per spec
        }
