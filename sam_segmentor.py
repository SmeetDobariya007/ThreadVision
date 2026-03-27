# sam_segmentor.py — MobileSAM bolt segmentation (v6 — FINAL)
# =============================================================
# After 5 iterations, the root cause was identified:
#   - ArUco marker = massive dark region that dominates contour search
#   - Bolt + V-block shadow merge into one giant blob
#   - Image border darkness creates false positives far from ArUco
#
# SOLUTION v6: DON'T search for contours at all.
# Instead, use a COLUMN PROJECTION scan:
#   1. Create edge map (Canny).
#   2. Mask out ArUco region.
#   3. For each column, count edge pixels → edge density profile.
#   4. The bolt's threaded region creates a BAND of HIGH edge density.
#   5. The prompt point = center of the densest non-ArUco band.
# This works because thread teeth create far more edges per column
# than smooth surfaces (background, V-block, marker).

import numpy as np
import cv2
import os
import logging

logger = logging.getLogger("ThreadVision.sam")

SAM_AVAILABLE = False
try:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logger.warning("MobileSAM not installed.")


class SAMSegmentationError(Exception):
    pass


class SAMSegmentor:

    def __init__(self, weights_path="mobile_sam.pt"):
        if not SAM_AVAILABLE:
            raise SAMSegmentationError(
                "MobileSAM not installed. "
                "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )
        if not os.path.exists(weights_path):
            raise SAMSegmentationError(f"Weights not found: {weights_path}")
        logger.info("Loading MobileSAM weights...")
        model = sam_model_registry["vit_t"](checkpoint=weights_path)
        model.to("cpu")
        model.eval()
        self._predictor = SamPredictor(model)
        logger.info("MobileSAM ready.")

    # ── Public entry point ────────────────────────────────────────

    def segment_bolt(self, image, p2mm, aruco_corners=None,
                     manual_point=None, fallback_fn=None):
        H, W = image.shape[:2]

        if manual_point is not None:
            prompt_pt = manual_point
            logger.info(f"SAM: manual prompt {prompt_pt}")
        elif aruco_corners is not None and len(aruco_corners) == 4:
            prompt_pt = self._find_bolt_by_edges(image, aruco_corners)
            logger.info(f"SAM: edge-scan prompt {prompt_pt}")
        else:
            prompt_pt = self._find_bolt_by_edges(image, None)
            logger.info(f"SAM: auto prompt {prompt_pt}")

        # Encode image
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)

        # Prompts: foreground on bolt + negative on ArUco
        coords = [[prompt_pt[0], prompt_pt[1]]]
        labels = [1]
        if aruco_corners is not None and len(aruco_corners) == 4 \
                and manual_point is None:
            acx = int(np.mean(aruco_corners[:, 0]))
            acy = int(np.mean(aruco_corners[:, 1]))
            coords.append([acx, acy])
            labels.append(0)
            logger.info(f"SAM: negative prompt at ArUco ({acx},{acy})")

        # ── BACKGROUND SUPPRESSION: Negative border prompts ──
        # Objects like V-blocks create dark shadows that connect to the bolt.
        # Since these objects touch the image edges, we add negative prompts
        # at the borders to force SAM to exclude them.
        H, W = image.shape[:2]
        bx, by = int(W * 0.05), int(H * 0.05)
        border_pts = [
            [bx, H // 3], [bx, 2 * H // 3],              # Left edge
            [W - bx, H // 3], [W - bx, 2 * H // 3],      # Right edge
            [W // 2, by],                                # Top edge
            [W // 2, H - by]                             # Bottom edge
        ]
        
        border_negatives = 0
        for pt in border_pts:
            # Don't place a negative prompt too close to the bolt prompt
            dist = np.hypot(pt[0] - prompt_pt[0], pt[1] - prompt_pt[1])
            if dist > max(W, H) * 0.15:
                # Also don't place one too close to ArUco text we already prompted
                if aruco_corners is not None and len(aruco_corners) == 4:
                    ac_dist = np.hypot(pt[0] - acx, pt[1] - acy)
                    if ac_dist < max(W, H) * 0.05:
                        continue
                        
                coords.append(pt)
                labels.append(0)
                border_negatives += 1
                
        logger.info(f"SAM: added {border_negatives} border negative prompts")

        try:
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(coords),
                point_labels=np.array(labels),
                multimask_output=True,
            )
        except Exception as e:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(f"Predict failed: {e}")

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255
        score = float(scores[best_idx])
        coverage = np.count_nonzero(mask) / (H * W)
        logger.info(f"SAM: score={score:.3f}, coverage={coverage:.1%}")

        if coverage < 0.003 or coverage > 0.80:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(
                f"SAM mask coverage={coverage:.1%} out of range."
            )

        # Hard-erase ArUco
        if aruco_corners is not None and len(aruco_corners) == 4:
            mask = self._erase_aruco_from_mask(mask, aruco_corners)

        # ── MASK REFINEMENT: keep only the bolt body, drop shadow ──
        # Within the SAM mask, separate bolt (dark metal) from shadow
        # (lighter). Otsu threshold on the masked gray pixels.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked_gray = gray.copy()
        masked_gray[mask == 0] = 255  # set non-mask pixels to white
        
        # Otsu on the masked area to find dark-vs-light cutoff
        mask_pixels = gray[mask > 0]
        if len(mask_pixels) > 100:
            otsu_val, _ = cv2.threshold(
                mask_pixels, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Keep only pixels darker than the Otsu threshold
            refined = np.zeros_like(mask)
            refined[(mask > 0) & (gray < otsu_val)] = 255
            
            # Morphological close to fill small gaps in bolt body
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            
            refined_coverage = np.count_nonzero(refined) / (H * W)
            logger.info(
                f"SAM: mask refined by Otsu ({otsu_val:.0f}), "
                f"coverage {coverage:.1%} → {refined_coverage:.1%}"
            )
            
            # Only use refined mask if it's reasonable (not too small)
            if refined_coverage > 0.003:
                mask = refined
            else:
                logger.warning("Refined mask too small — keeping original SAM mask")

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError("No contours after ArUco erase.")

        # Select the contour that contains the prompt point (= guaranteed on bolt).
        # Fall back to closest contour by distance if none contains it exactly.
        best_c = None
        min_dist = float('inf')
        for cnt in contours:
            dist = cv2.pointPolygonTest(cnt, (float(prompt_pt[0]), float(prompt_pt[1])), True)
            if dist >= 0:  # point is inside or on edge
                if best_c is None or cv2.contourArea(cnt) > cv2.contourArea(best_c):
                    best_c = cnt
            elif abs(dist) < min_dist:
                min_dist = abs(dist)
                closest_c = cnt
        
        if best_c is None:
            # No contour contains prompt; use closest, or largest
            best_c = closest_c if min_dist < float('inf') else max(contours, key=cv2.contourArea)
            logger.warning(f"No contour contains prompt {prompt_pt}, using closest (dist={min_dist:.0f}px)")

        x, y, w, h = cv2.boundingRect(best_c)
        width_mm = w / p2mm

        if not (3.0 <= width_mm <= 60.0):
            if fallback_fn:
                return fallback_fn()
            raise SAMSegmentationError(
                f"SAM object {width_mm:.1f}mm — not a bolt."
            )

        logger.info(f"SAM: bbox ({x},{y},{w},{h}) = {width_mm:.1f}mm × {h/p2mm:.1f}mm")
        cs = self.get_confidence_signals(mask, scores)
        cs['aspect_ratio'] = float(h / w) if w > 0 else 0.0
        return (best_c, x, y, w, h), mask, cs

    def get_confidence_signals(self, mask, scores):
        return {
            'sam_score': float(np.max(scores)),
            'coverage': float(np.count_nonzero(mask)) / mask.size,
            'bbox_fill': float(np.max(scores)),
        }

    # ── Bolt finder using edge-transition scanning ───────────────

    def _find_bolt_by_edges(self, image, aruco_corners):
        """
        Find the bolt using edge-TRANSITION counting per row/column.
        
        Thread teeth create many edge transitions (bright→dark→bright
        zigzag). Smooth surfaces (background, shadows, marker borders)
        create few transitions per row.
        
        This is fully generalized: no hardcoded bolt sizes. All 
        parameters scale proportionally to image dimensions.
        """
        H, W = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 40, 120)

        # ArUco exclusion zone
        ex1 = ey1 = ex2 = ey2 = 0
        if aruco_corners is not None:
            ax1 = int(aruco_corners[:, 0].min())
            ay1 = int(aruco_corners[:, 1].min())
            ax2 = int(aruco_corners[:, 0].max())
            ay2 = int(aruco_corners[:, 1].max())
            mw = ax2 - ax1
            mh = ay2 - ay1
            pad = int(max(mw, mh) * 0.15)
            ex1 = max(0, ax1 - pad)
            ey1 = max(0, ay1 - pad)
            ex2 = min(W, ax2 + pad)
            ey2 = min(H, ay2 + pad)
            edges[ey1:ey2, ex1:ex2] = 0

        # Mask image borders (3%)
        bx = int(W * 0.03)
        by = int(H * 0.03)
        edges[:by, :] = 0
        edges[H-by:, :] = 0
        edges[:, :bx] = 0
        edges[:, W-bx:] = 0

        # Count edge TRANSITIONS per row (not just edge pixels).
        # A transition = pixel changes from 0→255 or 255→0.
        # Thread teeth create 10-30+ transitions per row.
        # Smooth borders create 2-4 transitions per row.
        edge_binary = (edges > 0).astype(np.uint8)
        row_transitions = np.sum(np.abs(np.diff(edge_binary, axis=1)), axis=1).astype(float)
        col_transitions = np.sum(np.abs(np.diff(edge_binary, axis=0)), axis=0).astype(float)

        # Smooth profiles
        from scipy.ndimage import uniform_filter1d
        row_smooth = uniform_filter1d(row_transitions, size=max(1, H // 40))

        # Sliding window on ROWS to find the thread-teeth band
        # Use a proportional window: 1/12 of height covers most bolt diameters
        win_r = max(1, H // 12)

        if len(row_smooth) > win_r:
            row_cum = np.cumsum(row_smooth)
            row_sums = row_cum[win_r:] - np.concatenate([[0], row_cum[:-win_r-1]])
            best_r_start = int(np.argmax(row_sums))
        else:
            best_r_start = 0

        r1 = max(0, best_r_start)
        r2 = min(H, best_r_start + win_r)
        best_row = (r1 + r2) // 2

        # Within the detected row band, find the COLUMN with highest
        # transition count — this is the bolt shaft, not the background.
        band_transitions = np.sum(
            np.abs(np.diff(edge_binary[r1:r2, :], axis=0)), axis=0
        ).astype(float)
        
        # Smooth and find peak column
        col_smooth_band = uniform_filter1d(band_transitions, size=max(1, W // 30))
        best_col = int(np.argmax(col_smooth_band))

        # Snap to the darkest pixel near (best_col, best_row)
        # Search a small patch around the intersection
        patch_r = win_r // 2
        patch_c = max(1, W // 20)
        pr1 = max(0, best_row - patch_r)
        pr2 = min(H, best_row + patch_r)
        pc1 = max(0, best_col - patch_c)
        pc2 = min(W, best_col + patch_c)
        
        patch = gray[pr1:pr2, pc1:pc2]
        if patch.size > 0:
            min_loc = np.unravel_index(np.argmin(patch), patch.shape)
            prompt = (int(pc1 + min_loc[1]), int(pr1 + min_loc[0]))
        else:
            prompt = (int(best_col), int(best_row))

        logger.info(
            f"Edge-transition scan: band row={best_r_start}-{best_r_start+win_r}, "
            f"best_col={best_col}, prompt={prompt}"
        )

        # Sanity: if prompt is inside ArUco zone, shift to opposite side
        if aruco_corners is not None:
            if ex1 <= prompt[0] <= ex2 and ey1 <= prompt[1] <= ey2:
                logger.warning("Prompt in ArUco zone — shifting")
                acx = (ax1 + ax2) // 2
                acy = (ay1 + ay2) // 2
                prompt = self._opposite_center(H, W, acx, acy)

        return prompt

    def _opposite_center(self, H, W, aruco_cx, aruco_cy):
        """Center of the image half opposite to ArUco."""
        px = W // 4 if aruco_cx > W // 2 else 3 * W // 4
        py = H // 4 if aruco_cy > H // 2 else 3 * H // 4
        logger.warning(f"SAM: opposite-center → ({px},{py})")
        return (px, py)

    def _erase_aruco_from_mask(self, mask, aruco_corners):
        """Hard-erase the ArUco region from the output mask."""
        H, W = mask.shape[:2]
        ax1 = int(aruco_corners[:, 0].min())
        ay1 = int(aruco_corners[:, 1].min())
        ax2 = int(aruco_corners[:, 0].max())
        ay2 = int(aruco_corners[:, 1].max())
        mw = ax2 - ax1
        mh = ay2 - ay1
        pad = int(max(mw, mh) * 0.15)
        ex1 = max(0, ax1 - pad)
        ey1 = max(0, ay1 - pad)
        ex2 = min(W, ax2 + pad)
        ey2 = min(H, ay2 + pad)
        mask[ey1:ey2, ex1:ex2] = 0
        return mask
