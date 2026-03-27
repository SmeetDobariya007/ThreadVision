"""Debug: verify edge-scan prompt on 21.1.jpeg with v6."""
import cv2, numpy as np, sys, logging
logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, '.')

from analyze_thread_image import load_image, normalize_orientation, detect_aruco_scale
from sam_segmentor import SAMSegmentor

img = load_image("test_image/21.1.jpeg")
img_norm, was_rotated = normalize_orientation(img)
H, W = img_norm.shape[:2]
print(f"After normalize: {W}x{H}, rotated={was_rotated}")

mm_per_pixel, conf, corner_pts, aruco_cs = detect_aruco_scale(img_norm, 21.0)
p2mm = 1.0 / mm_per_pixel
print(f"p2mm={p2mm:.2f}")

acx = int(np.mean(corner_pts[:, 0]))
acy = int(np.mean(corner_pts[:, 1]))
print(f"ArUco center: ({acx}, {acy})")

seg = SAMSegmentor.__new__(SAMSegmentor)
seg._predictor = None
prompt = seg._find_bolt_by_edges(img_norm, corner_pts)
print(f"\n>>> PROMPT POINT: {prompt}")

vis = img_norm.copy()
cv2.circle(vis, (acx, acy), 20, (255, 0, 0), -1)
cv2.circle(vis, prompt, 25, (0, 255, 0), -1)
cv2.putText(vis, f"PROMPT {prompt}", (prompt[0]+30, prompt[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.putText(vis, f"ARUCO ({acx},{acy})", (acx+30, acy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

ax1 = int(corner_pts[:, 0].min())
ay1 = int(corner_pts[:, 1].min())
ax2 = int(corner_pts[:, 0].max())
ay2 = int(corner_pts[:, 1].max())
mw = ax2-ax1; mh = ay2-ay1
pad = int(max(mw,mh)*0.15)
cv2.rectangle(vis, (max(0,ax1-pad), max(0,ay1-pad)),
              (min(W,ax2+pad), min(H,ay2+pad)), (0,0,255), 3)

cv2.imwrite("test_image/debug_prompt.jpg", vis)
print("Saved debug_prompt.jpg")
