import cv2
import numpy as np
import logging

logger = logging.getLogger("ThreadVision.silhouette")

def generate_silhouette(image, min_bg_median=128):
    """
    Converts a raw photo into a high-contrast silhouette.
    - Smooths out background texture using Bilateral Filtering.
    - Uses Otsu thresholding to segment the bolt and ArUco marker.
    - Composes a final image where background is pure gray and 
      foreground is pure black.
    
    Args:
        image: BGR numpy array
        min_bg_median: The minimum gray value to assign to the background.
    
    Returns:
        silhouette_image: BGR numpy array
    """
    H, W = image.shape[:2]
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Extract background color safely by sampling rims of the image
    # (assuming the bolt is mostly central, the absolute border pixels are bg)
    rim_pixels = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg_median = float(np.median(rim_pixels))
    
    # Optional contrast adjustment (CLAHE) to help Otsu if lighting is uneven
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Bilateral filter to smooth flat textures but preserve sharp threads
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 4. Otsu threshold to separate foreground (dark) from background (light)
    ots_val, binary_mask = cv2.threshold(
        smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    logger.info(f"Silhouette: Otsu threshold determined as {ots_val}")
    
    # 5. Clean up the mask using morphology to remove tiny noise islands and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. Compose the new image
    # We want a pure gray background and pure black foreground.
    # We can output a BGR image to remain compatible with the rest of the pipeline.
    silhouette = np.full((H, W, 3), max(bg_median, min_bg_median), dtype=np.uint8)
    
    # Set foreground (where mask > 0) to black [0, 0, 0]
    silhouette[binary_mask > 0] = [0, 0, 0]
    
    return silhouette

if __name__ == "__main__":
    # Test script for easy verification
    import sys
    import os
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            sil = generate_silhouette(img)
            name, ext = os.path.splitext(img_path)
            cv2.imwrite(f"{name}_silhouetted{ext}", sil)
            print(f"Saved {name}_silhouetted{ext}")
