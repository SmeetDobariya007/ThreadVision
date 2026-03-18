import cv2
import numpy as np
from cv2 import aruco

def generate_marker():
    # Use EXACTLY the dictionary our script looks for
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Generate marker ID 0, 500x500 pixels
    # The aruco.generateImageMarker function creates just the black pattern
    marker_size_px = 500
    marker_image = aruco.generateImageMarker(aruco_dict, 0, marker_size_px)
    
    # CRITICAL: We MUST add a thick white border. OpenCV cannot detect it without this!
    border_size = 100
    final_image = cv2.copyMakeBorder(
        marker_image, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )
    
    cv2.imwrite("PERFECT_ARUCO_MARKER_TO_PRINT.png", final_image)
    print("Successfully generated PERFECT_ARUCO_MARKER_TO_PRINT.png")

if __name__ == "__main__":
    generate_marker()
