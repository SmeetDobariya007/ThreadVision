import cv2
import numpy as np
from cv2 import aruco

# Use the exact same dictionary your analyze_thread_image.py uses!
ARUCO_DICT = aruco.DICT_4X4_50

def run_webcam_test():
    print("Opening webcam... (Press 'Q' to quit)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers!
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # If successfully detected
        if ids is not None:
            # Draw the green outline around it
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Print the side lengths explicitly calculating exactly like analyze_thread_image.py
            corner = corners[0][0] # shape (4,2)
            side_0 = np.linalg.norm(corner[0] - corner[1])
            side_1 = np.linalg.norm(corner[1] - corner[2])
            side_2 = np.linalg.norm(corner[2] - corner[3])
            side_3 = np.linalg.norm(corner[3] - corner[0])
            
            sides = [side_0, side_1, side_2, side_3]
            marker_size_px = np.mean(sides)
            
            cv2.putText(frame, f"ArUco ID: {ids[0][0]} Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {marker_size_px:.1f} pixels", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Show me a 4x4_50 ArUco Marker!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('ArUco Real-Time Tester', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_test()
