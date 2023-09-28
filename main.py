import cv2
import numpy as np
import cv2.aruco as aruco

# Aruco detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    print(ids)
    if np.all(ids):
        image = aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Display', image)
    else:
        display = frame
        cv2.imshow('Display', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
