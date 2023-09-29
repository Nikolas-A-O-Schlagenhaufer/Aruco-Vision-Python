import cv2
import numpy as np
import cv2.aruco as aruco


def center(corners):
    # calculates the center of the marker, given its corners
    x = (corners[0][0] + corners[1][0] +
         corners[2][0] + corners[3][0])/4
    y = (corners[0][1] + corners[1][1] +
         corners[2][1] + corners[3][1])/4
    return (x, y)


blank_frame = np.ones((480, 640, 3), dtype=np.uint8)

# Aruco detection
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    # posições dos cantos do marcador, número do marcador
    # ordem dos cantos CE, CD, BD, BE

    if np.all(ids):
        image = aruco.drawDetectedMarkers(frame, corners, ids)
        marker_center = center(corners[0][0])
        blank_frame[int(marker_center[1]), int(
            marker_center[0])] = [255, 255, 255]
        cv2.imshow('Blank', blank_frame)
        cv2.imshow('Display', image)
    else:
        display = frame
        cv2.imshow('Blank', blank_frame)
        cv2.imshow('Display', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
