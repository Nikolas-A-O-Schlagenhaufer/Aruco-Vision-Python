import cv2
import numpy as np
import cv2.aruco as aruco
import requests
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def center(aruco):
    # calculates the center of the marker, given its aruco
    x = (aruco[0][0] + aruco[1][0] +
         aruco[2][0] + aruco[3][0])/4
    y = (aruco[0][1] + aruco[1][1] +
         aruco[2][1] + aruco[3][1])/4
    return np.array([int(x), int(y)], dtype=np.uint16)


def square_exists(ids):
    return [1] in ids.tolist() and [2] in ids.tolist() and [3] in ids.tolist() and [4] in ids.tolist()


def robot_in_square(ids):
    return [1] in ids.tolist() and [5] in ids.tolist()


def all_arucos(ids):
    return len(ids) == 5


blank_frame = np.ones((480, 640, 3), dtype=np.uint8)
# blank_frame = np.ones((1080, 1920, 3), dtype=np.uint8)
screen_start_center = np.array([0, 0], dtype=np.uint16)
track = []

# Aruco detection
cap = cv2.VideoCapture(0)
# url = "http://192.168.0.63:8080/shot.jpg"
# url = "http://10.67.154.71:8080/shot.jpg"
# url = "http://10.66.119.16:8080/shot.jpg"

while True:
    _, frame = cap.read()

    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # frame = cv2.imdecode(img_arr, -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    # posições dos cantos do marcador, número do marcador
    # ordem dos cantos CE, CD, BD, BE

    try:
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is not None and robot_in_square(ids):
            marker5_corners = corners[ids.tolist().index([5])][0]
            marker0_corners = corners[ids.tolist().index([1])][0]

            marker0_center = np.mean(marker0_corners, axis=0).astype(int)
            marker5_center = np.mean(marker5_corners, axis=0).astype(int)
            track.append(marker5_center)

        if ids is not None and all_arucos(ids):
            marker1_corners = corners[ids.tolist().index([1])][0]
            marker2_corners = corners[ids.tolist().index([2])][0]
            marker3_corners = corners[ids.tolist().index([3])][0]
            marker4_corners = corners[ids.tolist().index([4])][0]
            marker5_corners = corners[ids.tolist().index([5])][0]

            # Calculate the center of each marker
            marker1_center = np.mean(marker1_corners, axis=0).astype(int)
            marker2_center = np.mean(marker2_corners, axis=0).astype(int)
            marker3_center = np.mean(marker3_corners, axis=0).astype(int)
            marker4_center = np.mean(marker4_corners, axis=0).astype(int)
            marker5_center = np.mean(marker5_corners, axis=0).astype(int)

            robot = Point(marker5_center[0], marker5_center[1])

            maze = Polygon([(marker1_center[0], marker1_center[1]), (marker2_center[0], marker2_center[1]),
                           (marker3_center[0], marker3_center[1]), (marker4_center[0], marker4_center[1])])

            if maze.contains(robot):
                # Draw a line connecting the centers of the detected ArUco markers
                cv2.line(frame, tuple(marker1_center),
                         tuple(marker2_center), (0, 255, 0), 2)
                cv2.line(frame, tuple(marker2_center),
                         tuple(marker3_center), (0, 255, 0), 2)
                cv2.line(frame, tuple(marker3_center),
                         tuple(marker4_center), (0, 255, 0), 2)
                cv2.line(frame, tuple(marker4_center),
                         tuple(marker1_center), (0, 255, 0), 2)
            else:
                cv2.line(frame, tuple(marker1_center),
                         tuple(marker2_center), (0, 0, 255), 2)
                cv2.line(frame, tuple(marker2_center),
                         tuple(marker3_center), (0, 0, 255), 2)
                cv2.line(frame, tuple(marker3_center),
                         tuple(marker4_center), (0, 0, 255), 2)
                cv2.line(frame, tuple(marker4_center),
                         tuple(marker1_center), (0, 0, 255), 2)

    except Exception as e:
        pass

    # Draw the centers of marker 5 in the frame
    div = 256 / len(track) if track else 255
    # for point in track:
    for i in range(len(track)):
        cv2.circle(frame, tuple(track[i]), 5,
                   (int(256-i*div), 0, int(div*i)), -1)

    cv2.imshow('Display', frame)
    # cv2.imshow('Tracking', blank_frame)

    # print(frame.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save blank frame to img
files = os.listdir('trackings')
if len(files) == 0:
    value = 0
else:
    last_file = files[-1]
    value = int(last_file.split('-')[1].split('.')[0]) + 1
cv2.imwrite(f"trackings/tracking-{str(value).zfill(5)}.png", frame)
