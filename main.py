import cv2
import numpy as np
import cv2.aruco as aruco
import requests
import os


def center(aruco):
    # calculates the center of the marker, given its aruco
    x = (aruco[0][0] + aruco[1][0] +
         aruco[2][0] + aruco[3][0])/4
    y = (aruco[0][1] + aruco[1][1] +
         aruco[2][1] + aruco[3][1])/4
    return np.array([int(x), int(y)], dtype=np.uint16)


# blank_frame = np.ones((480, 640, 3), dtype=np.uint8)
blank_frame = np.ones((1080, 1920, 3), dtype=np.uint8)
screen_start_center = np.array([0,0], dtype=np.uint16)
track = []

# Aruco detection
# cap = cv2.VideoCapture(0)
url = "http://10.79.225.241:8080/shot.jpg"

while True:
    # _, frame = cap.read()

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    frame = cv2.imdecode(img_arr, -1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    # posições dos cantos do marcador, número do marcador
    # ordem dos cantos CE, CD, BD, BE

    if ids is not None and len(ids) >= 2:
        for i in range(4):
        # Extract the corners of the detected markers
            try:
                marker1_corners = corners[ids.tolist().index([i+1])][0]
                if i == len(ids) - 2:
                    marker2_corners = corners[ids.tolist().index([1])][0]
                else:
                    marker2_corners = corners[ids.tolist().index([i+2])][0]
                marker5_corners = corners[ids.tolist().index([5])][0]
                marker0_corners = corners[ids.tolist().index([1])][0]

                # Calculate the center of each marker
                marker1_center = np.mean(marker1_corners, axis=0).astype(int)
                marker2_center = np.mean(marker2_corners, axis=0).astype(int)
                marker5_center = np.mean(marker5_corners, axis=0).astype(int)
                track.append(marker5_center)
                marker0_center = np.mean(marker0_corners, axis=0).astype(int)

                # Calculate the distance between the centers of the two markers
                distance = np.linalg.norm(marker5_center - marker0_center)

                # Draw a line connecting the centers of the detected ArUco markers
                # cv2.line(frame, tuple(marker1_center), tuple(marker2_center), (0, 255, 0), 2)
                # cv2.line(frame, tuple(marker5_center), tuple(marker0_center), (255, 0, 0), 2)

                # Display the distance between the markers
                # cv2.putText(frame, f"Distance: {distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw the centers of marker 5 in the frame
                for point in track:
                    cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            except:
                pass

        


    cv2.imshow('Display', frame)
    # cv2.imshow('Tracking', blank_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # time.sleep(0.5)

    

# cap.release()
cv2.destroyAllWindows()

# save blank frame to img
files = os.listdir('trackings')
if len(files) == 0:
    value = 0
else:
    last_file = files[-1]
    value = int(last_file.split('-')[1].split('.')[0]) + 1
cv2.imwrite(f"trackings/tracking-{value}.png", frame)