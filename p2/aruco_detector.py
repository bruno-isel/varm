"""
ArUco marker detection + camera pose estimation.

Usage:
    python aruco_detector.py

Requires camera_calibration.npz (run calibrate.py first).

Controls:
    q — quit
"""

import sys
import numpy as np
import cv2

CALIBRATION_FILE = "camera_calibration.npz"
MARKER_SIZE = 0.05       # metres — measure your printed marker side
ARUCO_DICT = cv2.aruco.DICT_6X6_250


def load_calibration():
    try:
        data = np.load(CALIBRATION_FILE)
        return data['mtx'], data['dist']
    except FileNotFoundError:
        print(f"Erro: '{CALIBRATION_FILE}' nao encontrado. Corre calibrate.py primeiro.")
        sys.exit(1)


def main():
    mtx, dist = load_calibration()

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a camara.")
        sys.exit(1)

    print("A detetar marcadores ArUco. Prima q para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, mtx, dist
            )

            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE * 0.5)

                dist_m = np.linalg.norm(tvec)
                corner = corners[i][0][0].astype(int)
                cv2.putText(frame, f"ID:{ids[i][0]}  {dist_m:.2f}m",
                            (corner[0], corner[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        n = len(ids) if ids is not None else 0
        cv2.putText(frame, f"Marcadores: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ArUco Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
