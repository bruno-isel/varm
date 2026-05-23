"""
Marker-Based AR — virtual objects registered per ArUco marker ID.

Usage:
    python main.py

Requires camera_calibration.npz (run calibrate.py first).

Each marker ID renders a different 3D wireframe object:
    ID 0 — cube
    ID 1 — pyramid
    ID 2 — coordinate axes only (default for any other ID)

Controls:
    q — quit
"""

import sys
import numpy as np
import cv2

CALIBRATION_FILE = "camera_calibration.npz"
MARKER_SIZE = 0.05        # metres — measure your printed marker side
ARUCO_DICT = cv2.aruco.DICT_6X6_250


def load_calibration():
    try:
        data = np.load(CALIBRATION_FILE)
        return data['mtx'], data['dist']
    except FileNotFoundError:
        print(f"Erro: '{CALIBRATION_FILE}' nao encontrado. Corre calibrate.py primeiro.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 3D object definitions — vertices in marker coordinate space (metres)
# Origin = marker centre, Z points up from marker surface
# ---------------------------------------------------------------------------

def _cube_edges(s=0.04):
    h = s
    v = np.float32([
        [-s/2, -s/2, 0], [s/2, -s/2, 0], [s/2, s/2, 0], [-s/2, s/2, 0],
        [-s/2, -s/2, h], [s/2, -s/2, h], [s/2, s/2, h], [-s/2, s/2, h],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return v, edges


def _pyramid_edges(s=0.04):
    h = s * 1.5
    v = np.float32([
        [-s/2, -s/2, 0], [s/2, -s/2, 0], [s/2, s/2, 0], [-s/2, s/2, 0],
        [0, 0, h],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (0,4),(1,4),(2,4),(3,4),
    ]
    return v, edges


OBJECTS = {
    0: _cube_edges,
    1: _pyramid_edges,
}

COLORS = {
    0: (0, 200, 255),
    1: (0, 255, 100),
}
DEFAULT_COLOR = (200, 100, 255)


def draw_object(frame, marker_id, rvec, tvec, mtx, dist):
    build_fn = OBJECTS.get(marker_id)
    color = COLORS.get(marker_id, DEFAULT_COLOR)

    if build_fn is None:
        cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE * 0.5)
        return

    vertices, edges = build_fn()
    projected, _ = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
    projected = projected.reshape(-1, 2).astype(int)

    for a, b in edges:
        cv2.line(frame, tuple(projected[a]), tuple(projected[b]), color, 2)

    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE * 0.3)


def main():
    mtx, dist = load_calibration()

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a camara.")
        sys.exit(1)

    print("AR ativo. Aponta a camara para os marcadores. Prima q para sair.")
    print("  ID 0 -> cubo  |  ID 1 -> piramide  |  outro ID -> eixos")

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
                marker_id = int(ids[i][0])
                draw_object(frame, marker_id, rvec, tvec, mtx, dist)

                dist_m = np.linalg.norm(tvec)
                corner = corners[i][0][0].astype(int)
                cv2.putText(frame, f"ID:{marker_id}  {dist_m:.2f}m",
                            (corner[0], corner[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Marker AR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
