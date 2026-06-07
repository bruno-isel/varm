"""
Camera calibration using a chessboard pattern.

Usage:
    python calibrate.py            # capture mode (webcam)
    python calibrate.py --check    # verify saved calibration

Controls (capture mode):
    SPACE  — capture current frame if chessboard is detected
    c      — run calibration with captured frames (min 10)
    q      — quit
"""

import sys
import numpy as np
import cv2

CHESSBOARD = (9, 6)       # inner corners (cols, rows)
SQUARE_SIZE = 2.5         # cm — measure your printed square size
OUTPUT_FILE = "camera_calibration.npz"
MIN_FRAMES = 10

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points of chessboard corners in world space (Z=0 plane)
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


def capture_mode():
    obj_points = []
    img_points = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a camara.")
        sys.exit(1)

    print(f"Aponta a camara para o tabuleiro de xadrez ({CHESSBOARD[0]}x{CHESSBOARD[1]} cantos internos).")
    print("SPACE para capturar | c para calibrar | q para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD, corners, found)
            cv2.putText(display, "Detetado! SPACE para capturar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Tabuleiro nao detetado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Capturas: {len(obj_points)}/{MIN_FRAMES}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Calibracao", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' ') and found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"Frame {len(obj_points)} capturada.")

        elif key == ord('c'):
            if len(obj_points) < MIN_FRAMES:
                print(f"Precisas de pelo menos {MIN_FRAMES} capturas (tens {len(obj_points)}).")
            else:
                _run_calibration(obj_points, img_points, gray.shape[::-1])
                break

    cap.release()
    cv2.destroyAllWindows()


def _run_calibration(obj_points, img_points, img_size):
    print(f"\nA calibrar com {len(obj_points)} frames...")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    h, w = img_size[1], img_size[0]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    print(f"\nErro de reprojecao (RMS): {rms:.4f} px")
    print(f"Matriz intrínseca K:\n{mtx}")
    print(f"Coeficientes de distorcao: {dist.ravel()}")
    print(f"  fx={mtx[0,0]:.2f}  fy={mtx[1,1]:.2f}  cx={mtx[0,2]:.2f}  cy={mtx[1,2]:.2f}")

    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, new_mtx=new_mtx, roi=roi)
    print(f"\nCalibracao guardada em '{OUTPUT_FILE}'.")


def check_mode():
    try:
        data = np.load(OUTPUT_FILE)
    except FileNotFoundError:
        print(f"Ficheiro '{OUTPUT_FILE}' nao encontrado. Corre primeiro sem --check.")
        sys.exit(1)

    mtx = data['mtx']
    dist = data['dist']
    print("Calibracao carregada:")
    print(f"  fx={mtx[0,0]:.4f}  fy={mtx[1,1]:.4f}  cx={mtx[0,2]:.4f}  cy={mtx[1,2]:.4f}")
    print(f"  dist={dist.ravel()}")

    cap = cv2.VideoCapture(0)
    print("\nA mostrar feed sem distorcao. Prima q para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
        x, y, rw, rh = roi
        undistorted = undistorted[y:y+rh, x:x+rw]
        cv2.imshow("Original", frame)
        cv2.imshow("Sem distorcao", undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_mode()
    else:
        capture_mode()
