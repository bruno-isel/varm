"""
ArUco marker generator.

Usage:
    python generate_markers.py

Generates marker images for IDs 0 and 1 (DICT_6X6_250) into the markers/ folder.
Uses cv2.aruco.generateImageMarker() — equivalent to drawMarker() in older OpenCV versions.
"""

import os
import cv2

ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_IDS = [0, 1]
MARKER_SIZE_PX = 300   # pixels (add white border when printing)
OUTPUT_DIR = "markers"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    for marker_id in MARKER_IDS:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE_PX)
        path = os.path.join(OUTPUT_DIR, f"marker_{marker_id}.png")
        cv2.imwrite(path, img)
        print(f"Gerado: {path}")

    print(f"\n{len(MARKER_IDS)} marcadores guardados em '{OUTPUT_DIR}/'.")
    print("Imprime com margem branca de pelo menos 1-2 cm em todos os lados.")


if __name__ == "__main__":
    main()
