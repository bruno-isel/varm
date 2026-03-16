import os
import cv2
import numpy as np


class FaceNormalizer:
    """
    Normalizes face images according to MPEG-7 specification:
    - Size: 56 rows x 46 columns
    - Eyes aligned horizontally at row 24
    - Left eye at column 16, right eye at column 31
    - Grayscale, 8-bit (256 levels)
    """

    # MPEG-7 target dimensions and eye positions
    TARGET_W = 46
    TARGET_H = 56
    LEFT_EYE_TARGET = np.array([16, 24], dtype=np.float32)   # (x, y)
    RIGHT_EYE_TARGET = np.array([31, 24], dtype=np.float32)  # (x, y)

    def __init__(self):
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        )
        # Multiple eye cascades for robustness (glasses, etc.)
        self.eye_cascades = [
            cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml')),
            cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye_tree_eyeglasses.xml')),
        ]

    def _try_detect_eyes(self, face_roi):
        """
        Try multiple cascades and parameter sets to detect exactly 2 eyes
        in the upper half of the face ROI.
        Returns list of 2 eye rects sorted by x, or None.
        """
        h = face_roi.shape[0]
        # Only search in the upper 60% of the face (eyes are never in the lower half)
        upper_face = face_roi[0:int(h * 0.6), :]

        # Try each cascade with progressively relaxed parameters
        param_sets = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (20, 20)},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (15, 15)},
            {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (10, 10)},
        ]

        for cascade in self.eye_cascades:
            for params in param_sets:
                eyes = cascade.detectMultiScale(upper_face, **params)
                if len(eyes) >= 2:
                    # Sort by x and take the two most separated (avoid double-detections)
                    eyes = sorted(eyes, key=lambda e: e[0])
                    # Pick the leftmost and rightmost
                    left = eyes[0]
                    right = eyes[-1]
                    # Sanity check: eyes should be horizontally separated
                    left_cx = left[0] + left[2] // 2
                    right_cx = right[0] + right[2] // 2
                    if right_cx - left_cx > face_roi.shape[1] * 0.15:
                        return [left, right]

        return None

    def _detect_eyes(self, image):
        """
        Detects face and both eyes in the image.
        Returns (left_eye_center, right_eye_center) in image coordinates, or (None, None).
        """
        # Apply histogram equalization to improve detection in dark/bright images
        equalized = cv2.equalizeHist(image)

        # Try detection on both original and equalized
        for img in [image, equalized]:
            faces = self.face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                continue

            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Extract face ROI (no extra padding - keep it tight for eye search)
            face_roi = img[y:y + h, x:x + w]

            eyes = self._try_detect_eyes(face_roi)
            if eyes is None:
                continue

            left_eye, right_eye = eyes

            # Eye centers in original image coordinates
            left_center = np.array([
                x + left_eye[0] + left_eye[2] // 2,
                y + left_eye[1] + left_eye[3] // 2
            ], dtype=np.float32)

            right_center = np.array([
                x + right_eye[0] + right_eye[2] // 2,
                y + right_eye[1] + right_eye[3] // 2
            ], dtype=np.float32)

            return left_center, right_center

        return None, None

    def normalize(self, image):
        """
        Normalizes a single face image according to MPEG-7 spec.
        Uses a similarity transform to map detected eye positions
        directly to the target MPEG-7 eye positions.
        Returns the normalized 56x46 grayscale image, or None if detection fails.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try the image as-is, and also rotated (for sideways photos)
        rotations = [
            None,                      # original
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]

        for rotation in rotations:
            if rotation is None:
                img = image
            else:
                img = cv2.rotate(image, rotation)

            left_eye, right_eye = self._detect_eyes(img)
            if left_eye is not None:
                # Build similarity transform: maps detected eyes → target MPEG-7 positions
                # Using getAffineTransform with 3 points (2 eyes + a derived third point)
                # Or more directly: compute rotation, scale, translation from the 2 eye pairs

                # Compute similarity transform components
                src_diff = right_eye - left_eye
                dst_diff = self.RIGHT_EYE_TARGET - self.LEFT_EYE_TARGET

                src_dist = np.linalg.norm(src_diff)
                dst_dist = np.linalg.norm(dst_diff)

                if src_dist == 0:
                    continue

                scale = dst_dist / src_dist
                angle_src = np.arctan2(src_diff[1], src_diff[0])
                angle_dst = np.arctan2(dst_diff[1], dst_diff[0])
                angle = angle_dst - angle_src

                # Build 2x3 affine matrix for similarity transform
                cos_a = scale * np.cos(angle)
                sin_a = scale * np.sin(angle)

                # The transform maps left_eye → LEFT_EYE_TARGET
                tx = self.LEFT_EYE_TARGET[0] - (cos_a * left_eye[0] - sin_a * left_eye[1])
                ty = self.LEFT_EYE_TARGET[1] - (sin_a * left_eye[0] + cos_a * left_eye[1])

                M = np.array([
                    [cos_a, -sin_a, tx],
                    [sin_a,  cos_a, ty]
                ], dtype=np.float32)

                # Apply transform to get the normalized face
                normalized = cv2.warpAffine(
                    img, M, (self.TARGET_W, self.TARGET_H),
                    flags=cv2.INTER_CUBIC
                )

                return normalized

        return None
