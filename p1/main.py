"""
VARM – Project 1: Face Detection, Recognition & AR Overlay
===========================================================
Student: Bruno Rodrigues, nº 52323

Pipeline
--------
1. Face Detection     – OpenCV Haar Cascade (frontalface_default.xml)
2. Face Normalization – MPEG-7: eye alignment, rotation, scaling, crop to 56×46
3. Face Recognition   – PCA / Eigenfaces (implemented with NumPy, no extra libs)
4. AR Overlay         – Virtual glasses drawn on each detected face
5. Validation         – Test set evaluation with accuracy metrics

Usage
-----
    python main.py                    # process whole dataset, save results
    python main.py --demo             # display results on-screen (requires GUI)
    python main.py --image <path>     # single image
    python main.py --retrain          # force re-training

Output
------
    results/   – annotated PNG files
    model/     – saved PCA data (numpy .npz)
"""

import os
import sys
import argparse
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR          = '/Users/btavr/dev/isel/2oSem/varm/pedroMjorgeDataSet'
DATASET_NORMALIZED   = os.path.join(SCRIPT_DIR, 'dataset_normalized')
RESULTS_DIR          = os.path.join(SCRIPT_DIR, 'results')
MODEL_DIR            = os.path.join(SCRIPT_DIR, 'model')
MODEL_PATH           = os.path.join(MODEL_DIR, 'eigenfaces.npz')
TRAIN_TEST_SPLIT     = 0.7  # 70% training, 30% test

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(DATASET_NORMALIZED, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FACE_SIZE       = (80, 80)    # face crops are resized to this before PCA
N_COMPONENTS    = 8           # number of principal components to keep
KNOWN_LABEL     = 0
KNOWN_NAME      = "Pedro/Jorge"
UNKNOWN_NAME    = "Unknown"
# Threshold on reconstruction error (normalised): lower = better match
DIST_THRESHOLD  = 3500.0


# ─────────────────────────────────────────────────────────────────────────────
# 1 ▸ Face Detection  (Haar Cascade)
# ─────────────────────────────────────────────────────────────────────────────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


def detect_faces(gray: np.ndarray):
    """Return list of (x, y, w, h) bounding boxes."""
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    return list(faces) if len(faces) > 0 else []


def detect_eyes(face_gray: np.ndarray):
    """Return eye bounding boxes inside a face ROI."""
    eyes = _eye_cascade.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=5
    )
    return list(eyes) if len(eyes) > 0 else []


# ─────────────────────────────────────────────────────────────────────────────
# 2 ▸ Face Normalization (MPEG-7) & Recognition (PCA / Eigenfaces)
# ─────────────────────────────────────────────────────────────────────────────

MPEG7_SIZE    = (46, 56)   # width=46, height=56 (MPEG-7 spec)
EYE_ROW       = 24         # eyes at row 24
EYE_LEFT_COL  = 16         # left eye at column 16
EYE_RIGHT_COL = 31         # right eye at column 31


def normalize_face_mpeg7(face_gray: np.ndarray) -> np.ndarray:
    """
    MPEG-7 face normalization: align eyes horizontally, crop, resize to 56×46.

    Steps:
    1. Detect both eyes
    2. Calculate rotation angle to align eyes horizontally
    3. Rotate the face
    4. Crop and resize to MPEG-7 dimensions (56×46)
    5. Position eyes at (EYE_ROW, EYE_LEFT_COL) and (EYE_ROW, EYE_RIGHT_COL)

    Returns
    -------
    normalized : np.ndarray
        Grayscale face image 56×46 with aligned eyes
    """
    face = face_gray.copy()
    eyes = detect_eyes(face)

    # If we can detect 2+ eyes, align them; otherwise fall back to simple resize
    if len(eyes) >= 2:
        # Get the two largest eyes
        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])  # sort by x-coordinate

        ex1, ey1, ew1, eh1 = eyes_sorted[0]  # left eye
        ex2, ey2, ew2, eh2 = eyes_sorted[1]  # right eye

        # Eye centers
        c1x, c1y = ex1 + ew1 // 2, ey1 + eh1 // 2
        c2x, c2y = ex2 + ew2 // 2, ey2 + eh2 // 2

        # Calculate rotation angle to align eyes horizontally
        dy = c2y - c1y
        dx = c2x - c1x
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # Rotate face
        h, w = face.shape
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        face_rot = cv2.warpAffine(face, rot_matrix, (w, h), borderValue=0)

        # Re-detect eyes in rotated image
        eyes_rot = detect_eyes(face_rot)
        if len(eyes_rot) >= 2:
            eyes_rot_sorted = sorted(eyes_rot, key=lambda e: e[2] * e[3], reverse=True)[:2]
            eyes_rot_sorted = sorted(eyes_rot_sorted, key=lambda e: e[0])

            ex1_r, ey1_r, ew1_r, eh1_r = eyes_rot_sorted[0]
            ex2_r, ey2_r, ew2_r, eh2_r = eyes_rot_sorted[1]
            c1x_r, c1y_r = ex1_r + ew1_r // 2, ey1_r + eh1_r // 2
            c2x_r, c2y_r = ex2_r + ew2_r // 2, ey2_r + eh2_r // 2

            # Calculate scale: desired eye distance vs actual
            eye_dist_desired = EYE_RIGHT_COL - EYE_LEFT_COL
            eye_dist_actual = c2x_r - c1x_r
            scale = eye_dist_desired / (eye_dist_actual + 1e-6)

            # Apply scaling
            scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
            face_scaled = cv2.warpAffine(face_rot, scale_matrix, (w, h), borderValue=0)

            # Re-detect eyes in scaled image to get final positions
            eyes_scaled = detect_eyes(face_scaled)
            if len(eyes_scaled) >= 2:
                eyes_scaled_sorted = sorted(eyes_scaled, key=lambda e: e[2] * e[3], reverse=True)[:2]
                eyes_scaled_sorted = sorted(eyes_scaled_sorted, key=lambda e: e[0])

                ex1_s, ey1_s, ew1_s, eh1_s = eyes_scaled_sorted[0]
                ex2_s, ey2_s, ew2_s, eh2_s = eyes_scaled_sorted[1]
                c1x_s, c1y_s = ex1_s + ew1_s // 2, ey1_s + eh1_s // 2
                c2x_s, c2y_s = ex2_s + ew2_s // 2, ey2_s + eh2_s // 2

                # Calculate crop region centered on eyes
                eye_center_x = (c1x_s + c2x_s) // 2
                eye_center_y = (c1y_s + c2y_s) // 2

                # Position eyes at target row; calculate crop boundaries
                crop_top = max(0, eye_center_y - EYE_ROW)
                crop_bottom = crop_top + MPEG7_SIZE[1]
                crop_left = max(0, eye_center_x - (EYE_LEFT_COL + EYE_RIGHT_COL) // 2)
                crop_right = crop_left + MPEG7_SIZE[0]

                # Adjust if crop exceeds image boundaries
                if crop_bottom > h:
                    crop_bottom = h
                    crop_top = max(0, crop_bottom - MPEG7_SIZE[1])
                if crop_right > w:
                    crop_right = w
                    crop_left = max(0, crop_right - MPEG7_SIZE[0])

                face_crop = face_scaled[crop_top:crop_bottom, crop_left:crop_right]

                # Resize to final MPEG-7 size
                if face_crop.size > 0:
                    face_norm = cv2.resize(face_crop, MPEG7_SIZE)
                else:
                    face_norm = cv2.resize(face_scaled, MPEG7_SIZE)
            else:
                face_norm = cv2.resize(face_scaled, MPEG7_SIZE)
        else:
            face_norm = cv2.resize(face_rot, MPEG7_SIZE)
    else:
        # Fallback: simple resize
        face_norm = cv2.resize(face, MPEG7_SIZE)

    # Histogram equalization for better contrast
    face_norm = cv2.equalizeHist(face_norm)
    return face_norm

class EigenfaceRecogniser:
    """
    PCA-based face recogniser (Turk & Pentland, 1991).

    Training
    --------
    1. Flatten each (H, W) face image into a 1-D vector.
    2. Compute the mean face and subtract it from every sample.
    3. Build the covariance matrix and compute its eigenvectors (eigenfaces).
    4. Project every training face into eigenface space.

    Recognition
    -----------
    Project the query face into eigenface space, then find the nearest
    training sample using Euclidean distance.  If the distance exceeds
    DIST_THRESHOLD the face is classified as Unknown.
    """

    def __init__(self, n_components: int = N_COMPONENTS):
        self.n_components = n_components
        self.mean_face    = None   # shape (D,)
        self.eigenfaces   = None   # shape (n_components, D)
        self.projections  = None   # shape (N_train, n_components)
        self.labels       = None   # shape (N_train,)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(face_gray: np.ndarray) -> np.ndarray:
        """
        For raw images: MPEG-7 normalization + pixel [0,1] normalization + flatten.
        (Used during inference/predict on new images)
        """
        # Normalize face geometry (eye alignment)
        face_norm = normalize_face_mpeg7(face_gray)

        # Normalize pixel values to [0, 1]
        face_float = face_norm.astype(np.float64) / 255.0

        return face_float.flatten()

    # ── training ─────────────────────────────────────────────────────────

    def train(self, faces_preprocessed, labels):
        """
        Train PCA model on already-preprocessed face data.

        Parameters
        ----------
        faces_preprocessed : list of np.ndarray  – flattened, normalized faces (already processed)
        labels             : list of int
        """
        X = np.stack(faces_preprocessed)  # (N, D)
        self.mean_face = X.mean(axis=0)
        X_centred = X - self.mean_face

        # SVD is more numerically stable than eigendecomposition of X^T X
        # when N < D (which is the case here: 11 faces, D=6400).
        # We use the "dual PCA" trick: compute SVD of X_centred.
        U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

        # Vt[i] is the i-th right singular vector = i-th eigenface
        self.eigenfaces = Vt[:self.n_components]        # (k, D)
        self.projections = X_centred @ self.eigenfaces.T  # (N, k)
        self.labels = np.array(labels)

    # ── recognition ──────────────────────────────────────────────────────

    def predict(self, face_gray: np.ndarray):
        """
        Returns (label, distance).
        Lower distance = better match.
        """
        vec = self._preprocess(face_gray) - self.mean_face   # (D,)
        proj = vec @ self.eigenfaces.T                        # (k,)
        dists = np.linalg.norm(self.projections - proj, axis=1)
        idx   = int(np.argmin(dists))
        return int(self.labels[idx]), float(dists[idx])

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez(path,
                 mean_face=self.mean_face,
                 eigenfaces=self.eigenfaces,
                 projections=self.projections,
                 labels=self.labels,
                 n_components=np.array([self.n_components]))
        print(f"[Model] Saved to {path}")

    def load(self, path: str):
        data = np.load(path)
        self.mean_face    = data['mean_face']
        self.eigenfaces   = data['eigenfaces']
        self.projections  = data['projections']
        self.labels       = data['labels']
        self.n_components = int(data['n_components'][0])
        print(f"[Model] Loaded from {path}  "
              f"(k={self.n_components}, N_train={len(self.labels)})")


# ── dataset helpers ──────────────────────────────────────────────────────────

def normalize_and_save_dataset(raw_dataset_dir: str, output_dir: str):
    """
    Normalize all faces in raw dataset using MPEG-7 and save to disk as JPG.

    This creates a normalized dataset that can be reused for different algorithms.
    Saves as JPG to allow visual inspection of normalization results.

    Parameters
    ----------
    raw_dataset_dir : str
        Path to directory with raw face images
    output_dir : str
        Path to save normalized faces (JPG files)

    Returns
    -------
    normalized_files : list[str]
        List of saved normalized face filenames (without extension)
    """
    img_files = sorted([
        f for f in os.listdir(raw_dataset_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\n[Normalization] Processing {len(img_files)} images from {raw_dataset_dir}")

    normalized_files = []
    for fname in img_files:
        img = cv2.imread(os.path.join(raw_dataset_dir, fname))
        if img is None:
            print(f"  ⚠  Cannot read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face and extract
        detections = detect_faces(gray)
        if not detections:
            print(f"  ⚠  No face found in {fname}")
            continue

        x, y, w, h = max(detections, key=lambda r: r[2] * r[3])
        face_crop = gray[y:y+h, x:x+w]

        # Normalize using MPEG-7
        face_norm = normalize_face_mpeg7(face_crop)

        # Save as JPG (allows visual inspection)
        name_no_ext = fname.rsplit('.', 1)[0]
        save_path = os.path.join(output_dir, f"{name_no_ext}.jpg")
        cv2.imwrite(save_path, face_norm)
        normalized_files.append(name_no_ext)

        print(f"  ✓  {fname:30s}  →  {name_no_ext}.jpg  {face_norm.shape}")

    print(f"\n[Normalization] Saved {len(normalized_files)} normalized faces to {output_dir}")
    return normalized_files


def split_dataset(dataset_dir: str, train_ratio: float = TRAIN_TEST_SPLIT, seed: int = 42):
    """
    Split normalized dataset (JPG images) into training and test sets.

    Returns
    -------
    train_files : list[str]  (filenames without extension)
    test_files  : list[str]  (filenames without extension)
    """
    np.random.seed(seed)
    jpg_files = sorted([
        f[:-4]  # remove .jpg extension
        for f in os.listdir(dataset_dir)
        if f.lower().endswith('.jpg')
    ])
    n_train = int(len(jpg_files) * train_ratio)
    indices = np.random.permutation(len(jpg_files))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return [jpg_files[i] for i in train_indices], [jpg_files[i] for i in test_indices]


def build_dataset(normalized_dataset_dir: str, file_list):
    """
    Load pre-normalized face data from JPG files.

    Parameters
    ----------
    normalized_dataset_dir : str
        Path to directory containing normalized JPG files
    file_list : list[str]
        List of filenames (without .jpg extension)

    Returns
    -------
    faces  : list[np.ndarray]  (shape: (56*46,) flattened, normalized to [0,1])
    labels : list[int]
    """
    faces, labels = [], []
    for fname in file_list:
        jpg_path = os.path.join(normalized_dataset_dir, f"{fname}.jpg")
        try:
            face_norm = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            if face_norm is None:
                print(f"  ⚠  Cannot read {fname}.jpg")
                continue
            # Normalize pixel values to [0, 1] and flatten
            face_float = face_norm.astype(np.float64) / 255.0
            faces.append(face_float.flatten())
            labels.append(KNOWN_LABEL)
            print(f"  ✓  {fname}.jpg  loaded")
        except Exception as e:
            print(f"  ⚠  Error loading {fname}.jpg: {e}")
            continue
    return faces, labels


def load_or_train(raw_dataset_dir: str, force: bool = False):
    rec = EigenfaceRecogniser(n_components=N_COMPONENTS)
    if os.path.exists(MODEL_PATH) and not force:
        rec.load(MODEL_PATH)
    else:
        # Step 1: Normalize raw dataset if not already done
        jpg_count = len([f for f in os.listdir(DATASET_NORMALIZED) if f.endswith('.jpg')])
        if jpg_count == 0:
            normalize_and_save_dataset(raw_dataset_dir, DATASET_NORMALIZED)

        # Step 2: Split normalized dataset
        train_files, test_files = split_dataset(DATASET_NORMALIZED)
        print(f"\n[Dataset Split] {len(train_files)} train, {len(test_files)} test")

        # Step 3: Load training data
        print(f"\n[Training] Loading normalized training data…")
        faces_train, labels_train = build_dataset(DATASET_NORMALIZED, train_files)
        if not faces_train:
            raise RuntimeError("No training faces — check normalized dataset.")

        rec.train(faces_train, labels_train)
        rec.save(MODEL_PATH)

        # Step 4: Validate on test set
        print(f"\n[Validation] Testing on test set…")
        faces_test, labels_test = build_dataset(DATASET_NORMALIZED, test_files)
        if faces_test:
            evaluate_model(rec, faces_test, labels_test, test_files)

    return rec


def evaluate_model(rec: EigenfaceRecogniser, faces_test, labels_test, test_files):
    """Evaluate model on test set (pre-normalized data)."""
    correct = 0
    total = len(faces_test)

    for i, face_vec in enumerate(faces_test):
        # face_vec is already flattened and normalized (from build_dataset)
        # Directly use it for prediction without calling _preprocess()
        vec = face_vec - rec.mean_face  # center with mean face from training
        proj = vec @ rec.eigenfaces.T                        # (k,)
        dists = np.linalg.norm(rec.projections - proj, axis=1)
        idx = int(np.argmin(dists))
        label = int(rec.labels[idx])
        dist = float(dists[idx])

        is_correct = (label == labels_test[i]) and (dist < DIST_THRESHOLD)
        status = "✓" if is_correct else "✗"
        correct += is_correct

        fname = test_files[i] if i < len(test_files) else "unknown"
        name = KNOWN_NAME if dist < DIST_THRESHOLD else UNKNOWN_NAME
        print(f"  {status}  {fname:30s}  →  {name:15s}  dist={dist:.0f}")

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n[Accuracy] {correct}/{total} = {accuracy:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 3 ▸ AR Overlay  (virtual glasses)
# ─────────────────────────────────────────────────────────────────────────────

def draw_glasses(frame: np.ndarray, face_rect, face_gray: np.ndarray):
    """
    Overlay stylised vector glasses on the detected face.
    Uses eye positions when available; falls back to geometric estimation.
    """
    fx, fy, fw, fh = face_rect
    eyes = detect_eyes(face_gray)

    if len(eyes) >= 2:
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes_sorted[0]
        ex2, ey2, ew2, eh2 = eyes_sorted[1]
        c1x = fx + ex1 + ew1 // 2
        c1y = fy + ey1 + eh1 // 2
        c2x = fx + ex2 + ew2 // 2
        c2y = fy + ey2 + eh2 // 2
        lens_rx = max(ew1, ew2) // 2 + 6
    else:
        c1x = fx + fw // 3
        c1y = fy + fh * 2 // 5
        c2x = fx + 2 * fw // 3
        c2y = fy + fh * 2 // 5
        lens_rx = fw // 7

    lens_ry = int(lens_rx * 0.65)
    frame_col  = (30, 20, 10)     # dark brown frames
    lens_col   = (255, 210, 100)  # amber tint

    # Semi-transparent lens fill
    overlay = frame.copy()
    cv2.ellipse(overlay, (c1x, c1y), (lens_rx, lens_ry), 0, 0, 360, lens_col, -1)
    cv2.ellipse(overlay, (c2x, c2y), (lens_rx, lens_ry), 0, 0, 360, lens_col, -1)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # Frame outlines
    thickness = 2
    cv2.ellipse(frame, (c1x, c1y), (lens_rx, lens_ry), 0, 0, 360, frame_col, thickness)
    cv2.ellipse(frame, (c2x, c2y), (lens_rx, lens_ry), 0, 0, 360, frame_col, thickness)

    # Nose bridge
    cv2.line(frame, (c1x + lens_rx, c1y), (c2x - lens_rx, c2y), frame_col, thickness)

    # Temple arms
    arm_len = fw // 7
    cv2.line(frame, (c1x - lens_rx, c1y),
             (c1x - lens_rx - arm_len, c1y + 4), frame_col, thickness)
    cv2.line(frame, (c2x + lens_rx, c2y),
             (c2x + lens_rx + arm_len, c2y + 4), frame_col, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img: np.ndarray, rec: EigenfaceRecogniser):
    """
    Detect faces, recognise each, draw AR glasses and labels.

    Returns
    -------
    annotated : np.ndarray
    results   : list[dict]
    """
    annotated = img.copy()
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detect_faces(gray)
    results    = []

    for (x, y, w, h) in detections:
        face_gray  = gray[y:y+h, x:x+w]
        label, dist = rec.predict(face_gray)
        name = KNOWN_NAME if dist < DIST_THRESHOLD else UNKNOWN_NAME
        colour = (0, 200, 0) if name == KNOWN_NAME else (0, 0, 220)

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 2)

        # Label
        label_text = f"{name}  d={dist:.0f}"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), colour, -1)
        cv2.putText(annotated, label_text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # AR overlay
        draw_glasses(annotated, (x, y, w, h), face_gray)

        results.append(dict(bbox=(x, y, w, h), name=name, distance=dist))

    return annotated, results


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_on_dataset(rec: EigenfaceRecogniser, demo: bool = False):
    img_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"\n[Pipeline] Processing {len(img_files)} images …\n")

    for fname in img_files:
        img = cv2.imread(os.path.join(DATASET_DIR, fname))
        if img is None:
            print(f"  ⚠  Cannot read {fname}")
            continue

        annotated, results = process_image(img, rec)
        out_path = os.path.join(RESULTS_DIR, f"result_{fname.rsplit('.', 1)[0]}.jpg")
        cv2.imwrite(out_path, annotated)

        for r in results:
            x, y, w, h = r['bbox']
            print(f"  {fname:30s}  │  {w}×{h}  │  "
                  f"{r['name']:15s}  dist={r['distance']:.0f}")

        if demo:
            cv2.imshow('VARM P1 – Face Detection & AR', annotated)
            key = cv2.waitKey(0)
            if key == 27:
                break

    if demo:
        cv2.destroyAllWindows()

    print(f"\n[Done] Annotated images saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VARM P1 – Face Pipeline')
    parser.add_argument('--demo',    action='store_true',
                        help='Display results (requires a display)')
    parser.add_argument('--image',   type=str, default=None,
                        help='Process a single image')
    parser.add_argument('--retrain', action='store_true',
                        help='Force re-training')
    args = parser.parse_args()

    rec = load_or_train(DATASET_DIR, force=args.retrain)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: cannot open {args.image}")
            sys.exit(1)
        annotated, results = process_image(img, rec)
        out_path = os.path.join(RESULTS_DIR,
                                'result_' + os.path.basename(args.image))
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")
        for r in results:
            print(f"  {r['name']}  dist={r['distance']:.0f}")
        if args.demo:
            cv2.imshow('VARM P1', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_on_dataset(rec, demo=args.demo)
