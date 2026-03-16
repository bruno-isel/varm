"""
VARM – Project 1: Face Detection, Recognition & AR Overlay
===========================================================
Student: Bruno Rodrigues, nº 52323

Pipeline
--------
1. Face Detection   – OpenCV Haar Cascade (frontalface_default.xml)
2. Face Recognition – PCA / Eigenfaces (implemented with NumPy, no extra libs)
3. AR Overlay       – Virtual glasses drawn on each detected face

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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = '/Users/btavr/dev/isel/2oSem/varm/pedroMjorgeDataSet'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'eigenfaces.npz')
TRAIN_TEST_SPLIT = 0.7  # 70% training, 30% test

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


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
# 2 ▸ Face Recognition  (PCA / Eigenfaces)
# ─────────────────────────────────────────────────────────────────────────────

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
        """Resize, equalise histogram, flatten and normalise to [0,1]."""
        face = cv2.resize(face_gray, FACE_SIZE)
        face = cv2.equalizeHist(face)
        return face.astype(np.float64).flatten()

    # ── training ─────────────────────────────────────────────────────────

    def train(self, faces_gray, labels):
        """
        Parameters
        ----------
        faces_gray : list of np.ndarray  – grayscale face crops (any size)
        labels     : list of int
        """
        X = np.stack([self._preprocess(f) for f in faces_gray])  # (N, D)
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

def split_dataset(dataset_dir: str, train_ratio: float = TRAIN_TEST_SPLIT, seed: int = 42):
    """
    Split image files into training and test sets.

    Returns
    -------
    train_files : list[str]
    test_files  : list[str]
    """
    np.random.seed(seed)
    img_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    n_train = int(len(img_files) * train_ratio)
    indices = np.random.permutation(len(img_files))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return [img_files[i] for i in train_indices], [img_files[i] for i in test_indices]


def build_dataset(dataset_dir: str, file_list):
    """
    Build face data from a list of files.

    Returns
    -------
    faces  : list[np.ndarray]
    labels : list[int]
    """
    faces, labels = [], []
    for fname in file_list:
        img = cv2.imread(os.path.join(dataset_dir, fname))
        if img is None:
            print(f"  ⚠  Cannot read {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detect_faces(gray)
        if not detections:
            print(f"  ⚠  No face found in {fname}")
            continue
        x, y, w, h = max(detections, key=lambda r: r[2] * r[3])
        faces.append(gray[y:y+h, x:x+w])
        labels.append(KNOWN_LABEL)
        print(f"  ✓  {fname}  →  {w}×{h} face at ({x},{y})")
    return faces, labels


def load_or_train(dataset_dir: str, force: bool = False):
    rec = EigenfaceRecogniser(n_components=N_COMPONENTS)
    if os.path.exists(MODEL_PATH) and not force:
        rec.load(MODEL_PATH)
    else:
        train_files, test_files = split_dataset(dataset_dir)
        print(f"\n[Dataset Split] {len(train_files)} train, {len(test_files)} test")

        print(f"\n[Training] Building training data…")
        faces_train, labels_train = build_dataset(dataset_dir, train_files)
        if not faces_train:
            raise RuntimeError("No training faces — check DATASET_DIR.")

        rec.train(faces_train, labels_train)
        rec.save(MODEL_PATH)

        print(f"\n[Validation] Testing on test set…")
        faces_test, labels_test = build_dataset(dataset_dir, test_files)
        if faces_test:
            evaluate_model(rec, faces_test, labels_test, test_files)

    return rec


def evaluate_model(rec: EigenfaceRecogniser, faces_test, labels_test, test_files):
    """Evaluate model on test set."""
    correct = 0
    total = len(faces_test)

    for i, face in enumerate(faces_test):
        label, dist = rec.predict(face)
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
