import numpy as np

class FisherEigenFaceRecognizer:
    """
    A Face Recognition Classifier based on Eigenfaces and Fisherfaces 
    (PCA + LDA) using numpy and OpenCV.
    Designed to mimic the behavior of OpenCV's face recognizer API.
    """
    def __init__(self, normalizer=None, num_components=0, threshold=np.inf):
        """
        Initialize the recognizer.
        :param normalizer: An object with a .normalize(images) method. 
                           If None, images passed to train/predict are assumed to be already normalized matrices.
        :param num_components: The number of components to keep in PCA/LDA (0 means keep all possible).
        :param threshold: The threshold applied in the prediction. If the distance is larger, it returns -1.
        """
        self.normalizer = normalizer
        self.num_components = num_components
        self.threshold = threshold
        
        # Model parameters to be learned
        self.mean_face = None
        self.W_pca = None
        self.W_mda = None
        self.W = None       # Combined projection matrix (W_pca @ W_mda)
        
        # Training data projections and labels
        self.projections = []
        self.labels = []
        
    def _compute_wpca(self, X):
        """
        Compute the principal components (Eigenfaces).
        :param X: Mean-centered data (N x D).
        """
        # For Fisherfaces, we keep N-C components in PCA step to ensure Sw is not singular.
        num_classes = len(np.unique(self.labels))
        n_components = X.shape[0] - num_classes
        
        if self.num_components > 0:
            n_components = min(n_components, self.num_components)

        # Check for non-finite values (NaN/Inf) without creating a full boolean mask
        if not np.isfinite(np.sum(X)):
            X = np.nan_to_num(X, copy=False)

        # Force float32 for efficiency
        X = np.ascontiguousarray(X, dtype=np.float32)
        
        # --- SMALL MATRIX TRICK FOR PCA ---
        # Instead of SVD(X) which is (N x D), we solve eig(X @ X.T) which is (N x N)
        # 1. Compute L = X @ X.T (Samples x Samples)
        L = np.dot(X, X.T)
        
        # 2. Compute eigenvalues/vectors of the small matrix L
        # eigh is for symmetric matrices (L is symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
       
        # 3. Sort by eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 4. Filter components and manage dimensionality
        # Fisherfaces is sensitive to noisy/redundant components. 
        # Components with extremely small eigenvalues are mostly noise.
        # We discard components that contribute less than 1e-6 relative to the max.
        if len(eigenvalues) > 0:
            threshold = eigenvalues[0] * 1e-6
            keep = eigenvalues > threshold
            eigenvalues = eigenvalues[keep]
            eigenvectors = eigenvectors[:, keep]
        
        # To prevent overfitting, we should cap the number of components.
        # Keeping N-C components is a mathematical limit, but keeping too many 
        # (e.g. > 150) often allows LDA to overfit to training noise.
        # If self.num_components is 0, we use a sensible default limit.
        max_components = 150 
        if self.num_components > 0:
            n_components = min(self.num_components, len(eigenvalues))
        else:
            # Automatic limit: min(N-C, available, 150)
            n_components = min(n_components, len(eigenvalues), max_components)
        
        if n_components <= 0:
            raise ValueError("No valid PCA components found. Check if images are identical or too few.")

        eigenvectors = eigenvectors[:, :n_components]
        
        # 5. Get Eigenfaces of X.T @ X: W_pca = X.T @ eigenvectors
        W_pca = np.dot(X.T, eigenvectors)

        # 6. Normalize each eigenvector to unit length
        # Using sqrt(eigenvalues) for normalization is more standard, 
        # but manual norm is robust to scaling.
        norms = np.linalg.norm(W_pca, axis=0)
        W_pca /= (norms + 1e-15)
        
        return W_pca.astype(np.float32)

    def _compute_wmda(self, X_pca, labels):
        """
        Compute the Fisher Linear Discriminant (Fisherfaces).
        :param X_pca: Training data projected onto PCA subspace (N x K).
        :param labels: Corresponding labels.
        """
        n_features = X_pca.shape[1]
        classes = np.unique(labels)
        mean_overall = np.mean(X_pca, axis=0)
        
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        for c in classes:
            X_c = X_pca[labels == c]
            mean_c = np.mean(X_c, axis=0)
            
            # Within-class scatter: sum of (x-mean_c)(x-mean_c)^T
            diff = X_c - mean_c
            Sw += np.dot(diff.T, diff)
            
            # Between-class scatter: n_c * (mean_c-mean_o)(mean_c-mean_o)^T
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            Sb += n_c * np.dot(mean_diff, mean_diff.T)
            
        # Solve generalized eigenvalue problem Sb*v = lambda*Sw*v
        # Since we reduced dimension to N-C in PCA, Sw should be non-singular.
        # We use pinv for extra robustness.
        
        # Check for NaNs in Sw or Sb
        if not np.all(np.isfinite(Sw)) or not np.all(np.isfinite(Sb)):
            raise ValueError("Scatter matrices contain non-finite values. Check input data.")

        try:
            # Compute pinv(Sw) @ Sb
            inv_Sw = np.linalg.pinv(Sw)
            mat = np.dot(inv_Sw, Sb)
            
            # Use eigh for symmetric matrices, but mat is not necessarily symmetric
            # eigenvalues, eigenvectors = np.linalg.eigh(mat) 
            # Actually use eig since pinv(Sw)@Sb is not symmetric
            eigenvalues, eigenvectors = np.linalg.eig(mat)
            
            # Take only the real part (eigenvalues of pinv(Sw)@Sb should be real)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
        except np.linalg.LinAlgError as e:
            print(f"Eigenvalue decomposition failed: {e}. Check class consistency.")
            # Fallback to zeros/identity to avoid crash, though this means training failed
            return np.zeros((n_features, len(classes)-1))

        # Sort by eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Fisherfaces has at most C-1 components
        return eigenvectors[:, :len(classes)-1]
        
    def train(self, images, labels):
        """
        Trains the face recognizer with a given dataset.
        :param images: List of images OR a pre-normalized matrix (samples x features).
        :param labels: List or array of integer labels corresponding to the images.
        """
        if len(images) == 0 or len(labels) == 0:
            raise ValueError("Training images and labels cannot be empty.")
            
        # 1. Normalize images if a normalizer is provided AND images are not already pre-normalized
        if self.normalizer is not None and not isinstance(images, np.ndarray):
            # For training, we STRICTLY require eyes to keep the model subspace clean.
            # normalize() now returns (X, filtered_labels) if you pass labels.
            X, self.labels = self.normalizer.normalize(images, labels, require_eyes=True)
        else:
            # Assume images is already a matrix (samples x features) or pre-normalized array
            self.labels = np.array(labels)
            X = np.array(images, dtype=np.float32)
            
        # Flatten if X is 3D (Samples, Rows, Cols)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 2 and not isinstance(images, np.ndarray):
            # If it's a list of flattened vectors, np.array(images) is already (N, D)
            pass
        
        # 2. Compute Mean Face and Center Data (In-place to save memory)
        # Force float32 to avoid dtype promotion to float64 and save memory
        self.mean_face = np.mean(X, axis=0, dtype=np.float32)
        X -= self.mean_face
        
        # 3. Compute W_pca (Eigenfaces step)
        self.W_pca = self._compute_wpca(X) # X is already centered
        
        # 4. Project centered data onto PCA subspace
        X_pca = np.dot(X, self.W_pca)
        
        # 5. Compute W_mda (Fisherfaces / LDA step)
        self.W_mda = self._compute_wmda(X_pca, self.labels)
        
        # 6. Compute combined projection matrix W = W_pca * W_mda
        self.W = np.dot(self.W_pca, self.W_mda)
        
        # 7. Project all training samples into the Fisherface subspace
        self.projections = np.dot(X, self.W)

    def predict(self, image, require_eyes=True):
        """
        Predicts the label for a given image.
        :param image: A single test image (unnormalized) or a single row normalized vector.
        :param require_eyes: If True, only recognizes if eyes are detected.
        :return: Tuple (predicted_label, confidence/distance). 
                 Returns (-1, inf) if below threshold, or (-2, inf) if eyes/face not detected.
        """
        if self.W is None:
            raise RuntimeError("The model has not been trained yet. Call train() first.")
            
        # 1. Normalize the single image if a normalizer is provided
        if self.normalizer is not None:
            # Most normalizers expect a list/batch, so we wrap it
            x = self.normalizer.normalize([image], require_eyes=require_eyes)
            
            if x.shape[0] == 0:
                # No face/eyes detected
                return -2, np.inf

            # Flatten if normalizer returned (1, Rows, Cols)
            if x.ndim == 3:
                x = x.reshape(1, -1)
            elif x.ndim == 2 and x.shape[0] != 1: # Single image HxW
                x = x.reshape(1, -1)
        else:
            x = np.array(image).reshape(1, -1)
        
        # 2. Subtract the mean face
        x_centered = x - self.mean_face
        
        # 3. Project into the Fisherface space
        projection = np.dot(x_centered, self.W)
        
        # 4. Find the nearest neighbor (using Euclidean distance)
        distances = np.linalg.norm(self.projections - projection, axis=1)
        
        # 5. Find the index of the minimum distance
        min_dist_idx = np.argmin(distances)
        min_distance = distances[min_dist_idx]
        
        # 6. Apply distance threshold
        if min_distance > self.threshold:
            return -1, min_distance
            
        predicted_label = self.labels[min_dist_idx]
        return predicted_label, min_distance

    def save(self, filename):
        """
        Saves the trained model to a file using numpy's npz format.
        """
        if self.W is None:
            raise RuntimeError("Cannot save an untrained model.")
        
        np.savez(filename, 
                 mean_face=self.mean_face, 
                 W=self.W, 
                 projections=self.projections, 
                 labels=self.labels,
                 num_components=self.num_components,
                 threshold=self.threshold)

    def load(self, filename):
        """
        Loads a trained model from an .npz file.
        :param filename: Path to the .npz file
        """
        data = np.load(filename)
        self.mean_face = data['mean_face']
        self.W = data['W']
        self.projections = data['projections']
        self.labels = data['labels']
        self.num_components = data['num_components']
        self.threshold = data['threshold']