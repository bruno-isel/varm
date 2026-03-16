import os
import cv2
import numpy as np
import random
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from recognizer import FisherEigenFaceRecognizer

def load_dataset(dataset_path):
    """
    Loads normalized images and labels from the dataset directory.
    Handles both:
    1. Dataset/train|test/Person/Image.jpg
    2. Dataset/Person/Image.jpg
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return None, []

    # 1. Identify all person folders
    sub_dirs = [d for d in os.listdir(dataset_path) if d.lower() in ['train', 'test']]
    has_train_test = any(d.lower() == 'train' for d in sub_dirs) and any(d.lower() == 'test' for d in sub_dirs)
    
    all_people = set()
    
    if has_train_test:
        for sd in ['train', 'test']:
            sd_path = os.path.join(dataset_path, sd)
            if os.path.exists(sd_path):
                folders = [f for f in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, f))]
                all_people.update(folders)
    else:
        # Assume person folders are direct children
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        all_people.update(folders)
    
    class_names = sorted(list(all_people))
    name_to_label = {name: i for i, name in enumerate(class_names)}
    
    if not class_names:
        return None, []

    print(f"Found {len(class_names)} classes.")
    
    # 2. Map images to classes
    data_by_class = {i: [] for i in range(len(class_names))}
    
    def load_from_dir(base_path):
        count = 0
        if not os.path.exists(base_path): return 0
        for person_name in os.listdir(base_path):
            person_path = os.path.join(base_path, person_name)
            if not os.path.isdir(person_path) or person_name not in name_to_label:
                continue
                
            label = name_to_label[person_name]
            for img_name in os.listdir(person_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        data_by_class[label].append(img)
                        count += 1
        return count

    if has_train_test:
        load_from_dir(os.path.join(dataset_path, 'train'))
        load_from_dir(os.path.join(dataset_path, 'test'))
    else:
        load_from_dir(dataset_path)
                        
    return data_by_class, class_names

def ensure_uniform_size(images, target_size=None):
    """
    Ensures all images in a list have the same dimensions.
    Returns (processed_list, size_used).
    """
    if not images:
        return [], None
    
    if target_size is None:
        # Take the size of the first image as reference
        target_size = images[0].shape[:2] # (height, width)
        print(f"Reference image size: {target_size[1]}x{target_size[0]}")
    
    processed = []
    resize_count = 0
    for img in images:
        if img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            resize_count += 1
        processed.append(img)
    
    if resize_count > 0:
        print(f"Resized {resize_count} images to match reference size.")
        
    return processed, target_size

def stratified_split(data_by_class, train_percent):
    """
    Splits the data into training and testing sets.
    """
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    for label, images in data_by_class.items():
        if not images:
            continue
            
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_percent / 100.0)
        
        # Ensure at least 1 image for training and 1 for testing if possible
        if n > 1:
            if n_train == 0: n_train = 1
            if n_train == n: n_train = n - 1
            
        train_images.extend(images[:n_train])
        train_labels.extend([label] * n_train)
        
        test_images.extend(images[n_train:])
        test_labels.extend([label] * (n - n_train))
        
    return (train_images, train_labels), (test_images, test_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FisherEigenFaceRecognizer.")
    parser.add_argument("--dataset", type=str, 
                        default=r"default face database",
                        help="Path to the dataset")
    parser.add_argument("--split", type=float, default=80.0, help="Percentage of images for training (0-100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"--- Face Recognition Evaluation ---")
    print(f"Dataset: {args.dataset}")
    
    data_by_class, class_names = load_dataset(args.dataset)
    if data_by_class is None:
        return
        
    num_classes = len(class_names)
    
    print(f"Splitting data ({args.split}% train)...")
    (train_img, train_lbl), (test_img, test_lbl) = stratified_split(data_by_class, args.split)
    
    if not train_img:
        print("Error: No training images found.")
        return

    # Ensure all images are the same size
    print("Checking image uniformity...")
    train_img, ref_size = ensure_uniform_size(train_img)
    test_img, _ = ensure_uniform_size(test_img, target_size=ref_size)
    
    print(f"Training samples: {len(train_img)}")
    print(f"Testing samples: {len(test_img)}")
    
    # Initialize and train recognizer
    recognizer = FisherEigenFaceRecognizer(normalizer=None)
    
    print("Training FisherEigenFaceRecognizer...")
    try:
        recognizer.train(train_img, train_lbl)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Testing predictions...")
    predictions = []
    for img in test_img:
        label, conf = recognizer.predict(img)
        predictions.append(label)
        
    # Accuracy and Metrics using scikit-learn
    if len(test_lbl) > 0:
        accuracy = accuracy_score(test_lbl, predictions) * 100
        
        print(f"\nResults Summary:")
        print(f"Overall Accuracy: {accuracy:.2f}% ({sum(np.array(test_lbl) == np.array(predictions))}/{len(test_lbl)})")
        
        print("\nClassification Report:")
        print(classification_report(test_lbl, predictions, target_names=class_names))

        # Confusion Matrix calculation and plotting
        cm = confusion_matrix(test_lbl, predictions, labels=range(num_classes))
        
        print("Plotting Confusion Matrix...")
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        # Save the plot
        plot_filename = "confusion_matrix.png"
        plt.savefig(plot_filename)
        print(f"Confusion matrix plot saved as '{plot_filename}'")
        
        # Also show the plot (this might block if running in a real GUI environment, 
        # but for scripts it's good to have both or just save)
        plt.show()
    else:
        print("\nNo test samples available for evaluation.")

if __name__ == "__main__":
    main()