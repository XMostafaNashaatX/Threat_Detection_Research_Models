# -*- coding: utf-8 -*-
"""Abdelwahab_XAI_model3_(1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N8H1D_uJf_71r7nNlJ1lY__cNZ04d53v
"""

pip install opencv-python numpy face_recognition scikit-learn tqdm matplotlib

import os
import cv2
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

class FaceAnalysisSystem:
    """
    A system inspired by BANE for face detection, feature extraction,
    and matching across image collections.
    """

    def __init__(self, match_threshold=0.67, quality_threshold=0.5):
        """
        Initialize the face analysis system.

        Parameters:
        - match_threshold: Threshold above which two faces are considered a match (0 to 1)
        - quality_threshold: Threshold for face quality to be considered valid
        """
        self.match_threshold = match_threshold
        self.quality_threshold = quality_threshold

    def detect_face(self, image_path):
        """
        Detect a face in an image and return its location.
        For pre-cropped faces, we'll assume the entire image is the face.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For pre-cropped faces, we'll use the entire image
        h, w, _ = rgb_image.shape
        face_location = (0, w, h, 0)  # top, right, bottom, left format

        return rgb_image, face_location

    def calculate_face_quality(self, face_image):
        """
        Calculate a quality score for a face.

        This is a simplified version since we don't have access to DSTG's algorithm.
        We'll use metrics like face size, image sharpness, and brightness as proxies for quality.
        """
        # Face size (larger is generally better for recognition)
        h, w, _ = face_image.shape
        size_score = min(1.0, (h * w) / (224 * 224))  # Normalize size

        # Image sharpness using Laplacian variance (higher is sharper)
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)  # Normalize sharpness

        # Brightness and contrast
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        brightness_score = 1.0 - 2.0 * abs(0.5 - brightness)  # Penalize too bright or too dark
        contrast_score = min(1.0, contrast)

        # Combine scores (you may want to weight these differently)
        quality_score = 0.3 * size_score + 0.3 * sharpness_score + 0.2 * brightness_score + 0.2 * contrast_score

        return quality_score

    def extract_facial_features(self, rgb_image, face_location):
        """
        Extract facial features from a detected face.

        Returns:
        - face_encoding: facial feature vector
        - quality_score: estimated quality of the face image
        """
        # Extract the face from the image
        top, right, bottom, left = face_location
        face_image = rgb_image[top:bottom, left:right]

        # Calculate face quality
        quality_score = self.calculate_face_quality(face_image)

        # Extract face encoding (embedding)
        face_encoding = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])

        if len(face_encoding) == 0:
            return None, quality_score

        return face_encoding[0], quality_score

    def match_faces(self, encoding1, encoding2):
        """
        Match two face encodings and return a similarity score.

        The score is transformed to be between -1 and 1, where higher values indicate more similarity.
        """
        # Calculate cosine similarity between encodings
        similarity = cosine_similarity([encoding1], [encoding2])[0][0]

        # Transform to range [-1, 1] as specified in the paper
        # face_recognition distances are already 0-1 range where lower means more similar
        # so we need to invert and rescale
        transformed_score = 2 * similarity - 1

        return transformed_score

    def process_image_collection(self, image_dir, label=None):
        """
        Process a collection of images (e.g., frames from one video).

        Returns:
        - representative_faces: list of (encoding, quality, path) for the best face of each person
        - all_faces: all detected faces with their encodings, qualities, and paths
        """
        all_faces = []  # Store all detected faces

        # Get all image files
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Process every image
        for image_file in tqdm(image_files, desc=f"Processing images in {os.path.basename(image_dir)}"):
            image_path = os.path.join(image_dir, image_file)

            # Detect face in the image
            rgb_image, face_location = self.detect_face(image_path)
            if rgb_image is None:
                continue

            # Extract facial features and quality score
            face_encoding, quality_score = self.extract_facial_features(rgb_image, face_location)

            if face_encoding is not None and quality_score >= self.quality_threshold:
                all_faces.append({
                    'encoding': face_encoding,
                    'quality': quality_score,
                    'path': image_path,
                    'label': label
                })

        # Identify duplicate faces (same person in different images)
        # Group faces by identity
        identity_groups = self.group_faces_by_identity(all_faces)

        # For each identity group, keep the face with the highest quality score
        representative_faces = []
        for identity_group in identity_groups:
            best_face = max(identity_group, key=lambda x: x['quality'])
            representative_faces.append(best_face)

        return representative_faces, all_faces

    def group_faces_by_identity(self, faces):
        """
        Group faces that belong to the same person based on similarity.

        Returns a list of lists, where each inner list contains faces of the same person.
        """
        if not faces:
            return []

        # Start with each face in its own group
        groups = [[face] for face in faces]

        # Merge groups if faces match across groups
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                # Check if any face in group i matches any face in group j
                match_found = False
                for face_i in groups[i]:
                    for face_j in groups[j]:
                        match_score = self.match_faces(face_i['encoding'], face_j['encoding'])
                        if match_score >= self.match_threshold:
                            # Merge group j into group i
                            groups[i].extend(groups[j])
                            groups.pop(j)
                            match_found = True
                            break
                    if match_found:
                        break

                if not match_found:
                    j += 1
            i += 1

        return groups

    def compare_collections(self, collection1, collection2):
        """
        Compare two collections of representative faces and find matches.

        Returns:
        - matches: list of (face1, face2, score) for matched pairs above threshold
        """
        matches = []

        # Compare each representative face from collection1 to each from collection2
        for face1 in collection1:
            for face2 in collection2:
                match_score = self.match_faces(face1['encoding'], face2['encoding'])

                if match_score >= self.match_threshold:
                    matches.append((face1, face2, match_score))

        return matches

    def evaluate_accuracy(self, matches, verbose=True):
        """
        Evaluate the accuracy of matches based on their labels (if available).
        """
        if not matches:
            return None, None

        # Count correct and incorrect matches
        correct_matches = 0
        incorrect_matches = 0

        for face1, face2, score in matches:
            if 'label' in face1 and 'label' in face2:
                if face1['label'] == face2['label']:
                    correct_matches += 1
                else:
                    incorrect_matches += 1

        total_evaluated = correct_matches + incorrect_matches

        if total_evaluated == 0:
            if verbose:
                print("No labeled faces to evaluate accuracy.")
            return None, None

        accuracy = correct_matches / total_evaluated if total_evaluated > 0 else 0

        if verbose:
            print(f"Match Accuracy: {accuracy:.3f} ({correct_matches}/{total_evaluated})")
            print(f"False Match Rate: {incorrect_matches/total_evaluated:.3f} ({incorrect_matches}/{total_evaluated})")

        return accuracy, incorrect_matches/total_evaluated

    def visualize_matches(self, matches, output_dir, max_to_show=10):
        """
        Visualize matches by displaying pairs of matched faces side by side.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Sort matches by score (highest first)
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

        # Limit the number to display
        matches_to_show = sorted_matches[:max_to_show]

        for i, (face1, face2, score) in enumerate(matches_to_show):
            # Load images
            img1 = cv2.imread(face1['path'])
            img2 = cv2.imread(face2['path'])

            if img1 is None or img2 is None:
                continue

            # Resize to same height if needed
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            # Choose the smaller height
            target_height = min(h1, h2, 300)  # Limit max height

            # Resize maintaining aspect ratio
            img1 = cv2.resize(img1, (int(w1 * target_height / h1), target_height))
            img2 = cv2.resize(img2, (int(w2 * target_height / h2), target_height))

            # Create a side-by-side image with labels
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            # Create combined image
            combined_width = w1 + w2 + 20  # 20 pixels between images
            combined_img = np.zeros((target_height + 50, combined_width, 3), dtype=np.uint8) + 255

            # Add images
            combined_img[:h1, :w1] = img1
            combined_img[:h2, w1+20:w1+20+w2] = img2

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            label1 = face1.get('label', 'Unknown')
            label2 = face2.get('label', 'Unknown')

            cv2.putText(combined_img, f"Match Score: {score:.3f}", (10, target_height + 20),
                        font, font_scale, (0, 0, 0), 1)
            cv2.putText(combined_img, f"Left: {label1}", (10, target_height + 40),
                        font, font_scale, (0, 0, 0), 1)
            cv2.putText(combined_img, f"Right: {label2}", (combined_width//2, target_height + 40),
                        font, font_scale, (0, 0, 0), 1)

            # Save the combined image
            output_path = os.path.join(output_dir, f"match_{i+1}_score_{score:.3f}.jpg")
            cv2.imwrite(output_path, combined_img)

        print(f"Visualizations saved to {output_dir}")

def process_dataset(base_dir, output_dir, match_threshold=0.67):
    """
    Process an entire dataset organized as:
    base_dir/
        emotion1/
            image1.jpg
            image2.jpg
        emotion2/
            image3.jpg
            ...
    """
    # Initialize the face analysis system
    face_system = FaceAnalysisSystem(match_threshold=match_threshold)

    # Get emotion categories (subdirectories)
    emotion_categories = [d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d))]

    print(f"Found emotion categories: {emotion_categories}")

    all_representative_faces = []
    emotion_representatives = {}

    # Process each emotion category
    for emotion in emotion_categories:
        emotion_dir = os.path.join(base_dir, emotion)
        print(f"\nProcessing emotion: {emotion}")

        # Process all images in this emotion category
        representative_faces, all_faces = face_system.process_image_collection(
            emotion_dir, label=emotion)

        print(f"Found {len(representative_faces)} unique faces out of {len(all_faces)} total faces")

        # Save representative faces for this emotion
        emotion_representatives[emotion] = representative_faces
        all_representative_faces.extend(representative_faces)

        # Save the processed data
        output_file = os.path.join(output_dir, f"{emotion}_faces.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'representative_faces': representative_faces,
                'all_faces': all_faces
            }, f)

        print(f"Saved processed data to {output_file}")

    # Cross-emotion matching
    print("\nPerforming cross-emotion matching...")
    all_matches = []

    # Compare each emotion category with every other
    for i, emotion1 in enumerate(emotion_categories):
        for j, emotion2 in enumerate(emotion_categories[i:], i):
            if i == j:  # Skip self-comparison
                continue

            print(f"Comparing {emotion1} vs {emotion2}")
            matches = face_system.compare_collections(
                emotion_representatives[emotion1],
                emotion_representatives[emotion2]
            )

            print(f"Found {len(matches)} matches above threshold {face_system.match_threshold}")
            all_matches.extend(matches)

            # Evaluate accuracy if labels are available
            face_system.evaluate_accuracy(matches)

            # Visualize some matches
            vis_dir = os.path.join(output_dir, f"vis_{emotion1}_vs_{emotion2}")
            face_system.visualize_matches(matches, vis_dir)

    # Save all matches
    all_matches_file = os.path.join(output_dir, "all_matches.pkl")
    with open(all_matches_file, 'wb') as f:
        pickle.dump(all_matches, f)

    print(f"Saved all matches to {all_matches_file}")

    return all_representative_faces, all_matches

def main():
    # Paths
    DATASET_DIR = "/Users/abdelwahab/3rd-year/s2/7_emotion_dataset"
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    # Create output directories
    TRAIN_OUTPUT_DIR = os.path.join(DATASET_DIR, "analysis_train")
    TEST_OUTPUT_DIR = os.path.join(DATASET_DIR, "analysis_test")
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    print("Starting face analysis system...")

    # Process train dataset
    print("\n=== Processing TRAIN dataset ===")
    train_faces, train_matches = process_dataset(TRAIN_DIR, TRAIN_OUTPUT_DIR)

    # Process test dataset
    print("\n=== Processing TEST dataset ===")
    test_faces, test_matches = process_dataset(TEST_DIR, TEST_OUTPUT_DIR)

    # Compare train representative faces with test representative faces
    print("\n=== Comparing TRAIN vs TEST datasets ===")
    face_system = FaceAnalysisSystem()
    train_test_matches = face_system.compare_collections(train_faces, test_faces)

    print(f"Found {len(train_test_matches)} matches between train and test datasets")

    # Evaluate accuracy of train-test matches
    face_system.evaluate_accuracy(train_test_matches)

    # Visualize some train-test matches
    vis_dir = os.path.join(DATASET_DIR, "vis_train_vs_test")
    face_system.visualize_matches(train_test_matches, vis_dir, max_to_show=20)

    # Save train-test matches
    train_test_file = os.path.join(DATASET_DIR, "train_test_matches.pkl")
    with open(train_test_file, 'wb') as f:
        pickle.dump(train_test_matches, f)

    print(f"Saved train-test matches to {train_test_file}")

    print("\nCompleted face analysis process.")

if __name__ == "__main__":
    main()

pip install --upgrade threadpoolctl scikit-learn numpy

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_face_quality_distribution(faces, dataset_name):
    """Visualize quality distribution of the faces in the dataset."""
    qualities = [face['quality_score'] for face in faces]  # Assuming quality_score is available
    plt.figure(figsize=(10, 6))
    sns.histplot(qualities, bins=20, kde=True, color='blue', alpha=0.7)
    plt.title(f'{dataset_name} Quality Score Distribution')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.show()

def basic_data_info(faces, dataset_name):
    """Print basic data info like total faces and features."""
    print(f"=== {dataset_name} Dataset Info ===")
    print(f"Total faces processed: {len(faces)}")

    # Assuming each face is a dictionary containing image-related features
    if len(faces) > 0:
        sample_face = faces[0]
        print(f"Sample face features: {list(sample_face.keys())}")

def check_missing_or_corrupted_faces(faces, dataset_name):
    """Check for missing or corrupted faces."""
    print(f"=== {dataset_name} Missing/Corrupted Faces Check ===")
    missing_faces = [i for i, face in enumerate(faces) if face.get('image_data') is None]  # Assuming 'image_data' is a field
    if missing_faces:
        print(f"Missing/Corrupted Faces found at indices: {missing_faces}")
    else:
        print("No missing/corrupted faces found.")

def compare_train_test_statistics(train_faces, test_faces):
    """Compare statistics between train and test datasets."""
    print("\n=== Comparing Train and Test Dataset Statistics ===")

    # Compare average quality scores between the datasets
    train_qualities = [face['quality_score'] for face in train_faces]
    test_qualities = [face['quality_score'] for face in test_faces]

    print(f"Train dataset average quality score: {sum(train_qualities) / len(train_qualities)}")
    print(f"Test dataset average quality score: {sum(test_qualities) / len(test_qualities)}")

    # Visualize the comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(train_qualities, bins=20, kde=True, color='blue', alpha=0.6, label='Train')
    sns.histplot(test_qualities, bins=20, kde=True, color='red', alpha=0.6, label='Test')
    plt.title('Train vs Test Quality Score Comparison')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    # Paths
    DATASET_DIR = "/Users/abdelwahab/3rd-year/s2/7_emotion_dataset"
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    # Create output directories
    TRAIN_OUTPUT_DIR = os.path.join(DATASET_DIR, "analysis_train")
    TEST_OUTPUT_DIR = os.path.join(DATASET_DIR, "analysis_test")
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    print("Starting face analysis system...")

    # Process train dataset
    print("\n=== Processing TRAIN dataset ===")
    train_faces, train_matches = process_dataset(TRAIN_DIR, TRAIN_OUTPUT_DIR)

    # Process test dataset
    print("\n=== Processing TEST dataset ===")
    test_faces, test_matches = process_dataset(TEST_DIR, TEST_OUTPUT_DIR)

    # EDA for train dataset
    basic_data_info(train_faces, "TRAIN")
    visualize_face_quality_distribution(train_faces, "TRAIN")
    check_missing_or_corrupted_faces(train_faces, "TRAIN")

    # EDA for test dataset
    basic_data_info(test_faces, "TEST")
    visualize_face_quality_distribution(test_faces, "TEST")
    check_missing_or_corrupted_faces(test_faces, "TEST")

    # Compare statistics between train and test datasets
    compare_train_test_statistics(train_faces, test_faces)

    # Compare train representative faces with test representative faces
    print("\n=== Comparing TRAIN vs TEST datasets ===")
    face_system = FaceAnalysisSystem()
    train_test_matches = face_system.compare_collections(train_faces, test_faces)

    print(f"Found {len(train_test_matches)} matches between train and test datasets")

    # Evaluate accuracy of train-test matches
    face_system.evaluate_accuracy(train_test_matches)

    # Visualize some train-test matches
    vis_dir = os.path.join(DATASET_DIR, "vis_train_vs_test")
    face_system.visualize_matches(train_test_matches, vis_dir, max_to_show=20)

    # Save train-test matches
    train_test_file = os.path.join(DATASET_DIR, "train_test_matches.pkl")
    with open(train_test_file, 'wb') as f:
        pickle.dump(train_test_matches, f)

    print(f"Saved train-test matches to {train_test_file}")

    print("\nCompleted face analysis process.")

if __name__ == "__main__":
    main()

import os
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

def load_embeddings(filepath):
    """Load embeddings from a pickle file."""
    print(f"Loading embeddings from {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data["embeddings"], data["labels"]

def train_svm_model_fast(train_embeddings, train_labels):
    """
    Train an SVM model on the face embeddings with minimal computation time.
    Uses LinearSVC which is much faster than the standard SVC.

    Parameters:
    - train_embeddings: numpy array of face embeddings
    - train_labels: numpy array of emotion labels

    Returns:
    - trained SVM model
    - scaler for preprocessing new data
    """
    print("Preparing data for fast SVM training...")

    # Start timing
    start_time = time()

    # Preprocess embeddings - standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(train_embeddings)

    # Train Linear SVM model (much faster than RBF kernel)
    print("Training LinearSVC model (faster implementation)...")
    model = LinearSVC(C=1.0, dual="auto", random_state=42, max_iter=1000)
    model.fit(scaled_embeddings, train_labels)

    # End timing
    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return model, scaler

def evaluate_model(model, scaler, test_embeddings, test_labels):
    """
    Evaluate the trained SVM model on test data.

    Parameters:
    - model: trained SVM model
    - scaler: fitted StandardScaler
    - test_embeddings: numpy array of test face embeddings
    - test_labels: numpy array of test emotion labels

    Returns:
    - accuracy: overall model accuracy
    - report: classification report with precision, recall, f1-score
    """
    print("Evaluating model on test data...")

    # Preprocess test embeddings
    scaled_test_embeddings = scaler.transform(test_embeddings)

    # Start timing
    start_time = time()

    # Predict on test data
    predictions = model.predict(scaled_test_embeddings)

    # End timing
    prediction_time = time() - start_time
    print(f"Prediction completed in {prediction_time:.2f} seconds")

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)

    # Generate classification report
    report = classification_report(test_labels, predictions, output_dict=True)

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, predictions))

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predictions)

    return accuracy, report, cm, predictions

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.tight_layout()
    plt.close()

def save_model(model, scaler, class_names, filepath):
    """Save the trained model and scaler to a file."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'class_names': class_names
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {filepath}")

def convert_to_probability_model(predictions, n_classes):
    """
    Convert LinearSVC's decision function to probabilities.
    This is a simple conversion and not as accurate as SVC's built-in probabilities.
    """
    return np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)

def main():
    # Start timing the entire process
    total_start_time = time()

    # Paths
    DATASET_DIR = "/Users/abdelwahab/3rd-year/s2/7_emotion_dataset"
    EMBEDDINGS_DIR = os.path.join(DATASET_DIR, "embeddings")
    TRAIN_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "train_embeddings.pkl")
    TEST_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "test_embeddings.pkl")

    # Create output directory for results
    RESULTS_DIR = os.path.join(DATASET_DIR, "svm_results_fast")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load train and test embeddings
    try:
        train_embeddings, train_labels = load_embeddings(TRAIN_EMB_PATH)
        test_embeddings, test_labels = load_embeddings(TEST_EMB_PATH)

        print(f"Loaded {len(train_embeddings)} train embeddings and {len(test_embeddings)} test embeddings")

        # Get unique emotion classes
        class_names = sorted(np.unique(np.concatenate([train_labels, test_labels])))
        print(f"Emotion classes: {class_names}")

        # Train SVM model with fast approach
        print("\n=== Training Fast SVM Model ===")
        model, scaler = train_svm_model_fast(train_embeddings, train_labels)

        # Evaluate model
        print("\n=== Evaluating Model ===")
        accuracy, report, cm, predictions = evaluate_model(model, scaler, test_embeddings, test_labels)

        # Plot and save confusion matrix
        cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, save_path=cm_path)

        # Save classification report
        report_path = os.path.join(RESULTS_DIR, "classification_report.pkl")
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)

        # Save model
        model_path = os.path.join(RESULTS_DIR, "emotion_svm_model.pkl")
        save_model(model, scaler, class_names, model_path)

        # Save predictions
        pred_path = os.path.join(RESULTS_DIR, "test_predictions.pkl")
        with open(pred_path, 'wb') as f:
            pickle.dump({
                'true_labels': test_labels,
                'predictions': predictions
            }, f)

        print("\n=== Results Summary ===")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Print per-class performance
        print("\nPer-class performance:")
        for emotion in class_names:
            precision = report[emotion]['precision']
            recall = report[emotion]['recall']
            f1 = report[emotion]['f1-score']
            support = report[emotion]['support']

            print(f"{emotion}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")

        # End timing the entire process
        total_time = time() - total_start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")

        print(f"\nAll results saved to {RESULTS_DIR}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've generated the embeddings first using the face embedding script.")

    # Create a simple prediction function for new faces
    prediction_code = """
import numpy as np
import cv2
import face_recognition
import pickle

def predict_emotion_from_image(image_path, model_data_path):
    # Load the model data
    with open(model_data_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    class_names = model_data['class_names']

    # Extract face embedding from image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # For pre-cropped faces, use the entire image
    h, w, _ = rgb_image.shape
    face_location = [(0, w, h, 0)]  # top, right, bottom, left format

    # Generate face encoding
    face_encodings = face_recognition.face_encodings(rgb_image, face_location)

    if not face_encodings:
        return "No face found", 0.0

    face_encoding = face_encodings[0]

    # Preprocess the embedding
    scaled_embedding = scaler.transform([face_encoding])

    # Get prediction
    emotion_label = model.predict(scaled_embedding)[0]

    # Get decision values (not actual probabilities but will work for confidence)
    decision_values = model.decision_function(scaled_embedding)

    # For LinearSVC we don't have probabilities, so we use the decision value
    # normalized across all classes as a confidence proxy
    confidence_proxy = np.exp(decision_values) / np.sum(np.exp(decision_values))
    max_confidence = np.max(confidence_proxy)

    return emotion_label, max_confidence

# Example usage
if __name__ == "__main__":
    # Path to your image and model
    IMAGE_PATH = "path/to/your/image.jpg"
    MODEL_PATH = "path/to/emotion_svm_model.pkl"

    # Predict emotion
    emotion, confidence = predict_emotion_from_image(IMAGE_PATH, MODEL_PATH)
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.4f}")
"""

    # Save the prediction code
    pred_code_path = os.path.join(RESULTS_DIR, "emotion_predictor.py")
    with open(pred_code_path, 'w') as f:
        f.write(prediction_code)

    print(f"Prediction script saved to {pred_code_path}")

if __name__ == "__main__":
    main()





