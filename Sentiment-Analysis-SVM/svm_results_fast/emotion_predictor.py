
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
