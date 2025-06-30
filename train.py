import os
import cv2
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuration
DATASET_DIR = "dataset/"
IMG_SIZE = (64, 64)
MODEL_PATH = "mask_detector_model.pkl"
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

def load_data(dataset_dir, img_size=(64, 64)):
    images, labels = [], []
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                try:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img.flatten())
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)

def train_model():
    print("Loading dataset...")
    images, labels = load_data(DATASET_DIR, IMG_SIZE)
    print(f"Loaded {len(images)} images.")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Standardize features
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)

    # Dimensionality reduction with PCA
    print("Applying PCA...")
    pca = PCA(n_components=0.95)
    images_reduced = pca.fit_transform(images_scaled)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images_reduced, labels_encoded, test_size=0.2, random_state=42)

    # Train classifier
    print("Training the model...")
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save the model with pickle
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump((classifier, scaler, pca, le), file)
    print(f"Model saved to {MODEL_PATH}.")

def predict_image(image_path):
    # Load the model with pickle
    print("Loading the model...")
    with open(MODEL_PATH, 'rb') as file:
        classifier, scaler, pca, le = pickle.load(file)

    # Preprocess the input image
    print("Processing the image...")
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()

    # Standardize and reduce dimensions
    img_scaled = scaler.transform([img])
    img_reduced = pca.transform(img_scaled)

    # Predict the class
    prediction = classifier.predict(img_reduced)
    predicted_class = le.inverse_transform(prediction)
    print(f"Prediction: {predicted_class[0]}")

if __name__ == "__main__":
    option = input("Choose an option: (1) Train (2) Predict: ")
    if option == "1":
        train_model()
    elif option == "2":
        image_path = input("Enter image path: ")
        predict_image(image_path)
    else:
        print("Invalid option.")
