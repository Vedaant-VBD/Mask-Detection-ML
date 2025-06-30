import cv2
import numpy as np
import pickle  

# Configuration
MODEL_PATH = "mask_detector_model.pkl"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["mask", "no_mask", "mask_worn_incorrectly"]

# Load the trained model and preprocessing tools
print("Loading the model...")
with open(MODEL_PATH, 'rb') as file:
    classifier, scaler, pca, label_encoder = pickle.load(file)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE).flatten()
    scaled = scaler.transform([resized])
    reduced = pca.transform(scaled)
    return reduced

print("Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_frame(face_roi)
        prediction = classifier.predict(processed_face)
        confidence = classifier.predict_proba(processed_face).max()
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        label = f"{predicted_class} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if predicted_class == "mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
