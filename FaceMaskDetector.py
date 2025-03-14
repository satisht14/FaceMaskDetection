import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model

# Dataset Path
train_dir = r"C:\Users\SATISH\OneDrive\Desktop\SatishPro\Face Mask Dataset\Train"
test_dir = r"C:\Users\SATISH\OneDrive\Desktop\SatishPro\Face Mask Dataset\Test"

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.1, shear_range=0.1, horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=16, class_mode='binary')
test_generator = datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=16, class_mode='binary')

# Build CNN Model (Lighter Architecture)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model (Only 3 Epochs)
history = model.fit(train_generator, epochs=3, steps_per_epoch=10, validation_data=test_generator, validation_steps=5)

# Evaluate the Model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the Model
model.save(r"C:\Users\SATISH\OneDrive\Desktop\SatishPro\face_mask_detector.h5")

# Load the trained model
model_path = r"C:\Users\SATISH\OneDrive\Desktop\SatishPro\face_mask_detector.h5"
model = tf.keras.models.load_model(model_path)

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change to '1' or '2' if using an external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        face = cv2.resize(face, (64, 64))  # Resize to match model input
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict mask or no mask
        prediction = model.predict(face)[0][0]

        # Define label and bounding box color
        label = "Mask" if prediction > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box and label on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the video frame with detections
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
