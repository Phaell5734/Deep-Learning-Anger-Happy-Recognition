import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your emotion classifier model
emotion_model = load_model('C:/Users/frat5/Desktop/jupyter/best_model.keras')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each face in the frame
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=3)

        # Predict the emotion of the face
        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label = np.argmax(emotion_prediction)
        emotion_text = "Sad" if emotion_label == 1 else "Happy"  # Modify according to your model's label

        # Put the emotion text on the frame
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv2.destroyAllWindows()