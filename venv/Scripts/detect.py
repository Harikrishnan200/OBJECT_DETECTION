import cv2 as cv
import numpy as np

# FACE DETECTION
try:
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
except:
    print('Error loading the face cascade')
    exit(0)

# Start video capture (0 for the default webcam, you can change this if you have multiple webcams)
cap = cv.VideoCapture(0)

# Continuously capture frames from the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Resize the frame (optional)
    scale_percent = 50  # Resize to 50% of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    # Display the resulting frame
    cv.imshow('Face Detection', resized_frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv.destroyAllWindows()
