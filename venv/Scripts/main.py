import cv2 as cv
import numpy as np

# FACE DETECTION
try:
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
except:
    print('Error loading the face cascade')
    exit(0)
    
# Read the image
img = cv.imread('IMAGES/my_photo.jpg')

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Resize the image (e.g., by half)
scale_percent = 50  # Percent of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Display the output
cv.imshow('img', resized_img)

# Wait for user input to terminate
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
