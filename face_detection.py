import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('guruji.jpg')

# Check if the image has been loaded properly
if img is None:
    print("Error: Image not found.")
    exit()

# Resize the image to make processing faster (optional)
resized_img = cv2.resize(img, (300, 300))

# Convert to grayscale for face detection
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

# Draw rectangles around the faces on the original colored image (not grayscale)
for (x, y, w, h) in faces:
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the resulting image with faces marked
cv2.imshow('Detected Faces', resized_img)

# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()


