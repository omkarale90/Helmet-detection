import cv2
import numpy as np

# Read and resize image
img = cv2.imread('leaf.jpg')
resized = cv2.resize(img, (600, 400))

# Convert to HSV
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# Define HSV range for detecting disease (yellow/brown regions typically)
lower = np.array([10, 50, 50])
upper = np.array([30, 255, 255])

# Create mask and apply
mask = cv2.inRange(hsv, lower, upper)
diseased = cv2.bitwise_and(resized, resized, mask=mask)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
output = resized.copy()
cv2.drawContours(output, contours, -1, (0, 0, 225), 3)

# Show results
cv2.imshow('Original Leaf', resized)
cv2.imshow('Mask - Disease Detection', mask)
cv2.imshow('Detected Disease Area', diseased)
cv2.imshow('Contours', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
