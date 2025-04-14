import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from class_fences_detection import ShapeCorners
from class_fences_detection import ShapeDetector



image = cv2.imread('controllers/main/assignment/example_picture_2.png')

# Define color range for purple
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([170, 255, 255])

# Create shape detector and detect shapes
detector = ShapeDetector()
shapes = detector.detect_shapes(image, lower_purple, upper_purple)

# Print information about each shape
for i, shape in enumerate(shapes):
    print(f"Shape {i+1}:")
    print(shape)
    print()

# Draw shapes on image
detector.draw_shapes(image)
# Display result
cv2.imshow('Detected Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()