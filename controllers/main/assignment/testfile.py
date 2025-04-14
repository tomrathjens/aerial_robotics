import cv2
import numpy as np
#import function detect_gates from detect_gates.py
from detect_gates import detect_gates


def test_detect_gates(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image!")
        return

    gates = detect_gates(image)
    print("Detected gates (x, y, w, h):", gates)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with a sample image
gates = test_detect_gates("controllers/main/assignment/example_picture.png")

# Check if any gates were found
if gates:
    print(f"Detected {len(gates)} gate(s):")
    for i, (x, y, w, h) in enumerate(gates, 1):
        print(f"Gate {i}: x={x}, y={y}, width={w}, height={h}")
# else:
#     print("No gates detected")