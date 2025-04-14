import cv2
import numpy as np

def detect_gates(image):
    """
    Detect purple gates in the image
    Returns: [x, y, width, height] coordinates of detected gates
    """
    # Convert to HSV color space (better for color detection)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Remove alpha channel if present
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    
    #range of purple
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    
    # Create a mask for purple color
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gates=[]   
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            # Get rotated rectangle (center, size, angle)
            rot_rect = cv2.minAreaRect(contour)
            
            # Get 4 corner points of the rotated rectangle
            box_points = cv2.boxPoints(rot_rect)
            box_points = np.int0(box_points)  # Convert to integers
            
            gates.append(box_points)  # Stores all 4 corners
            

    # Display a green square around detected gates 
    result_img = image.copy()
    # Draw lines between them
    for gate in gates:  # Loop through each detected gate
        # Draw lines between consecutive corners of THIS gate
        for i in range(4):
            cv2.line(result_img, 
                    tuple(gate[i]),  # Current corner (convert to tuple)
                    tuple(gate[(i + 1) % 4]),  # Next corner (with wrap-around)
                    (0, 255, 0),  # Green color
                    2)  # Line thickness

    cv2.imshow("Gate Detection", result_img)
    
    return gates