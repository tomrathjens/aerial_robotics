import cv2
import numpy as np

image = cv2.imread('controllers/main/assignment/example_picture_2.png')


#create me a class for the corners of the purple shape
class Corners:
    def __init__(self, corners):
        self.corners = corners
        self.center = np.mean(corners, axis=0).astype(int)  # Calculate the center point
        self.width = abs(corners[0][0] - corners[1][0])
        self.height = abs(corners[0][1] - corners[2][1])

    def get_corners(self):
        return self.corners

# HSV : to isolate better colors (hue = actual colors, saturation = intensity, value = brightness)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_purple = np.array([130, 50, 50])
upper_purple = np.array([170, 255, 255])
mask = cv2.inRange(hsv, lower_purple, upper_purple)

# Find contours with purple mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL : outputs only the extreme outer contours
                                                                                 # CHAIN_APPROX_SIMPLE :to output only 4 corners coordinates
# Draw contours and extract corner points
for i in contours:
    # 1. Check if it's a fence
    epsilon = 0.02 * cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, epsilon, True)
    if len(approx) == 4:  # Rectangle (4 corners)

        corners = [tuple(i[0]) for i in approx]  #i reprensents one corner point
        print("Corners of the purple shape:", corners)

        #draw
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)#draw the edges
        for corner in corners:
            cv2.circle(image, corner, 5, (255, 0, 0), -1)#draw the corners

        #find the center point of the fence : 
        center = np.mean(corners, axis=0).astype(int)
        print("Center of the purple shape:", center)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)  # Draw center point in red








cv2.imshow('Detected Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()