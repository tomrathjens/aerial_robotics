import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional


############################ Class 1 : ShapeCorners to compute all important infos ############################
class ShapeCorners:
    def __init__(self, corners: List[Tuple[int, int]]):
        self.corners = corners # List of corner points (x, y)
        self._sort_corners() #see fction below
        # Calculate basic properties
        self.center = np.mean(self.corners, axis=0).astype(int)
        self.calculate_dimensions()
        self.area = self.calculate_area()
        
    def _sort_corners(self) -> None: #ordre des aiguilles d'une montre, top left corner first
        center = np.mean(self.corners, axis=0)
        
        # Sort corners by their angle from center
        sorted_corners = sorted(self.corners, key=lambda pt: np.arctan2(pt[1] - center[1], pt[0] - center[0]))
        self.corners = sorted_corners
        
        if len(self.corners) == 4:
            self.top_left = self.corners[0]
            self.top_right = self.corners[1]
            self.bottom_right = self.corners[2]
            self.bottom_left = self.corners[3]
        
    def calculate_dimensions(self) -> None:#for width and height (check again, probably wrong)
        x_coords = [corner[0] for corner in self.corners]
        y_coords = [corner[1] for corner in self.corners]
        self.width = max(x_coords) - min(x_coords)
        self.height = max(y_coords) - min(y_coords)
        # Calculate bounding box
        self.bounding_box = {
            'x_min': min(x_coords),
            'y_min': min(y_coords),
            'x_max': max(x_coords),
            'y_max': max(y_coords)
        }
    
    def calculate_area(self) -> float: #defined as Airmax-airmin/2
        n = len(self.corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.corners[i][0] * self.corners[j][1]
            area -= self.corners[j][0] * self.corners[i][1]
        return abs(area) / 2.0
    
    
    
    def get_center(self) -> Tuple[int, int]:#return coord of center as tuples format
        return tuple(self.center)
    
    def draw_on_image(self, image, color_contour=(0, 255, 0), color_corners=(255, 0, 0), color_center=(0, 0, 255)) -> None:
        contour = np.array(self.corners).reshape(-1, 1, 2)
        cv2.drawContours(image, [contour], 0, color_contour, 3)
        for corner in self.corners:#draws corners
            cv2.circle(image, corner, 5, color_corners, -1)

        cv2.circle(image, tuple(self.center), 5, color_center, -1)# Draws center
    
    def __str__(self) -> str: #return infos about the shape for the user
        return (f"Shape with {len(self.corners)} corners:\n"
                f"- Corners: {self.corners}\n"
                f"- Center: {tuple(self.center)}\n"
                f"- Width x Height: {self.width} x {self.height}\n"
                f"- Area: {self.area:.2f}\n")

############################ Class 2 : ShapeDetector to process everything :  ############################
class ShapeDetector:
    
    def __init__(self):
        self.shapes = []
    
    def detect_shapes(self, image, color_lower, color_upper) -> List[ShapeCorners]:
        self.shapes = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#retr_ext to only take into account the outer contours. 
                                                                              #CHAIN_APPROX_SIMPLE :to output only the 4 corners coordinates
        
        for contour in contours:
            # Approximate the contour to get corners
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            #only take into account if it is a rectangle (4 corners)
            if len(approx) != 4:
                continue
            corners = [tuple(point[0]) for point in approx] #corners becomes tuple format
            ########################################HERE : call previous class##################################    
            shape = ShapeCorners(corners)#CALL CLASS SHAPE CORNERS to compute all infos
            self.shapes.append(shape)
        
        return self.shapes
    
    
    def draw_shapes(self, image) -> None:
        for shape in self.shapes:
            shape.draw_on_image(image)

