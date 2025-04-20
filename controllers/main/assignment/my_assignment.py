import numpy as np
import time
import cv2
from typing import List, Tuple, Dict, Optional
import math

from exercises.ex1_pid_control import quadrotor_controller
from exercises.ex2_kalman_filter import kalman_filter as KF
from exercises.ex3_motion_planner import MotionPlanner3D as MP
## Finate state machine initialisation
fence_count = 0
state = "ground"
frame_time = 0
gate_coordinates = []
wait_start_time = None

def get_command(sensor_data, camera_data, dt):
    global round_counter, state, fence_count, origin_position, current_setpoint, frame_time, gate_coordinates, last_position, last_center,X, wait_start_time
    
    ############################## PART 1 : opencv : continuous fences detection ##################################################
    # Create fence detector and search for fences
    image = camera_data.copy() 
    detector = ShapeDetector()
    shapes = detector.detect_shapes(image)
    detector.draw_shapes(image)# To draw shapes on image

    #############################PART 2 : FSM to control the drone : #######################################################
    #STEP 0 : Initialisation : basic command input and last sensor output
    current_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
    control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']] 

    # STEP 1 : Take off
    if  state == "ground":
        control_command, status = move_to_position(current_position,[current_position[0],current_position[1],1,current_position[3]] )
        if status == "reached_target":
            #save the origin position
            gate_coordinates.append([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']])
            state = "step2_triangulation1"
            last_position = current_position
            last_center = shapes[0].center #saves the last picture's data for the triangulation
            print("Take off complete")
    
    # STEP 2 : Move to the first fence
    if state == "step2_triangulation1":
        offset = [-0.4, 0, 0, np.pi/4] #move to the right position of the fence
        control_command, status = move_to_position(current_position,np.add(last_position, offset))        
        if status == "reached_target":
            state = "step2_triangulation2"

    if state == "step2_triangulation2":
        if wait_start_time is None:        #wait few seconds to let the drone stabilize
                wait_start_time = time.time()  # Start waiting
                print("Reached first fence. Waiting to stabilize...")

        elif time.time() - wait_start_time >= 1.0:  # wait 2 seconds
            state = "round1"
            wait_start_time = None 
            current_center = shapes[0].center if shapes else None
            if current_center is not None:
                print("Current center:", current_center)
                
                # --- Proceed with triangulation --- last_center and current_center are the two pixel observations
                triangulated_point = triangulate_from_pixels(last_center, current_center, image, last_position, current_position)
                gate_coordinates.append([triangulated_point[0], triangulated_point[1], triangulated_point[2], sensor_data['yaw']])
                print("Triangulated 3D point:", triangulated_point)
                state = "round1"
                # u1p, v1p, reproj_err = reprojection_error(X, R1, t1, K, last_center)
                # print(f"Reprojected: ({u1p:.1f}, {v1p:.1f}),   error = {reproj_err:.2f} px")

    
    if state == "round1":
        
        control_command, status = move_to_position(current_position,[gate_coordinates[1][0],gate_coordinates[1][1],gate_coordinates[1][2], gate_coordinates[1][3]] )
        if status == "reached_target":
            print("reached gate 1")
            state = "triangulation 2"
            last_position = current_position
            
    if state == "triangulation 2":
        offset = [0, 0, 0, np.pi/2] #move to the right position of the fence
        control_command, status = move_to_position(current_position,np.add(last_position, offset))
        if status == "reached_target":
            state = "round2"
            print("Reached second fence. Waiting to stabilize...")
    


    cv2.imshow("Crazyflie FPV Camera", image)
    cv2.waitKey(1) 
    return control_command


#########################################_ADDITIONAL FUNCTIONS : ################################

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
        
    # Range for our purple color in hsv
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    color_lower = lower_purple
    color_upper = upper_purple

    def __init__(self):
        self.shapes = []
    
    def detect_shapes(self, image) -> List[ShapeCorners]:
        self.shapes = []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4:
                continue
            corners = [tuple(point[0]) for point in approx]
            shape = ShapeCorners(corners)
            self.shapes.append(shape)
        
        # If multiple shapes found, return the one with highest x-coordinate (rightmost)
        if len(self.shapes) > 1:
            self.shapes.sort(key=lambda s: s.center[0], reverse=True)
            return [self.shapes[0]]  # Return only the rightmost shape as a list
        
        return self.shapes
    
    def draw_shapes(self, image) -> None:
        for shape in self.shapes:
            shape.draw_on_image(image)


##############################_ADDITIONAL FUNCTIONS : ################################
##############################triangulation function : #######################################
#TO triangulate a 3D point from two pixel observations (pix1 and pix2).
# pix1, pix2 : Pixel coords (u,v) in each image = same pixel we want to interpolate
# K : Intrinsic matrix: [[fx, 0, u0],
#                        [0, fy, v0],   
#                        [0,  0,  1]]
# R1, R2 : Rotation matrix from camera frame to world frame.
# t1, t2 : translation vect of camera centers to world coordinates.

def triangulate_from_pixels(pix1, pix2, image, last_position,current_position):
    #variable initialisation :
    fc = 161.013922282
    u0, v0 = image.shape[1]/2, image.shape[0]/2 
    K = np.array([[fc,  0, u0],
                [ 0, fc, v0],
                [ 0,  0,  1]])

    R1_body = yaw_rotation(last_position[3])
    R2_body = yaw_rotation(current_position[3])
    R_cam_to_body = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]])
    R1 = R1_body @ R_cam_to_body
    R2 = R2_body @ R_cam_to_body

    cam_offset_body = np.array([0.03, 0, 0.01])
    t1 = np.array(last_position[:3]) + R1_body @ cam_offset_body
    t2 = np.array(current_position[:3]) + R2_body @ cam_offset_body

    # back‐project pixels into rays in camera frame:
    invK = np.linalg.inv(K)
    u1, v1 = pix1
    u2, v2 = pix2

    # homogeneous
    h1 = np.array([u1, v1, 1.0])
    h2 = np.array([u2, v2, 1.0])

    # direction in cam frame (unnormalized)
    d1_cam = invK @ h1
    d2_cam = invK @ h2

    # normalize 
    d1_cam /= np.linalg.norm(d1_cam)
    d2_cam /= np.linalg.norm(d2_cam)

    r = R1 @ d1_cam
    s = R2 @ d2_cam

    # origins P and Q
    P = np.asarray(t1, dtype=float)
    Q = np.asarray(t2, dtype=float)
    PQ = Q - P

    # build the 2×2 system from (F−G)·r = 0 and (F−G)·s = 0 
    rr = r.dot(r)
    ss = s.dot(s)
    rs = r.dot(s)

    A = np.array([[ rr, -rs ],
                  [ rs, -ss ]])
    b = np.array([ PQ.dot(r),
                   PQ.dot(s) ])

    # solve for λ, μ 
    lam, mu = np.linalg.solve(A, b)

    # compute the closest points on each ray 
    F = P + lam * r
    G = Q + mu  * s

    # compute midpoint = triangulated point 
    X = (F + G) / 2.0
    return X

############################## Move to position function : #######################################
def move_to_position(current_position, target_position):
    status = "moving"
    # Calculate Euclidean distance using only x, y, z coordinates (ignoring yaw)
    distancex_y = math.sqrt((target_position[0] - current_position[0])**2 + 
                        (target_position[1] - current_position[1])**2) 
    distancez = abs(target_position[2] - current_position[2])
    angle_error = abs(target_position[3]-current_position[3])
    if(distancex_y > 0.2) or (distancez > 0.01) or (angle_error > np.pi/8): #if we are not close to the target position, we can move to it
        # Move towards the target position
        control_command = [target_position[0], target_position[1], target_position[2], target_position[3]] #move to the center of the fence

        return control_command, status 

    else:
        status = "reached_target"
        control_command = [current_position[0], current_position[1], current_position[2], current_position[3]] #stay in the same position
        print("Reached target position")
    
    
    return control_command, status 


################################## yaw rotation function : #######################################
def yaw_rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0],
                     [math.sin(yaw),  math.cos(yaw), 0],
                     [            0,              0, 1]])


################################### test #######################################################