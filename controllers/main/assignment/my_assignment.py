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


##note : 
#control_command = A list of 4 control commands [vel_x, vel_y, altitude, yaw_rate] in the body reference frame


def get_command(sensor_data, camera_data, dt):
    global round_counter, state, fence_count, origin_position, current_setpoint, frame_time, gate_coordinates, last_position, last_center
    current_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
    
        ########################PART 1 : opencv : continuous fences detection##################################################
    
    image = camera_data.copy() 
    
    # Range for our purple color in hsv
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])

    # Create fence detector and search for fences
    detector = ShapeDetector()
    shapes = detector.detect_shapes(image, lower_purple, upper_purple)
    
    # Draw shapes on image
    detector.draw_shapes(image)
    #############################PART 2 : Fsm to control the drone : #######################################################
    #STEP 0 : basic commands to control the drone
    control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']] #stay in the same position
    # STEP 1 : Take off
    if  state == "ground":
        control_command, status = move_to_position(current_position,[current_position[0],current_position[1],1,current_position[3]] )
        if status == "reached_target":
            print("Take off complete")
            state = "start"
            last_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            #save the last picture's data for the triangulation
            last_center = shapes[0].center
            print("Last center:", last_center)
    

    
    #if sensor_data['z_global'] > 0.49 : state = "start"


    # STEP 2 : Move to the first fence
    if state == "start":
        offset = [-0.5, 0, 0, 0] #move to the right position of the fence
        control_command, status = move_to_position(current_position,np.add(last_position, offset))
        if status == "reached_target":
            print("Reached first fence")
            state = "round1"
            #
            current_center = shapes[0].center
            print("Current center:", current_center)
            #triangulation :
            ## triangulation : 
                # camera intrinsics
            fc=161.013922282 #focal length

            s_x=0.5 #pixel size in x direction
            s_y=0.5 #pixel size in y direction
            u0, v0 = 320, 240
            K = np.array([[fc/s_x,  0, u0],
                        [ 0, fc/s_y, v0],
                        [ 0,  0,  1]])

            # two pixel observations
            pix1 = last_center
            pix2 = current_center
            print("pix1:", pix1)
            print("pix2:", pix2)

            # example extrinsics (identity + small baseline)#r1=R2 and t1=t2
            # R1 = np.array([[0,-1,0],
            #             [0,0,-1],
            #             [1,0,0]]); 
            #t1 = np.array([-0.03,0,-0.01])#since zcam = xdrone, xcam = −ydrone, ycam = −zdrone. (needs to be checked later)
            #R2 = R1; t2 = t1
            R1 = np.eye(3)
            t1 = np.array([0, 0, 0])
            R2 = np.eye(3)
            t2 = np.array([-0.03, 0, 0]) #small baseline

            X = triangulate_from_pixels(pix1, pix2, K, R1, t1, R2, t2)
            print("Triangulated 3D point:", X)


    


        

    # if state == "start":
    #     if sensor_data['z_global'] > 0.48:
    #         print("Take off")
    #         origin_position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
    #         print("Origin position:", origin_position)
    #         state ="round1"

    # if state == "round1" and fence_count == 0:
    #     # Check if there are any shapes detected
    #     if shapes:
    #         # Get the first shape detected
    #         gate_coordinates.append(shapes[0])
            
    #         # move to the right position of the fence
    #         #control_command = [4, 4, 1.0, sensor_data['yaw']] #move to the center of the fence
    #         control_command = [sensor_data['x_global'], sensor_data['y_global']+2,sensor_data['z_global'], sensor_data['yaw']]
    #         #if reached the position we want to go to, we can start the triangulation
    #         fence_count = 1
    #         return control_command, camera_data #TKKK remove camera_data when submiting

    # if state == "round1" and fence_count ==1:
            
    #         detector = ShapeDetector()
    #         shapes = detector.detect_shapes(image, lower_purple, upper_purple)
    #         gate_coordinates.append(shapes[0])
    #         # Draw shapes on image

    #         ## triangulation : 
    #             # camera intrinsics
    #         fc=161.013922282 #focal length

    #         s_x=0.5 #pixel size in x direction
    #         s_y=0.5 #pixel size in y direction
    #         u0, v0 = 320, 240
    #         K = np.array([[fc/s_x,  0, u0],
    #                     [ 0, fc/s_y, v0],
    #                     [ 0,  0,  1]])

    #         # two pixel observations
    #         pix1 = np.array([gate_coordinates[0].center])
    #         pix2 = np.array([gate_coordinates[1].center])
    #         print("pix1:", pix1)
    #         print("pix2:", pix2)

    #         # example extrinsics (identity + small baseline)#r1=R2 and t1=t2
    #         R1 = np.array([[0,-1,0],
    #                     [0,0,-1],
    #                     [1,0,0]]); 
    #         t1 = np.array([-0.03,0,-0.01])#since zcam = xdrone, xcam = −ydrone, ycam = −zdrone. (needs to be checked later)
    #         R2 = R1; t2 = t1

    #         X = triangulate_from_pixels(pix1, pix2, K, R1, t1, R2, t2)
    #         print("Triangulated 3D point:", X)
    #         control_command = [sensor_data['x_global'], sensor_data['y_global'],sensor_data['z_global'], sensor_data['yaw']]

            
    camera_data_processed = image.copy()
    # Display images at approximately 100 fps
    # Calculate time to maintain frame rate
    # frame_time =frame_time +1
    cv2.imshow("Crazyflie FPV Camera", image)
    cv2.waitKey(1) 
    # if frame_time > 30:
        
        # frame_time =0
    



    return control_command, camera_data_processed #TKK at the end remove camera_dara_processed attention !!#camera_data # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


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


##############################_ADDITIONAL FUNCTIONS : ################################
##############################triangulation function : #######################################
import numpy as np

def triangulate_from_pixels(
    pix1,           # (u1, v1) in image 1
    pix2,           # (u2, v2) in image 2
    K,              # 3×3 intrinsic matrix
    R1, t1,         # rotation (3×3) and translation (3,) of cam 1 (cam→world)
    R2, t2          # rotation and translation of cam 2
):
    """
    Triangulate a 3D point from two pixel observations.
    
    Params
    ------
    pix1, pix2 : array‐like, shape (2,)
        Pixel coords (u,v) in each image.
    K : array‐like, shape (3,3)
        Intrinsic matrix: [[fx, 0, u0],
                           [0, fy, v0],
                           [0,  0,  1]]
    R1, R2 : array‐like, shape (3,3)
        Rotation from camera frame to world frame.
    t1, t2 : array‐like, shape (3,)
        Camera centers in world coordinates.

    Returns
    -------
    X : ndarray, shape (3,)
        Estimated 3D point in world coords.
    """

    # 1) back‐project pixels into rays in camera frame:
    invK = np.linalg.inv(K)
    u1, v1 = pix1
    u2, v2 = pix2

    # homogeneous
    h1 = np.array([u1, v1, 1.0])
    h2 = np.array([u2, v2, 1.0])

    # direction in cam frame (unnormalized)
    d1_cam = invK @ h1
    d2_cam = invK @ h2

    # normalize (optional but numerically stable)
    d1_cam /= np.linalg.norm(d1_cam)
    d2_cam /= np.linalg.norm(d2_cam)

    # 2) rotate rays into world frame (r,s in eqns 2–3)
    r = R1 @ d1_cam
    s = R2 @ d2_cam

    # origins P and Q
    P = np.asarray(t1, dtype=float)
    Q = np.asarray(t2, dtype=float)
    PQ = Q - P

    # build the 2×2 system from (F−G)·r = 0 and (F−G)·s = 0 (eqns 6–7)
    rr = r.dot(r)
    ss = s.dot(s)
    rs = r.dot(s)

    A = np.array([[ rr, -rs ],
                  [ rs, -ss ]])
    b = np.array([ PQ.dot(r),
                   PQ.dot(s) ])

    # solve for λ, μ (eqn 9)
    lam, mu = np.linalg.solve(A, b)

    # 3) compute the closest points on each ray (eqns 4–5)
    F = P + lam * r
    G = Q + mu  * s

    # 4) midpoint = triangulated point (eqn 8)
    X = (F + G) / 2.0
    return X



############################## Move to position function : #######################################
def move_to_position(current_position, target_position):
    status = "moving"
    # Calculate Euclidean distance using only x, y, z coordinates (ignoring yaw)
    distancex_y = math.sqrt((target_position[0] - current_position[0])**2 + 
                        (target_position[1] - current_position[1])**2) 
    distancez = abs(target_position[2] - current_position[2])
    if(distancex_y > 0.2) or (distancez > 0.01): #if we are not close to the target position, we can move to it
        # Move towards the target position
        control_command = [target_position[0], target_position[1], target_position[2], target_position[3]] #move to the center of the fence

        return control_command, status 

    else:
        status = "reached_target"
        control_command = [current_position[0], current_position[1], current_position[2], current_position[3]] #stay in the same position
        print("Reached target position")
    
    
    return control_command, status 

