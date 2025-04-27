import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import math

from exercises.ex1_pid_control import quadrotor_controller
from exercises.ex2_kalman_filter import kalman_filter as KF
from exercises.ex3_motion_planner import MotionPlanner3D as MP
## Finate state machine initialisation
fence_count = 0
state = None
frame_time = 0
gate_coordinates = []
wait_start_time = None
lookout_position = [[0,3,1,-np.pi/3], [3.5,1,1,np.pi*2/6], [5.5,2,1,np.pi/3], [7,5,1,np.pi*5/6],[5,7,1,np.pi+np.pi/6]] #position to look out for the fence
k=0
count =0
potential_gate_coord_saved=[]
#Personnal thoughts : would like to add : 
# side functions to make the code more robust : 
# - a function so that if fence is too fare <=> fence area too small : first move forwards. 
#like a correction fction : for example if to far or to much rotated (height not small but area small): then move forwards and redo the triangulation
# - the move to the right function to be proportionate to the distance of the fence's center to the center of the taken pictur

# class DroneState:
#     TAKEOFF = "takeoff"
#     SEARCH = "search"
#     APPROACH = "approach"
#     TRIANGULATE = "triangulate"
#     NAVIGATE = "navigate"
#     ALIGN = "align"# to center the drone with the gate before traversal.  
#     TRAVERSE = "traverse"
#     REPOSITION = "reposition"
#     COMPLETED = "completed"
#     EMERGENCY = "emergency"

#mode = "coding" # "coding" or "not coding"
mode = "not coding" # "coding" or "not coding"
def get_command(sensor_data, camera_data, dt):
    global round_counter, state, fence_count, origin_position, current_setpoint, frame_time, gate_coordinates, last_position, last_center, X, wait_start_time, gate_count, k, triangulated_point, max_area, target, count, trajectory_setpoints, trajectory_index, current_gate_index, gate_index, checkpoints,potential_gate_coord,emergency_detection, mode, potential_gate_coord_saved
    
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
    
    if  state == None: #initialisation of the state machine
        state = "takeoff"
        gate_count = 0
        if mode == "coding":
            print("Initialisation of the state machine")
        emergency_detection = "actif"

    # STEP 2 : emergency stop if the drone is too far from the origin
    if current_position[0]>8.5 or current_position[1]>8.5 or current_position[0]<-0.5 or current_position[1]<-0.5 and emergency_detection =="actif": #if the drone is too far from the origin, we need to go back to the origin
        gate_count +=1
        gate_coordinates.append([lookout_position])
        if gate_count ==5:
            state = "Race_mode"
            if mode == "coding":
                print("Emergency stop, Starting Race mode")
            emmergency_detection= "inactif"
        if gate_count <5:
            state = "navigate"
            if mode == "coding":
                print("Emergency stop, going to next gate")
            emmergency_detection = "inactif"
    if state == "takeoff":
        target = [current_position[0],current_position[1],1,current_position[3]]
        control_command, status = move_to_abs_position(current_position,target )
        if status == "reached_target":
            gate_coordinates.append([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']])
            state = "navigate"
            if mode == "coding":
                print("Take off complete")

    if state == "navigate":# Move to the first lookout position
        target = [lookout_position[gate_count][0],lookout_position[gate_count][1],lookout_position[gate_count][2],sensor_data['yaw']]
        control_command, status = move_to_abs_position(current_position, target)
        if status == "reached_target":
            state = "search"
            k=0


    if state == "search": #search for the fence
        emmergency_detection = "actif"
        if shapes and shapes[0].area > 100:
            state = "centering"
            # print("Shape detected with sufficient area:", shapes[0].area)
            last_position = current_position
            last_center = shapes[0].center  # Save the last picture's data for triangulation
            
        elif shapes:
            # print("Shape detected but area too small:", shapes[0].area)
            # Continue rotating to get a better view
            k+=1
            additional_rotation_angle = np.pi/16
            control_command, status = move_to_abs_position(current_position, 
                                                           [lookout_position[gate_count][0],lookout_position[gate_count][1],lookout_position[gate_count][2],lookout_position[gate_count][3]+additional_rotation_angle*k])
        if not shapes:
            # Rotate to the left until a shape is detected
            k+=1
            additional_rotation_angle = np.pi/8
            control_command, status = move_to_abs_position(current_position, 
                                                            [lookout_position[gate_count][0],lookout_position[gate_count][1],lookout_position[gate_count][2],lookout_position[gate_count][3]+additional_rotation_angle*k])

    if state == "centering":
        if shapes:
            shape = shapes[0]
            speed_factor = 1 #speed factor to increase the speed of the drone
            image_center_x = image.shape[1] // 2    
            image_center_y = image.shape[0] // 2
            d_yaw = shape.center[0] - image_center_x
            dz = shape.center[1] - image_center_y
            dx =sensor_data['x_global']
            dy = sensor_data['y_global']
            if shapes[0].get_orientation() > 0.05:
                dir = "right"
                distance = 0.1
                additional_rotation_angle = 0
                dx,dy,_,_ = abs_target_position_calculator(dir, distance, current_position, additional_rotation_angle)
                dx=dx*speed_factor#speed factor is there to increase the speed of the drone
                dy=dy*speed_factor
            if shapes[0].get_orientation() < -0.05:
                dir = "left"
                distance = 0.1
                additional_rotation_angle = 0
                dx,dy,_,_ = abs_target_position_calculator(dir, distance, current_position, additional_rotation_angle)
                dx=dx*speed_factor#speed factor is there to increase the speed of the drone
                dy=dy*speed_factor

            control_command, status = center_drone(current_position, dx,dy, d_yaw, dz, sensor_data)
            max_area = 0 #save the area for traverse fction

            angular_error = 0.0
            if shapes:
                angular_error = abs(shape.get_orientation())

            if -15<d_yaw<15 and -20<dz<20 and angular_error<0.05: 
                last_position = current_position
                last_center = shapes[0].center
                state = "traverse"

    if state == "traverse":
        angular_error = 0.0
        if shapes:
            shape = shapes[0]
            if shape.area < 20000 and shape.area > 6000: #if to close to the fence or too far, never move into centering mode
                angular_error = abs(shape.get_orientation())
                if angular_error > 0.05:
                    state = "centering"
                    if mode == "coding":
                        print("Drone not centered on the fence, going back to centering")

        control_command, status,max_area,count,potential_gate_coord,potential_gate_coord_saved = traverse_detection(current_position,camera_data, max_area,count,mode,potential_gate_coord_saved)
        if status == "reached_target":
            state = "forward"
            gate_count += 1
            gate_coordinates.append(potential_gate_coord)
            # Move forward few meters to be sure to cross the gate
            target = abs_target_position_calculator("forwards", 0.4, current_position, 0)
        

    if state == "forward": 
        control_command, status = move_to_abs_position(current_position, target )
        max_area = 0 #reset the max area for the next gate
        if status == "reached_target":
            if mode == "coding":
                print("reached gate number", gate_count)
            state = "navigate"
            if gate_count == 5:
                state = "Race_mode"
                if mode == "coding":
                    print("All gates traversed, Racing begin")
            

################################ PART 3 : Race mode ###############################################################
    if state == "Race_mode":
        state = "racing part1 : origin position"
        #extract the coordinates of the gates from the list of coordinates
        gate_coordinates2=[gate_coordinates[0], gate_coordinates[1], gate_coordinates[2], gate_coordinates[3], gate_coordinates[4],gate_coordinates[5], gate_coordinates[0]]
        #export as text file
        np.savetxt("gate_coordinates2.txt", gate_coordinates2, delimiter=",", fmt="%.6f")
        #generate the spline path with yaw
        checkpoints = generate_spline_path_with_yaw(gate_coordinates2, num_points=30)

    if state == "racing part1 : origin position":
        target = [gate_coordinates[0][0], gate_coordinates[0][1], gate_coordinates[0][2], 0]
        control_command, status = move_to_abs_position(current_position, target)
        if status == "reached_target":
            state = "racing part2 : Lap1"
            gate_index = 0
            if mode == "coding":
                print("Reached origin position")
            

    if state == "racing part2 : Lap1":
        emmergency_detection = "actif"
        if gate_index < len(checkpoints):
            target = [checkpoints[gate_index][0], checkpoints[gate_index][1], checkpoints[gate_index][2], checkpoints[gate_index][3]]
            a=2
            if gate_index < len(checkpoints)-a:
                next_target = [checkpoints[gate_index+a][0], checkpoints[gate_index+a][1], checkpoints[gate_index+a][2], checkpoints[gate_index+a][3]]
            else:
                next_target = target
            control_command, status = move_to_abs_positionrace(current_position, target,next_target)
            
            if status == "reached_target":
                state = "racing part2 : Lap1"
                gate_index += 1
                
        elif gate_index >= len(checkpoints):
            state = "racing part1 : origin position"
            gate_index = 0
            if mode == "coding":
                print("Lap 1 completed, starting Lap 2")





###############################################################
                ##############################
     # Display camera feed with annotations

    if mode == "coding":
        cv2.putText(image, f"State: {state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Gate: {gate_count}/{5}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        try:
            if angular_error:
                cv2.putText(image, f"Angular Error: {angular_error:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except NameError:
            pass  # angular_error is not defined

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
    def get_orientation(self) -> float:#get the orientation of the shape in radians relative to the drone. 0rad = drone perfectly facing the shape
        dx = self.top_right[0] - self.top_left[0]
        dy = self.top_right[1] - self.top_left[1]
        angle_rad = np.arctan2(dy, dx)
        return angle_rad

    def calculate_area(self) -> float: 
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
    
    def __str__(self) -> str:
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
            # self.shapes.sort(key=lambda s: s.center[0], reverse=True)
            #if area difference is too small, sort the one most to the right
            self.shapes.sort(key=lambda s: s.area, reverse=True) #BETTER TO SORT WITH HIGHEST AREA
            if abs(self.shapes[0].area - self.shapes[1].area) < 3000:
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
def move_to_abs_position(current_position, target_position):
    status = "moving"
    distancex_y = math.sqrt((target_position[0] - current_position[0])**2 + 
                        (target_position[1] - current_position[1])**2) 
    distancez = abs(target_position[2] - current_position[2])
    angle_error = abs(target_position[3]-current_position[3])
    if(distancex_y > 0.2) or (distancez > 0.01) or (angle_error > np.pi/20): #if we are not close to the target position, we can move to it
        # Move towards the target position
        control_command = [target_position[0], target_position[1], target_position[2], target_position[3]] #move to the center of the fence
        return control_command, status 

    else:
        status = "reached_target"
        control_command = [current_position[0], current_position[1], current_position[2], current_position[3]] #stay in the same position
        #print("Reached target position")
    
    
    return control_command, status 

################################### Move to direction #######################################################
def abs_target_position_calculator(dir, distance, last_position, additional_rotation_angle):
    # Calculate the target position based on the current position and wanted direction
    if dir == "forwards":
        dir = 0
    elif dir == "backwards":
        dir = -np.pi
    elif dir == "left":
        dir = np.pi*3/2
    elif dir == "right":
        dir = +np.pi/2
    target_position = np.array([last_position[0] + distance * math.cos(last_position[3]-dir),
                                last_position[1] + distance * math.sin(last_position[3]-dir),
                                last_position[2],
                                last_position[3]+additional_rotation_angle]) #add the additional rotation angle to the yaw
    
    return target_position


################################## yaw rotation function : #######################################
def yaw_rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0],
                     [math.sin(yaw),  math.cos(yaw), 0],
                     [            0,              0, 1]])

################################## Centering  on target function : #######################################
def center_drone(current_position,dx,dy, d_yaw, dz, sensor_data):
    gain_x = 0.05
    gain_y = 0.05
    gain_yaw = 0.01
    gain_z = 0.01

    target_x =  dx
    target_y = dy
    target_z = - dz*gain_z + sensor_data['z_global']  # Move up/down based on dy
    target_yaw = -d_yaw*gain_yaw + sensor_data['yaw']

    target = [target_x, target_y, target_z, target_yaw]

    control_command, status = move_to_abs_position(current_position, target)

    return control_command, status

################################## Traverse function : #######################################
def traverse_detection(current_position, camera_data, max_area, count, mode, potential_gate_coord_saved):
    potential_gate_coord = [current_position[0], current_position[1], current_position[2], current_position[3]]

    step_fwd = 0.20
    gain_lat = 0.002
    gain_alt = 0.001
    gain_yaw = 0.0025
    max_lat = 0.25
    max_alt = 0.15
    min_area = 800
    image = camera_data.copy()
    shapes = ShapeDetector().detect_shapes(image)

    # Move forward to the center of the fence
    yaw = current_position[3]
    step_world = np.array([step_fwd * math.cos(yaw),
                          step_fwd * math.sin(yaw),
                          0.0])

    lat_body = alt_body = d_yaw = 0.0

    status = "moving"
    
    # If shapes detected, update max_area and save position when area is significant
    if shapes and shapes[0].area > min_area:
        # Reset count when shapes are detected
        count = 0
        
        # Track the position with maximum area as a potential gate coordinate
        if max_area < shapes[0].area:
            max_area = shapes[0].area
            potential_gate_coord_saved = [current_position[0], current_position[1], current_position[2], current_position[3]]
            if mode == "coding":
                print(f"Updated potential gate coordinate, area: {max_area}")
        
        # Centering function
        img_cx = image.shape[1] / 2
        img_cy = image.shape[0] / 2
        dx_px = shapes[0].center[0] - img_cx    
        dy_px = shapes[0].center[1] - img_cy     

        lat_body = -dx_px * gain_lat
        alt_body = -dy_px * gain_alt
        d_yaw = -dx_px * gain_yaw

        # Limit to max values
        lat_body = np.clip(lat_body, -max_lat, max_lat)
        alt_body = np.clip(alt_body, -max_alt, max_alt)

        step_world += np.array([
            lat_body * -math.sin(yaw),
            lat_body * math.cos(yaw),
            alt_body])
        
        if shapes[0].area < max_area * 0.4 and max_area > 5000:  # Add minimum threshold for max_area
            status = "reached_target"
            potential_gate_coord = potential_gate_coord_saved
            if mode == "coding":
                print(f"Passed through gate, using saved coordinates: {potential_gate_coord_saved}")
    else:
        # No shapes detected
        count += 1
        if count > 400:
            status = "reached_target"
            # Use the last saved coordinates if available
            if potential_gate_coord_saved:
                potential_gate_coord = potential_gate_coord_saved
            else:
                potential_gate_coord = [current_position[0], current_position[1], current_position[2], current_position[3]]
                
            if mode == "coding":
                print("No shape detected for a while, moving to next gate")
                print(f"Using coordinates: {potential_gate_coord}")

    # Calculate target position for movement
    target = [current_position[0] + step_world[0],
              current_position[1] + step_world[1],
              current_position[2] + step_world[2],
              yaw + d_yaw]
              
    control_command, _ = move_to_abs_position(current_position, target)

    return control_command, status, max_area, count, potential_gate_coord, potential_gate_coord_saved
################################################ compute spline *###########################################
def generate_spline_path_with_yaw(gate_coordinates: List[List[float]], num_points: int = 20):
    def catmull_rom_segment(p0, p1, p2, p3, num_points=20):
        points = []
        for t in np.linspace(0, 1, num_points):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
            points.append(point)
        return np.array(points)

    def compute_yaws_from_curve(curve):
        deltas = np.diff(curve, axis=0)
        yaws = np.arctan2(deltas[:, 1], deltas[:, 0])
        yaws = np.append(yaws, yaws[-1])
        return yaws

    positions = np.array(gate_coordinates)[:, :3]
    padded = np.vstack([
        positions[0] - (positions[1] - positions[0]),
        positions,
        positions[-1] + (positions[-1] - positions[-2])
    ])
    curve = []
    for i in range(len(padded) - 3):
        segment = catmull_rom_segment(
            padded[i], padded[i+1], padded[i+2], padded[i+3],
            num_points=num_points
        )
        curve.append(segment)
    curve = np.vstack(curve)
    yaws_curve = compute_yaws_from_curve(curve)
    curve_with_yaw = np.hstack([curve, yaws_curve[:, np.newaxis]])
    return curve_with_yaw


################################ Race mode function : #######################################
def move_to_abs_positionrace(current_position, target_position,next_target):
    status = "moving"
    distancex_y = math.sqrt((target_position[0] - current_position[0])**2 + 
                        (target_position[1] - current_position[1])**2) 
    distancez = abs(target_position[2] - current_position[2])
    angle_error = abs(target_position[3]-current_position[3])
    if(distancex_y > 0.5) or (distancez > 0.5) : #if we are not close to the target position, we can move to it
        # Move towards the target position
        control_command = [next_target[0], next_target[1], next_target[2], next_target[3]] #move to the center of the fence
        return control_command, status 

    else:
        status = "reached_target"
        control_command = [current_position[0], current_position[1], current_position[2], current_position[3]] #stay in the same position
        #print("Reached target position")
    
    
    return control_command, status 