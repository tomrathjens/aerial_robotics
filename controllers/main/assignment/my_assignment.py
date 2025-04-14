import numpy as np
import time
import cv2
# from detect_gates import detect_gates
#import detect_gates


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

def get_command(sensor_data, camera_data, dt):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    # ---- YOUR CODE HERE ----
    control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]


    detect_gates(camera_data)
    
    return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

import cv2
import numpy as np

def test_detect_gates(image_path):
    # 1. Load the test image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image!")
        return
    
    # 2. Call detect_gates()
    gates = detect_gates(image)
    print("Detected gates (x, y, w, h):", gates)
    
    # 3. Wait for key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with a sample image
test_detect_gates("test_gate.jpg")  # Replace with your image path
