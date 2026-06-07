import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello


# Data source management #

class KittiSource:
    def __init__(self, sequence_path):
        self.img_dir = os.path.join(sequence_path, "image_0")
        self.images = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)])
        self.idx = 0
        
        # loading ground truths
        
        gt_path = os.path.join(sequence_path, "poses.txt")
        self.gt_poses = np.loadtxt(gt_path, dtype=float) if os.path.exists(gt_path) else None
        
        # KITTI Intrinsics
        self.K = np.array([[718.856,   0.0,   607.1928],
                           [  0.0,   718.856, 185.2157],
                           [  0.0,     0.0,     1.0]])
        
    def get_frame(self):
        if self.idx >= len(self.images):
            return None
        frame = cv2.imread(self.images[self.idx])
        self.idx += 1
        return frame
    
    def get_scale(self):
        """ this is merely to test the loop and see if it works since in the Kitti dataset we dont have any IMU data 
        instead we will use the group truths for the true scale and use it to test the loop. obviously this defeats 
        the purpose of odometry hence this should only be used to debugging the loop."""
        if self.gt_poses is None or self.idx < 2:
            return 1.0
        
        # data comes in a 12 number string and will be reshaped into a matrix with last col representing estimated scale
        p1 = self.gt_poses[self.idx -2].reshape(3, 4)[:, 3]
        p2 = self.gt_poses[self.idx -1].reshape(3, 4)[:, 3]
        return float(np.linalg.norm(p2 - p1))
    
class TelloSource:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        time.sleep(2.0) # delay for camera
        
        self.frame_reader = self.tello.get_frame_read()
        
        # Tello instrinsics based of research from paper
        self.K = np.array([[365.9667,   0.0,    213.3087],
                           [  0.0,    496.2820, 225.1782],
                           [  0.0,      0.0,      1.0]])
        
        self.last_time = time.time()
        
    def get_frame(self):
        frame = self.frame_reader.frame
        return frame
    
    def get_scale(self):
        """Using IMU data from Tello drone to estimate the scale"""
        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        # getting velocity in cm/s from IMU sensors
        vx = self.tello.get_velocity_x()
        vy = self.tello.get_velocity_y()
        vz = self.tello.get_velocity_z()
        print(f"Velocity X: {vx} cm/s, Y: {vy} cm/s, Z: {vz} cm/s")
        
        # Convert cm/s to meters/frame
        speed_mps = np.sqrt(vx**2 + vy**2 + vz**2) / 100.0
        scale = speed_mps * dt
        return scale if scale > 0.005 else 0.0  # Threshold micro-noise when hovering
        
def run_pipeline(mode="KITTI", data_path=None):
    # main program code #
        if mode == "KITTI":
            print("Processing Kitti feed")
            source = KittiSource(data_path)
        elif mode == "TELLO":
            print("Processing Tello feed")
        elif mode == "Airsim":
            print("Processing Airsim feed")
            
if __name__ == "__main__":
    # --- For Testing KITTI ---
    # Downoad a sequence and change this path to point to your data folder
    kitti_sequence_directory = "C:/data/KITTI_sequence_1/"
    run_pipeline(mode="KITTI", data_path=kitti_sequence_directory)
    
    # --- For Live Flight with the Drone ---
    # 1. Turn on Tello and connect laptop Wi-Fi to it.
    # 2. Pip install djitelloapy
    # 3. Comment out the KITTI run above and uncomment the line below:
    # run_pipeline(mode="TELLO")