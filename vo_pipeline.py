import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from djitellopy import Tello


# Data source management #

class KittiSource:
    def __init__(self, sequence_path):
            # 1. Update image path to match your "images" folder
            self.img_dir = os.path.join(sequence_path, "images")
            self.images = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.idx = 0
            
            # 2. Update to match your "ground_truth.txt" file name
            gt_path = os.path.join(sequence_path, "ground_truth.txt")
            self.gt_poses = np.loadtxt(gt_path, dtype=float) if os.path.exists(gt_path) else None
            
            # 3. Parse your specific "calib.txt" dynamically if it's there, 
            # otherwise fallback to default KITTI intrinsics
            calib_path = os.path.join(sequence_path, "calib.txt")
            self.K = self._load_intrinsics(calib_path)

    def _load_intrinsics(self, calib_path):
        """Attempts to parse calib.txt. If formatting differs, falls back to baseline."""
        try:
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    first_line = f.readline().strip().split()
                # Check if it's a standard KITTI 12-element projection row
                if len(first_line) >= 12:
                    # If it starts with an identifier like 'P0:', drop it
                    data = [float(x) for x in first_line[1:]] if ':' in first_line[0] else [float(x) for x in first_line]
                    P = np.array(data[:12]).reshape(3, 4)
                    return P[:, :3] # Return the 3x3 Intrinsic matrix K
        except Exception as e:
            print(f"Warning: Could not parse calib.txt ({e}). Using default KITTI matrix.")
            
        # Default fallback KITTI Intrinsics matrix
        return np.array([[718.856,   0.0,   607.1928],
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
            source = TelloSource()
        elif mode == "Airsim":
            print("Processing Airsim feed")
            return
        else:
            print("unknown feed")
            return
        
        # Trajectory State matrix setups
        cur_R = np.eye(3)
        cur_t = np.zeros((3, 1))
        
        # Storage of traversed positions for plotting shit live
        traj_x, traj_z = [0], [0]
        
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'ro-', label="Tracked Path")
        ax.legend()
        ax.grid(True)
        
        # Processing frame 0
        frame_prev = source.get_frame()
        if frame_prev is None: 
            print("Error: could not read first frame.")
            return
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        
        # initial feature detection
        pts_prev = cv2.goodFeaturesToTrack(gray_prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        
        while True:
            frame_curr = source.get_frame()
            if frame_curr is None:
                break
            
            gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
            scale = source.get_scale()
            
            # feature tracking part with optical flow
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_prev, None)
            
            # redetect features if lost optical flow
            if pts_curr is None or status is None:
                pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
                gray_prev = gray_curr
                pts_prev = pts_curr
                continue
            
            # good_prev =  pts_prev[status[:, 0] == 1]
            # good_curr =  pts_curr[status[:, 0] == 1]
            
            good_prev = pts_prev[status.ravel() == 1]
            good_curr = pts_curr[status.ravel() == 1]
            
            if len(good_curr) > 10:
                # Estimate Essential Matrix and Essential Motion
                E, mask = cv2.findEssentialMat(good_curr, good_prev, source.K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
                _, R, t, _ = cv2.recoverPose(E, good_curr, good_prev, source.K, mask=mask)

                # Accumulate pose if the drone actually moved
                if scale > 0.0:
                    cur_t = cur_t + scale * cur_R.dot(t)
                    cur_R = R.dot(cur_R)

                # Update live plotting lists (mapping 3D coordinates to a 2D floor plan)
                traj_x.append(cur_t[0, 0])
                traj_z.append(cur_t[2, 0])
                
                # Dynamic map scaling
                line.set_data(traj_x, traj_z)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

            # Visual Diagnostics Window
            frame_vis = frame_curr.copy()
            for pt in good_curr:
                x, y = pt.ravel()
                cv2.circle(frame_curr, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imshow("VO Camera Feed", frame_curr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Refresh tracking points if they drop too low
            if len(good_curr) < 200:
                pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
                
            gray_prev = gray_curr
            pts_prev = pts_curr

        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
            
            
if __name__ == "__main__":
    # --- For Testing KITTI ---
    # Download a sequence and change this path to point to your data folder
    kitti_sequence_directory = "./data/Kitti/flight_path_00"
    run_pipeline(mode="KITTI", data_path=kitti_sequence_directory)
    # run_pipeline(mode="TELLO")
    
    # --- For Live Flight with the Drone ---
    # 1. Turn on Tello and connect laptop Wi-Fi to it.
    # 2. Pip install djitelloapy
    # 3. Comment out the KITTI run above and uncomment the line below:
    # run_pipeline(mode="TELLO")