import os
import sys
import time
import threading
import queue
import numpy as np
import cv2
from scipy.optimize import least_squares
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Explicit djitellopy bindings from your connection setup
from djitellopy import Tello
from djitellopy.tello import TelloException

# Global Constants
MAX_FRAME = 1000
MIN_NUM_FEAT = 2000


class VideoSource(ABC):
    """Abstract Base Class defining the interface for Visual Odometry Data Sources."""
    
    @abstractmethod
    def get_next_frame(self):
        """Returns the next BGR frame or None if finished."""
        pass

    @abstractmethod
    def get_scale(self):
        """Returns the absolute translation scale factor for the current step."""
        pass

    @abstractmethod
    def get_calibration_data(self):
        """Returns focal length and principal point: (focal, (cx, cy))."""
        pass

    @abstractmethod
    def get_frame_id(self):
        """Returns the current frame index tracker."""
        pass

    @abstractmethod
    def close(self):
        """Cleans up video/thread resources."""
        pass


class KittiDatasetSource(VideoSource):
    """Data source implementation for parsing the offline KITTI dataset sequence files."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.frame_id = 0
        
        self.img_dir = os.path.join(dataset_path, "images")
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Could not locate image directory at {self.img_dir}")
            
        self.focal, self.pp = self._parse_calibration()
        self.lines = self._parse_ground_truth()

    def _parse_calibration(self):
        calib_file = os.path.join(self.dataset_path, "calib.txt")
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    first_line = f.readline().strip().split()
                data = [float(x) for x in first_line[1:]] if ':' in first_line[0] else [float(x) for x in first_line]
                P = np.array(data[:12]).reshape(3, 4)
                return P[0, 0], (P[0, 2], P[1, 2])
            except Exception as e:
                print(f"Warning parsing calibration file: {e}. Using KITTI defaults.")
        return 718.856, (607.1928, 185.2157)

    def _parse_ground_truth(self):
        poses_file = os.path.join(self.dataset_path, "ground_truth.txt")
        if not os.path.exists(poses_file):
            poses_file = os.path.join(self.dataset_path, "00.txt")
        
        if os.path.exists(poses_file):
            try:
                with open(poses_file, 'r') as f:
                    return f.readlines()
            except Exception as e:
                print(f"Error reading ground truth file: {e}")
        return []

    def get_next_frame(self):
        filename = os.path.join(self.img_dir, f"{self.frame_id:06d}.png")
        img = cv2.imread(filename)
        if img is not None:
            self.frame_id += 1
        return img

    def get_scale(self):
        curr_idx = self.frame_id - 1
        prev_idx = curr_idx - 1

        if prev_idx < 0 or curr_idx >= len(self.lines):
            return 0.0

        try:
            line_curr = np.fromstring(self.lines[curr_idx], sep=' ')
            line_prev = np.fromstring(self.lines[prev_idx], sep=' ')
            
            x_curr, y_curr, z_curr = line_curr[3], line_curr[7], line_curr[11]
            x_prev, y_prev, z_prev = line_prev[3], line_prev[7], line_prev[11]

            return float(np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2 + (z_curr - z_prev)**2))
        except Exception as e:
            print(f"Error parsing scale data: {e}")
            return 0.0

    def get_calibration_data(self):
        return self.focal, self.pp

    def get_frame_id(self):
        return self.frame_id

    def close(self):
        pass


class TelloDroneSource(VideoSource):
    """
    Data source managing live drone hardware execution, matching your exact 
    queue-bound threading worker architecture with an added initialization buffer check.
    """
    
    def __init__(self, default_scale=1.0):
        self.frame_id = 0
        self.default_scale = default_scale
        
        # User-provided 3x3 Camera Intrinsic Matrix (K)
        self.K = np.array([[921.0,   0.0, 480.0],
                           [  0.0, 921.0, 360.0],
                           [  0.0,   0.0,   1.0]], dtype=np.float64)
        
        self.focal = self.K[0, 0]              # fx
        self.pp = (self.K[0, 2], self.K[1, 2])  # (cx, cy)

        # Connect to Hardware using your exact layout
        self.drone = Tello()
        self.drone.connect()
        print(f"Connected! Battery Life: {self.drone.get_battery()}%")
        
        # Setup cross-thread shared queue structures
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        # Initialize background video daemon task
        self.vid_thread = threading.Thread(
            target=self._video_worker, 
            args=(self.drone, self.frame_queue, self.stop_event), 
            daemon=True
        )
        self.vid_thread.start()
        
        # --- FIXED: Wait for the video stream decode pipeline to actually populate the queue ---
        print("Waiting for Tello video hardware stream to warm up...")
        retries = 20
        while self.frame_queue.empty() and retries > 0:
            time.sleep(0.2)
            retries -= 1
            
        if self.frame_queue.empty():
            print("[WARNING] Video stream connection timed out. Forcing initialization proceed anyway...")
        else:
            print("Video pipeline active. Captured warm-up frames successfully.")

    @staticmethod
    def _video_worker(drone, frame_queue, stop_event):
        """ Your exact matching frame-dropping background consumer loop """
        drone.streamon()
        frame_read = drone.get_frame_read()
        while not stop_event.is_set():
            frame = frame_read.frame
            if frame is None:
                continue
            if frame_queue.full():
                try: 
                    frame_queue.get_nowait()
                except queue.Empty: 
                    pass
            frame_queue.put(frame)
            time.sleep(0.01)

    def get_next_frame(self):
        try:
            # Poll frame from the thread safe storage queue
            frame = self.frame_queue.get(timeout=1.0) # Bumped timeout slightly for network fluctuations
            self.frame_id += 1
            return frame.copy()
        except queue.Empty:
            return None

    def get_scale(self):
        return self.default_scale

    def get_calibration_data(self):
        return self.focal, self.pp

    def get_frame_id(self):
        return self.frame_id

    def close(self):
        print("\nEnding live flight loop. Shutting down hardware streams...")
        self.stop_event.set()
        if self.vid_thread.is_alive():
            self.vid_thread.join(timeout=1.0)
        try:
            self.drone.streamoff()
        except TelloException:
            pass


class VisualOdometryPipeline:
    """Core Visual Odometry tracking engine with Local Bundle Adjustment."""
    
    def __init__(self, source: VideoSource):
            self.source = source
            # Keeping this if you still want to see raw feature tracking
            self.traj = np.zeros((600, 600, 3), dtype=np.uint8) 
            
            f, pp = self.source.get_calibration_data()
            self.K = np.array([[f, 0, pp[0]],
                            [0, f, pp[1]],
                            [0, 0,    1]], dtype=np.float64)

            # Matplotlib Setup
            plt.ion() 
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_title("Drone Trajectory (Top-Down)")
            self.ax.set_xlabel("X (meters)")
            self.ax.set_ylabel("Z (meters)")
            self.line, = self.ax.plot([], [], 'm-o', markersize=2, linewidth=1)
            self.path_x, self.path_y = [], []

    @staticmethod
    def _feature_detection(img):
        pts = cv2.goodFeaturesToTrack(img, maxCorners=3000, qualityLevel=0.01, minDistance=10)
        if pts is not None:
            return pts.reshape(-1, 2)
        return np.array([], dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _feature_tracking(img_prev, img_curr, pts_prev):
        # Prevent crash if there are no points to track
        if pts_prev is None or len(pts_prev) == 0:
            return np.array([]), np.array([]), np.array([])
            
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, pts_prev, None, **lk_params)
        
        if status is None:
            return np.array([]), np.array([]), np.array([])
            
        valid = (status.ravel() == 1)
        return pts_prev[valid], pts_curr[valid], status

    @staticmethod
    def _reprojection_error(params, pts_3d, pts_2d, K):
        """
        The cost function for Bundle Adjustment.
        Takes 6 parameters (3 for rotation vector, 3 for translation),
        projects the 3D points back onto the 2D image, and calculates the drift error.
        """
        rvec = params[:3]
        tvec = params[3:6]
        
        # Project 3D points back to 2D camera plane
        proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
        proj_pts = proj_pts.reshape(-1, 2)
        
        # Calculate distance between predicted 2D points and actual measured 2D points
        return (proj_pts - pts_2d).ravel()

    def run(self):
        # Initial frame capture
        img_1_c = self.source.get_next_frame()
        img_2_c = self.source.get_next_frame()

        if img_1_c is None or img_2_c is None:
            print("Error: Empty video resources encountered on initialization.")
            return -1

        img_1 = cv2.cvtColor(img_1_c, cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2_c, cv2.COLOR_BGR2GRAY)

        points1 = self._feature_detection(img_1)
        
        if len(points1) == 0:
            print("[ERROR] Camera found 0 features. Check lighting or drone orientation.")
            self.source.close()
            return -1 
        
        # Initial tracking step
        points1, points2, status = self._feature_tracking(img_1, img_2, points1)
        focal, pp = self.source.get_calibration_data()

        # Pose recovery
        E, mask = cv2.findEssentialMat(points2, points1, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, points2, points1, focal=focal, pp=pp, mask=mask)

        R_f = R.copy()
        t_f = t.copy()

        prev_image = img_2.copy()
        prev_features = points2.copy()

        cv2.namedWindow("Camera View", cv2.WINDOW_AUTOSIZE)

        # Main Pipeline Loop
        while self.source.get_frame_id() < MAX_FRAME:
            curr_image_c = self.source.get_next_frame()
            if curr_image_c is None:
                print(f"Ending pipeline: Final frame reached.")
                break

            curr_image = cv2.cvtColor(curr_image_c, cv2.COLOR_BGR2GRAY)
            prev_features, curr_features, status = self._feature_tracking(prev_image, curr_image, prev_features)
            
            # Draw flow vectors on color frame
            if curr_features is not None and prev_features is not None:
                for i, (new, old) in enumerate(zip(curr_features, prev_features)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(curr_image_c, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

            E, mask = cv2.findEssentialMat(curr_features, prev_features, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is not None and E.shape == (3, 3):
                _, R, t, mask_pose = cv2.recoverPose(E, curr_features, prev_features, focal=focal, pp=pp, mask=mask)

                # Bundle Adjustment Optimization
                try:
                    rvec, _ = cv2.Rodrigues(R)
                    P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = self.K @ np.hstack((R, t))
                    
                    valid_pts1 = prev_features[mask_pose.ravel() > 0]
                    valid_pts2 = curr_features[mask_pose.ravel() > 0]
                    
                    if len(valid_pts1) >= 10:
                        pts_4d = cv2.triangulatePoints(P1, P2, valid_pts1.T, valid_pts2.T)
                        pts_3d = (pts_4d[:3] / pts_4d[3]).T
                        
                        initial_params = np.hstack((rvec.ravel(), t.ravel()))
                        res = least_squares(self._reprojection_error, initial_params, 
                                            args=(pts_3d, valid_pts2, self.K), 
                                            method='lm', max_nfev=15)
                        
                        R, _ = cv2.Rodrigues(res.x[:3])
                        t = res.x[3:6].reshape(3, 1)
                except Exception:
                    pass 

                scale = self.source.get_scale()
                if (scale > 0.1) or (isinstance(self.source, TelloDroneSource)):
                    t_f = t_f + scale * (R_f.dot(t))
                    R_f = R.dot(R_f)

            # Update Matplotlib Trajectory Plot
            self.path_x.append(t_f[0, 0])
            self.path_y.append(t_f[2, 0])
            self.line.set_xdata(self.path_x)
            self.line.set_ydata(self.path_y)
            
            # Dynamic Axis Scaling
            self.ax.set_xlim(min(self.path_x) - 10, max(self.path_x) + 10)
            self.ax.set_ylim(min(self.path_y) - 10, max(self.path_y) + 10)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Refresh Frames
            cv2.imshow("Camera View", curr_image_c)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_image = curr_image.copy()
            prev_features = curr_features.copy()

        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
        self.source.close()
        return 0


if __name__ == "__main__":
    # --- Option A: Run using offline KITTI dataset sequence folder ---
    kitti_path = r"D:\VisualOdometry\VisualOdometry\data\Kitti\flight_path_00"
    #source = KittiDatasetSource(kitti_path)
    
    # --- Option B: Run via your designated hardware pipeline queue layout ---
    source = TelloDroneSource(default_scale=5.0)
    
    pipeline = VisualOdometryPipeline(source)
    pipeline.run()