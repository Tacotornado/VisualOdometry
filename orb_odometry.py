import cv2
import numpy as np
import time
import threading
import queue
from djitellopy import Tello
from djitellopy.tello import TelloException

import matplotlib
# Initialize using 'Agg' so live flight rendering happens purely in memory
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as SciPyRot
from scipy.optimize import least_squares
from ahrs.filters import Madgwick

class LiveTelloVIOWithBA:
    def __init__(self, window_size=5):
        self.K = np.array([[921.0,   0.0, 480.0],
                           [  0.0, 921.0, 360.0],
                           [  0.0,   0.0,   1.0]], dtype=np.float64)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.cur_pose = np.eye(4, dtype=np.float64)
        
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # --- BUNDLE ADJUSTMENT WINDOW STORAGE ---
        self.window_size = window_size
        self.pose_history = []  # Stores past 4x4 relative camera matrices
        self.points_2d_history = []  # Stores matched pixel coordinates for optimization
        
    def reprojection_error_vector(self, params, points_3d, points_2d):
        """
        The mathematical cost function minimized by Bundle Adjustment.
        Adjusts the 3D translation vector to minimize pixel reprojection drift.
        """
        # Extract the optimized translation components from the flat parameter array
        t_opt = params[:3].reshape(3, 1)
        R = params[3:].reshape(3, 3)
        
        # Project 3D space points back into 2D camera coordinates using the tested pose
        projected_points = []
        for pt_3d in points_3d:
            # Transform point to camera frame: P_cam = R * P_world + t
            pt_cam = R @ pt_3d.reshape(3, 1) + t_opt
            if pt_cam[2, 0] == 0:
                pt_cam[2, 0] = 0.01  # Safeguard against division by zero depth
            
            # Project onto image sensor plane using Intrinsic Matrix K
            x_pixel = (self.K[0, 0] * pt_cam[0, 0] / pt_cam[2, 0]) + self.K[0, 2]
            y_pixel = (self.K[1, 1] * pt_cam[1, 0] / pt_cam[2, 0]) + self.K[1, 2]
            projected_points.append([x_pixel, y_pixel])
            
        projected_points = np.array(projected_points)
        # Return the flat residual vector (the distances between measured and projected pixels)
        return (projected_points - points_2d).flatten()

    def process_frame(self, frame, scale_factor, imu_R):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None or des is None or self.prev_des is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.cur_pose, frame

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 20:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.cur_pose, frame

        good_p1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        good_p2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(good_p1, good_p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.cur_pose, frame

        _, _, cam_t, _ = cv2.recoverPose(E, good_p1, good_p2, self.K, mask=mask)
        
        # VIO raw fusion step
        R = imu_R 
        t = cam_t * scale_factor
        
        # --- BUNDLE ADJUSTMENT STEP (SLIDING WINDOW OPTIMIZATION) ---
        # 1. Triangulate matched 2D features into temporary local 3D points using current R & initial t
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t.reshape(3, 1)))
        
        pts1_hom = good_p1.squeeze()
        pts2_hom = good_p2.squeeze()
        
        # Standard linear triangulation to get approximate 3D world landmarks
        pts_4d = cv2.triangulatePoints(P1, P2, pts1_hom.T, pts2_hom.T)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T  # Convert from homogeneous to standard Cartesian 3D coordinates
        
        # 2. Add current window components to history queues
        self.points_2d_history.append(pts2_hom)
        self.pose_history.append((R, t))
        
        # Keep window bound constrained to size parameter to maintain high optimization loop execution rates
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)
            self.points_2d_history.pop(0)
            
        # 3. Execute Levenberg-Marquardt optimizer across the compiled window history data
        if len(self.pose_history) >= 3:
            # Flatten the target variables we want to optimize (the translation components)
            init_params = np.hstack((t.flatten(), R.flatten()))
            
            # Run local bundle adjustment optimization via SciPy
            res = least_squares(
                self.reprojection_error_vector, 
                init_params, 
                args=(pts_3d, pts2_hom), 
                method='lm',
                max_nfev=15 # Hardcapped iterations to prevent streaming lag
            )
            
            # Extract optimized translation components out of the resolved parameter vector
            t_optimized = res.x[:3].reshape(3, 1)
            t = t_optimized
            
        # Accumulate the stabilized, optimized transformation parameters into global space
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        self.cur_pose = self.cur_pose @ np.linalg.inv(T)
        
        # Render feature landmarks
        display_frame = frame.copy()
        mask_ravel = mask.ravel()
        for i, m in enumerate(matches[:50]):
            if i < len(mask_ravel) and mask_ravel[i]:
                pt2 = tuple(map(int, kp[m.trainIdx].pt))
                display_frame = cv2.circle(display_frame, pt2, 4, (255, 0, 255), -1) # Magenta for BA-tracked nodes
            
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        return self.cur_pose, display_frame


def video_worker(drone, frame_queue, stop_event):
    drone.streamon()
    frame_read = drone.get_frame_read()
    while not stop_event.is_set():
        frame = frame_read.frame
        if frame is None:
            continue
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)
        time.sleep(0.01)


def main():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    vid_thread = threading.Thread(target=video_worker, args=(drone, frame_queue, stop_event), daemon=True)
    vid_thread.start()
    
    # Initialize the upgraded class with a sliding window of 5 keyframes
    vio = LiveTelloVIOWithBA(window_size=5)
    last_time = time.time()
    
    madgwick_filter = Madgwick(frequency=30.0, gain=0.1)
    current_q = np.array([1.0, 0.0, 0.0, 0.0])
    prev_angles = np.array([drone.get_roll(), drone.get_pitch(), drone.get_yaw()])
    
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    path_history = []
    frame_count = 0  
    
    plot_img = np.zeros((500, 500, 3), dtype=np.uint8)

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            dt = time.time() - last_time
            last_time = time.time()
            if dt <= 0:
                dt = 0.033
            
            # IMU Madgwick processing steps
            roll, pitch, yaw = drone.get_roll(), drone.get_pitch(), drone.get_yaw()
            current_angles = np.array([roll, pitch, yaw])
            
            angle_diff = np.radians(current_angles - prev_angles)
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            gyro_data = angle_diff / dt
            prev_angles = current_angles.copy()
            
            rot_current = SciPyRot.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            gravity_world = np.array([0.0, 0.0, 9.81])
            accel_data = rot_current.inv().apply(gravity_world)
            
            current_q = madgwick_filter.updateIMU(q=current_q, gyr=gyro_data, acc=accel_data)
            fused_rot_obj = SciPyRot.from_quat([current_q[1], current_q[2], current_q[3], current_q[0]]) 
            imu_R = fused_rot_obj.as_matrix()
            
            # Hardware speed metrics scale extraction
            vx, vy, vz = drone.get_speed_x(), drone.get_speed_y(), drone.get_speed_z()
            speed_magnitude = np.sqrt(vx**2 + vy**2 + vz**2) / 100.0
            
            scale = speed_magnitude * dt
            if scale <= 0: 
                scale = 0.05 * dt 
            
            # Compute position via local BA VIO pipeline
            current_pose, video_img = vio.process_frame(frame, scale_factor=scale, imu_R=imu_R)
            
            x, y, z = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
            path_history.append([x, y, z])
            frame_count += 1
            
            print(f"BA-VIO Optimized -> X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m", end="\r")
            
            # Render live plot updates to visual memory buffer matrix
            if frame_count % 5 == 0:
                ax.clear()
                path_np = np.array(path_history)
                
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2)
                ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', s=60)
                ax.scatter(x, y, z, color='red', marker='X', s=80)
                
                ax.set_title('Live BA Optimized 3D Map')
                pad = 0.4
                ax.set_xlim(np.min(path_np[:, 0]) - pad, np.max(path_np[:, 0]) + pad)
                ax.set_ylim(np.min(path_np[:, 1]) - pad, np.max(path_np[:, 1]) + pad)
                ax.set_zlim(np.min(path_np[:, 2]) - pad, np.max(path_np[:, 2]) + pad)
                
                fig.canvas.draw()
                rgba_buffer = fig.canvas.buffer_rgba()
                plot_img = np.asarray(rgba_buffer)[:, :, :3]
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
                plot_img = cv2.resize(plot_img, (500, 500))
            
            video_resized = cv2.resize(video_img, (666, 500)) 
            dashboard_window = np.hstack((video_resized, plot_img))
            
            cv2.imshow("Tello Integrated VIO Station", dashboard_window)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\n[INFO] Ending live flight loop. Shutting down hardware streams...")
        stop_event.set()
        vid_thread.join()
        cv2.destroyAllWindows()
        
        try: 
            drone.streamoff()
        except TelloException: 
            pass
        
        # --- POST-FLIGHT: GENERATE INTERACTIVE PERSISTENT PLOT ---
        if len(path_history) > 5:
            print("\n[SUCCESS] Generating permanent post-flight analysis graph...")
            
            plt.close(fig)
            
            import importlib
            matplotlib.use('TkAgg') 
            importlib.reload(plt) 
            
            final_fig = plt.figure(figsize=(10, 8))
            final_ax = final_fig.add_subplot(111, projection='3d')
            
            path_np = np.array(path_history)
            
            final_ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2.5, label='Traversed Path')
            final_ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', marker='o', s=120, label='Takeoff (Origin)')
            final_ax.scatter(path_np[-1, 0], path_np[-1, 1], path_np[-1, 2], color='red', marker='X', s=150, label='Landing Location')
            
            final_ax.set_title('Tello Trajectory - Bundle Adjusted VIO Final Path', fontsize=14, pad=20)
            final_ax.set_xlabel('X Position (meters)', fontsize=10)
            final_ax.set_ylabel('Y Position (meters)', fontsize=10)
            final_ax.set_zlabel('Z Position (meters)', fontsize=10)
            final_ax.legend(loc='upper right')
            final_ax.grid(True)
            
            print("Flight Complete. Close the standalone 3D plot window to completely exit the program.")
            plt.show() 
        else:
            print("[WARNING] Not enough position logs captured to generate an accurate path plot.")

if __name__ == "__main__":
    main()