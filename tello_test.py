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

class LiveTelloVIO:
    def __init__(self):
        self.K = np.array([[921.0,   0.0, 480.0],
                           [  0.0, 921.0, 360.0],
                           [  0.0,   0.0,   1.0]], dtype=np.float64)
        self.prev_frame = None
        self.cur_pose = np.eye(4, dtype=np.float64)
        
    def process_frame(self, frame, scale_factor, imu_R):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return self.cur_pose, frame

        p1 = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=150, qualityLevel=0.01, minDistance=10)
        if p1 is None or len(p1) < 10:
            self.prev_frame = gray
            return self.cur_pose, frame
            
        p2, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, p1, None)
        good_p1 = p1[st == 1]
        good_p2 = p2[st == 1]

        if len(good_p2) < 10:
            self.prev_frame = gray
            return self.cur_pose, frame

        E, mask = cv2.findEssentialMat(good_p1, good_p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self.prev_frame = gray
            return self.cur_pose, frame

        _, _, cam_t, _ = cv2.recoverPose(E, good_p1, good_p2, self.K, mask=mask)
        
        # VIO Fusion Update
        R = imu_R 
        t = cam_t * scale_factor
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        self.cur_pose = self.cur_pose @ np.linalg.inv(T)
        
        display_frame = frame.copy()
        for i, (new, old) in enumerate(zip(good_p2, good_p1)):
            a, b = new.ravel()
            display_frame = cv2.circle(display_frame, (int(a), int(b)), 3, (0, 255, 0), -1)
            
        self.prev_frame = gray
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
    
    vio = LiveTelloVIO()
    last_time = time.time()
    
    # --- SETUP HEADLESS LIVE DASHBOARD CANVASES ---
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
            
            # Read IMU + Telemetry states
            pitch, roll, yaw = drone.get_pitch(), drone.get_roll(), drone.get_yaw()
            r_obj = SciPyRot.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            imu_R = r_obj.as_matrix()
            
            vx, vy, vz = drone.get_speed_x(), drone.get_speed_y(), drone.get_speed_z()
            speed_magnitude = np.sqrt(vx**2 + vy**2 + vz**2) / 100.0
            
            dt = time.time() - last_time
            last_time = time.time()
            
            scale = speed_magnitude * dt
            if scale <= 0: 
                scale = 0.05 * dt 
            
            current_pose, video_img = vio.process_frame(frame, scale_factor=scale, imu_R=imu_R)
            
            x, y, z = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
            path_history.append([x, y, z])
            frame_count += 1
            
            print(f"VIO Target -> X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m", end="\r")
            
            # --- RENDER MAP TO MEMORY BUFFER ---
            if frame_count % 5 == 0:
                ax.clear()
                path_np = np.array(path_history)
                
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2)
                ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', s=60)
                ax.scatter(x, y, z, color='red', marker='X', s=80)
                
                ax.set_title('Live 3D Position Map')
                pad = 0.4
                ax.set_xlim(np.min(path_np[:, 0]) - pad, np.max(path_np[:, 0]) + pad)
                ax.set_ylim(np.min(path_np[:, 1]) - pad, np.max(path_np[:, 1]) + pad)
                ax.set_zlim(np.min(path_np[:, 2]) - pad, np.max(path_np[:, 2]) + pad)
                
                fig.canvas.draw()
                rgba_buffer = fig.canvas.buffer_rgba()
                plot_img = np.asarray(rgba_buffer)[:, :, :3]
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
                plot_img = cv2.resize(plot_img, (500, 500))
            
            # --- COMBINE CAMERA + DASHBOARD PANEL ---
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
            
            # Explicitly drop the headless memory figure to clean the workspace
            plt.close(fig)
            
            # Safely switch to our desktop display engine
            import importlib
            matplotlib.use('TkAgg') 
            importlib.reload(plt) # Re-imports plt cleanly to lock in the windowed backend
            
            final_fig = plt.figure(figsize=(10, 8))
            final_ax = final_fig.add_subplot(111, projection='3d')
            
            path_np = np.array(path_history)
            
            final_ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2.5, label='Traversed Path')
            final_ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', marker='o', s=120, label='Takeoff (Origin)')
            final_ax.scatter(path_np[-1, 0], path_np[-1, 1], path_np[-1, 2], color='red', marker='X', s=150, label='Landing Location')
            
            final_ax.set_title('Tello Trajectory - Visual-Inertial Odometry Final Path', fontsize=14, pad=20)
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