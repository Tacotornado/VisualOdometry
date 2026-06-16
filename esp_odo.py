import cv2
import numpy as np
import time
import threading
import queue
import socket
import struct
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

        # --- STATIC DRIFT ELIMINATION GATING ---
        pixel_displacements = np.linalg.norm(good_p1 - good_p2, axis=1)
        avg_pixel_move = np.mean(pixel_displacements)
        
        if avg_pixel_move < 0.6:
            scale_factor = 0.0
        # --------------------------------------------

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


# ==============================================================================
# MODULAR VIDEO INPUT CLASSES
# ==============================================================================

class BaseVideoSource(threading.Thread):
    """Abstract baseline class establishing queue management properties."""
    def __init__(self, frame_queue, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.daemon = True

    def _push_to_queue(self, frame):
        """Safely queues frames while discarding dropped latency backlog frames."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)


class TelloVideoSource(BaseVideoSource):
    """Handles native DJI Tello SDK camera capture commands."""
    def __init__(self, drone_instance, frame_queue, stop_event):
        super().__init__(frame_queue, stop_event)
        self.drone = drone_instance

    def run(self):
        print("[Video Pipeline] Activating Tello streaming hardware...")
        self.drone.streamon()
        frame_read = self.drone.get_frame_read()
        
        while not self.stop_event.is_set():
            frame = frame_read.frame
            if frame is None:
                continue
            self._push_to_queue(frame)
            time.sleep(0.01)
            
        try:
            self.drone.streamoff()
        except TelloException:
            pass
        print("[Video Pipeline] Tello streaming stream closed cleanly.")


class Esp32VideoSource(BaseVideoSource):
    """Parses incoming HTTP network stream variables from Seeed Studio XIAO ESP32-S3."""
    def __init__(self, stream_url, frame_queue, stop_event):
        super().__init__(frame_queue, stop_event)
        self.stream_url = stream_url

    def run(self):
        print(f"[Video Pipeline] Opening network camera feed at: {self.stream_url}")
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stop_event.is_set():
            if not cap.isOpened():
                print("[Video Pipeline] ESP32 stream disconnected. Attempting reconnection...")
                time.sleep(2.0)
                cap = cv2.VideoCapture(self.stream_url)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            self._push_to_queue(frame)
            
        cap.release()
        print("[Video Pipeline] ESP32 stream interface dropped cleanly.")


class Esp32UdpVideoSource(BaseVideoSource):
    """Parses fragmented binary UDP network stream from ESP32 camera hardware."""
    def __init__(self, local_ip, port, esp_ip, frame_queue, stop_event):
        super().__init__(frame_queue, stop_event)
        self.local_ip = local_ip
        self.port = port
        self.esp_ip = esp_ip
        self.chunk_hdr = 6

    def run(self):
        print(f"[Video Pipeline] Binding UDP Listener on {self.local_ip}:{self.port}, targeting ESP at {self.esp_ip}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.local_ip, self.port))
        sock.settimeout(1.0) # Lowered timeout slightly for responsive thread shutdowns

        # Signal ESP to begin transmitting data stream
        try:
            sock.sendto(b"OPEN", (self.esp_ip, self.port))
        except Exception as e:
            print(f"[Video Pipeline] Warning: Failed to send OPEN command: {e}")

        frames = {}

        while not self.stop_event.is_set():
            try:
                packet, _ = sock.recvfrom(1500)
            except socket.timeout:
                continue # Allows checking the stop_event periodic trigger
            except Exception:
                break

            if len(packet) < self.chunk_hdr:
                continue

            frame_id, chunk_id, total_chunks = struct.unpack("<HHH", packet[:self.chunk_hdr])
            data = packet[self.chunk_hdr:]

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks
                }
                
            frames[frame_id]["chunks"][chunk_id] = data

            if len(frames[frame_id]["chunks"]) == total_chunks:
                full = b"".join(frames[frame_id]["chunks"][i] for i in range(total_chunks))
                img = cv2.imdecode(np.frombuffer(full, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.flip(img, 0)

                if img is not None:
                    self._push_to_queue(img)
                
                del frames[frame_id]

        # Cleanup process sequence
        print("[Video Pipeline] Disconnecting UDP stream interface...")
        try:
            sock.sendto(b"CLOSE", (self.esp_ip, self.port))
        except Exception:
            pass
        sock.close()
        print("[Video Pipeline] UDP stream interface closed cleanly.")


# ==============================================================================
# MAIN SELECTION & PROCESSING ENVIRONMENT
# ==============================================================================

def main():
    # -------------------------------------------------------------------------
    # SOURCE SELECTOR SWITCH
    # Choose your active hardware mode here ("TELLO", "ESP32", or "ESP32_UDP")
    # -------------------------------------------------------------------------
    ACTIVE_MODE = "ESP32_UDP"  
    
    # Configuration inputs for each drone source
    ESP32_URL = "http://192.168.1.100:81/stream"  # Replace with actual ESP32 web link
    
    # UDP configuration elements
    ESP_IP = "192.168.43.42"
    LOCAL_IP = "0.0.0.0"
    UDP_PORT = 5000
    
    # Internal management pointers
    drone = None
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    # Spin up the selected class instance
    if ACTIVE_MODE == "TELLO":
        drone = Tello()
        drone.connect()
        print(f"Connected to Tello. Battery level: {drone.get_battery()}%")
        video_worker = TelloVideoSource(drone, frame_queue, stop_event)
    elif ACTIVE_MODE == "ESP32":
        print("Configuring operational script environment for self-made custom HTTP ESP32 drone platform...")
        video_worker = Esp32VideoSource(ESP32_URL, frame_queue, stop_event)
    elif ACTIVE_MODE == "ESP32_UDP":
        print("Configuring operational script environment for binary UDP custom ESP32 drone platform...")
        video_worker = Esp32UdpVideoSource(LOCAL_IP, UDP_PORT, ESP_IP, frame_queue, stop_event)
    else:
        raise ValueError("Invalid target source specified. Assign 'TELLO', 'ESP32', or 'ESP32_UDP'.")

    # Start the standalone video stream extraction thread safely
    video_worker.start()
    
    vio = LiveTelloVIO()
    last_time = time.time()
    
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
            if dt <= 0: dt = 0.033
            
            # --- HANDLE COMPANION TELEMETRY PER ACTIVE SOURCE TYPE ---
            if ACTIVE_MODE == "TELLO" and drone:
                pitch, roll, yaw = drone.get_pitch(), drone.get_roll(), drone.get_yaw()
                vx, vy, vz = drone.get_speed_x(), drone.get_speed_y(), drone.get_speed_z()
                speed_magnitude = np.sqrt(vx**2 + vy**2 + vz**2) / 100.0
            else:
                # Fallback metrics when running custom build variants without telemetry downlinks
                pitch, roll, yaw = 0.0, 0.0, 0.0
                speed_magnitude = 0.3  # Estimated flight velocity benchmark (m/s)

            r_obj = SciPyRot.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            imu_R = r_obj.as_matrix()
            
            scale = speed_magnitude * dt
            if scale <= 0: scale = 0.05 * dt 
            
            current_pose, video_img = vio.process_frame(frame, scale_factor=scale, imu_R=imu_R)
            
            x, y, z = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
            path_history.append([x, y, z])
            frame_count += 1
            
            print(f"[{ACTIVE_MODE} VIO Mode] -> X: {x:.2f}m, Y: {y:.2f}m, Z: {z:.2f}m", end="\r")
            
            if frame_count % 5 == 0:
                ax.clear()
                path_np = np.array(path_history)
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2)
                ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', s=60)
                ax.scatter(x, y, z, color='red', marker='X', s=80)
                
                ax.set_title(f'Live {ACTIVE_MODE} 3D Odometry Map')
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
            cv2.imshow("Drone Integrated VIO Station", dashboard_window)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\n[INFO] Terminating navigation loops. Closing camera threads...")
        stop_event.set()
        video_worker.join()
        cv2.destroyAllWindows()
        
        if len(path_history) > 5:
            print("\n[SUCCESS] Generating permanent post-flight visualization window...")
            plt.close(fig)
            
            import importlib
            matplotlib.use('TkAgg') 
            importlib.reload(plt) 
            
            final_fig = plt.figure(figsize=(10, 8))
            final_ax = final_fig.add_subplot(111, projection='3d')
            path_np = np.array(path_history)
            
            final_ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], color='blue', linewidth=2.5, label='Traversed Flight Path')
            final_ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2], color='green', marker='o', s=120, label='Takeoff (Origin)')
            final_ax.scatter(path_np[-1, 0], path_np[-1, 1], path_np[-1, 2], color='red', marker='X', s=150, label='Landing Node')
            
            final_ax.set_title(f'Final Spatial Mapping - {ACTIVE_MODE} Hardware Run', fontsize=14, pad=20)
            final_ax.set_xlabel('X Axis (Meters)', fontsize=10)
            final_ax.set_ylabel('Y Axis (Meters)', fontsize=10)
            final_ax.set_zlabel('Z Axis (Meters)', fontsize=10)
            final_ax.legend(loc='upper right')
            final_ax.grid(True)
            
            print("Session complete. Close the interactive chart canvas to terminate script process.")
            plt.show()

if __name__ == "__main__":
    main()