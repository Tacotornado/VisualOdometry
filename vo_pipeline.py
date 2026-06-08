import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import socket
import urllib.request
from djitellopy import Tello


# --- Data Source Management ---

class KittiSource:
    def __init__(self, sequence_path):
        self.img_dir = os.path.join(sequence_path, "images")
        self.images = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.idx = 0
        
        gt_path = os.path.join(sequence_path, "ground_truth.txt")
        self.gt_poses = np.loadtxt(gt_path, dtype=float) if os.path.exists(gt_path) else None
        
        calib_path = os.path.join(sequence_path, "calib.txt")
        self.K = self._load_intrinsics(calib_path)

    def _load_intrinsics(self, calib_path):
        try:
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    first_line = f.readline().strip().split()
                if len(first_line) >= 12:
                    data = [float(x) for x in first_line[1:]] if ':' in first_line[0] else [float(x) for x in first_line]
                    P = np.array(data[:12]).reshape(3, 4)
                    return P[:, :3]
        except Exception as e:
            print(f"Warning: Could not parse calib.txt ({e}). Using default KITTI matrix.")
            
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
        if self.gt_poses is None or self.idx < 2:
            return 1.0
        p1 = self.gt_poses[self.idx - 2].reshape(3, 4)[:, 3]
        p2 = self.gt_poses[self.idx - 1].reshape(3, 4)[:, 3]
        return float(np.linalg.norm(p2 - p1))
    

class TelloSource:
    def __init__(self):
        print("Attempting to connect to Tello drone ...")
        self.tello = Tello()
        self.tello.connect()
        
        print("success: connected to drone")
        print(f"Battery percentage at: {self.tello.get_battery()}%")
        
        print("Initializing camera video stream ...")
        self.tello.streamon()
        time.sleep(5.0)
        
        print("Hao de!, stream is on!")
        self.frame_reader = self.tello.get_frame_read()
        
        self.K = np.array([[365.9667,   0.0,    213.3087],
                           [  0.0,    496.2820, 225.1782],
                           [  0.0,      0.0,      1.0]])
        self.last_time = time.time()
        
    def get_frame(self):
        return self.frame_reader.frame
    
    def get_scale(self):
        dt = time.time() - self.last_time
        self.last_time = time.time()

        try:
            state = self.tello.get_current_state()
            vx = int(state.get('vgx', 0))
            vz = int(state.get('vgz', 0))
            
            alt = self.tello.get_height() / 100.0
            if not hasattr(self, 'last_alt'):
                self.last_alt = alt
            
            alt_delta = alt - self.last_alt
            self.last_alt = alt
        except Exception:
            vx, vz, alt_delta = 0, 0, 0.0
            print("Exception: state not found, using fallback")

        h_speed_mps = np.sqrt(vx**2 + vz**2) / 100.0
        horizontal_scale = h_speed_mps * dt

        self._last_alt_delta = alt_delta
        return horizontal_scale if horizontal_scale > 0.005 else 0.0
    

class Autohawk2ASource:
    def __init__(self, drone_ip="192.168.4.1", video_port=80, cmd_port=8081):
        self.drone_ip = drone_ip
        self.video_port = video_port
        self.cmd_port = cmd_port
        print(f"Attempting to connect to Autohawk2A at {self.drone_ip}...")
        
        self.stream_url = f"http://{self.drone_ip}:{self.video_port}/stream"
        try:
            self.stream = urllib.request.urlopen(self.stream_url, timeout=5)
            print("Video stream connected successfully")
        except Exception as e:
            print(f"Error connecting to video stream: {e}")
            self.stream = None
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        try:
            self.sock.bind(("", self.cmd_port))
        except Exception as e:
            print(f"Warning on UDP socket binding: {e}")
            
        self.K = np.array([[500.0,    0.0,  320.0],
                           [  0.0,  500.0,  240.0],
                           [  0.0,    0.0,    1.0]])
        self.bytes_buffer = bytes()
        self.last_time = time.time()
                    
    def get_frame(self):
        if self.stream is None:
            return None
        try:
            while True:
                self.bytes_buffer += self.stream.read(1024)
                a = self.bytes_buffer.find(b'\xff\xd8')
                b = self.bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg = self.bytes_buffer[a:b+2]
                    self.bytes_buffer = self.bytes_buffer[b+2:]
                    return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Failed to read camera byte frame: {e}")
            return None
            
    def get_scale(self):
        dt = time.time() - self.last_time
        self.last_time = time.time()
        vx, vy, vz = 0.0, 0.0, 0.0
        try:
            while True:
                data, addr = self.sock.recvfrom(1024)
                packet_str = data.decode('utf-8').strip()
                parts = packet_str.split(',')
                if len(parts) >= 3:
                    vx, vy, vz = float(parts[0]), float(parts[1]), float(parts[2])
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Telemetry decoding failed: {e}")
            
        speed_mps = np.sqrt(vx**2 + vy**2 + vz**2)
        scale = speed_mps * dt
        return scale if scale > 0.001 else 0.0


# --- Main Odometry Pipeline ---

def run_pipeline(mode="KITTI", data_path=None, shared_queue=None):
    if mode == "KITTI":
        print("Processing Kitti feed")
        source = KittiSource(data_path)
    elif mode == "TELLO":
        print("Processing Tello feed")
        source = TelloSource()
    elif mode == "AUTOHAWK2A":
        print("Processing Autohawk2A feed")
        source = Autohawk2ASource(drone_ip="192.168.4.1", video_port=80, cmd_port=8081)
    else:
        print("unknown feed")
        return
    
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    traj_x, traj_z = [0], [0]
    
    frame_counter = 0
    estimated_total_frames = 300
    
    if mode == "KITTI" and data_path and os.path.exists(data_path):
        estimated_total_frames = len(os.listdir(data_path))
    
    # Hide desktop windows completely if running through a multiprocessing queue
    show_local_plots = (shared_queue is None)
    
    if show_local_plots:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'ro-', label="Tracked Path")
        ax.legend()
        ax.grid(True)
    
    # --- FIX: Cycle frames until the drone camera sensor returns an actual valid picture ---
    frame_prev = None
    print("Waiting for stable camera feed initialization...")
    for _ in range(30):
        frame_prev = source.get_frame()
        if frame_prev is not None and np.sum(frame_prev) > 0:
            break
        time.sleep(0.1)
        
    if frame_prev is None or np.sum(frame_prev) == 0: 
        print("Error: could not read valid initial frame from source.")
        return
        
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    
    while True:
        frame_curr = source.get_frame()
        if frame_curr is None:
            break
        
        frame_counter += 1
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        scale = source.get_scale()
        
        # --- FIX: Guard block against empty tracking vectors before launching optical flow ---
        if pts_prev is None or len(pts_prev) == 0:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            gray_prev = gray_curr
            continue
        
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_prev, None)
        
        if pts_curr is None or status is None or len(pts_curr) == 0:
            pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            gray_prev = gray_curr
            pts_prev = pts_curr
            continue
        
        good_prev = pts_prev[status.ravel() == 1]
        good_curr = pts_curr[status.ravel() == 1]
        
        # --- FIX: Only calculate essential matrix if enough keypoints survived RANSAC filtering ---
        if len(good_curr) > 10:
            E, mask = cv2.findEssentialMat(good_curr, good_prev, source.K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                _, R, t, _ = cv2.recoverPose(E, good_curr, good_prev, source.K, mask=mask)

                if scale > 0.0:
                    delta_t = scale * cur_R.dot(t)
                    cur_t[0, 0] += delta_t[0, 0]
                    cur_t[2, 0] += delta_t[2, 0]
                    
                    if mode == "TELLO" and hasattr(source, '_last_alt_delta'):
                        cur_t[1, 0] += source._last_alt_delta
                    else:
                        cur_t[1, 0] += delta_t[1, 0]
                    
                    cur_R = R.dot(cur_R)
        
        traj_x.append(cur_t[0, 0])
        traj_z.append(cur_t[2, 0])
        
        # Superimpose feature tracking markers directly on frame
        frame_vis = frame_curr.copy()
        for pt in good_curr:
            x, y = pt.ravel()
            cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw local troubleshooting windows if running solo script
        if show_local_plots:
            line.set_data(traj_x, traj_z)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)
            cv2.imshow("VO Camera Feed", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Send telemetry and frame package down the pipeline queue
        if shared_queue is not None:
            real_x = float(cur_t[0, 0])
            real_y = -float(cur_t[1, 0])  # Normalizes climbing axis upward
            real_z = float(cur_t[2, 0])
            
            # Encode visual frame matrix to byte stream for robust multiprocessing transfer
            _, encoded_img = cv2.imencode('.jpg', frame_vis)
            jpeg_bytes = encoded_img.tobytes()
            
            shared_queue.put({
                "frame_idx": frame_counter,
                "total_frames": estimated_total_frames,
                "estimated": [real_x, real_y, real_z],
                "distance_from_start": float(np.linalg.norm(cur_t)),
                "video_frame": jpeg_bytes
            })

        if len(good_curr) < 200:
            pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            
        gray_prev = gray_curr
        pts_prev = pts_curr

    if show_local_plots:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

            
if __name__ == "__main__":
    kitti_sequence_directory = "./data/Kitti/flight_path_00"
    run_pipeline(mode="KITTI", data_path=kitti_sequence_directory)