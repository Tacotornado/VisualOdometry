import os
import time
import threading
import queue
import cv2
import numpy as np
import matplotlib.pyplot as plt
import socket
import urllib.request
from djitellopy import Tello


# --- Threading frame buffer to decouple live camera feeds from VO math --- #

class ThreadFrameBuffer:
    def __init__(self, source, maxsize=32):
        self._source = source
        self._queue = queue.Queue(maxsize=maxsize)
        self._running = False
        self._thread = None
        
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        
    def _reader(self):
        while self._running:
            frame = self._source.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            
            # For live hardware, drop the oldest frame if full to stay real-time
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            
    def get_frame(self, timeout=0.5):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def stop(self):
        self._running = False
        

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
        
        try:
            self.tello.set_video_bitrate(Tello.VIDEO_BITRATE_1Mbps)
            self.tello.set_video_fps(Tello.VIDEO_FPS_15)
            print("Video stream optimized: 1Mbps, 15 FPS profile activated.")
        except Exception as e:
            print(f"Warning: Could not apply bitrate/FPS optimizations: {e}")
        
        print("Initializing camera video stream ...")
        self.tello.streamon()
        time.sleep(3.0)
        
        print("Hao de!, stream is on!")
        self.frame_reader = self.tello.get_frame_read()
        
        self.K = np.array([[929.79642875, 0.0,            467.97552327],
                           [0.0,          936.04669222,   362.39887663],
                           [0.0,          0.0,            1.0         ]])
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
    

# --- Unified Processing Engine Loop --- 

def _vo_worker(frame_buffer, source, mode, shared_queue, show_local_plots, estimated_total_frames):
    cur_R    = np.eye(3)
    cur_t    = np.zeros((3, 1))
    smooth_t = np.zeros((3, 1))
    alpha    = 0.25
 
    traj_x, traj_z = [0.0], [0.0]
    frame_counter  = 0
 
    if show_local_plots:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'ro-', label="Tracked Path")
        ax.legend(); ax.grid(True)
 
    # FIX: Fetch initial frame straight from file array when in KITTI mode
    # --- Warm-up: grab a valid initial frame based on data stream mode ---
    print("Initializing tracking context vectors...")
    frame_prev = None
    
    if mode == "KITTI":
        frame_prev = source.get_frame()
    else:
        # Live streams need a few moments to spin up and flush empty frames
        print("Waiting for live camera stream to stabilize...")
        for _ in range(60):  # Give it up to 6 seconds max
            frame_prev = frame_buffer.get_frame(timeout=0.2)
            if frame_prev is not None and np.sum(frame_prev) > 0:
                print("Stable live camera stream caught successfully!")
                break
            time.sleep(0.1)
 
    if frame_prev is None or np.sum(frame_prev) == 0:
        print("Error: Could not capture a valid initial image element.")
        return
 
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    pts_prev  = cv2.goodFeaturesToTrack(gray_prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
 
    while True:
        # FIX: Pull frames sequentially from directory list for offline verification loops
        if mode == "KITTI":
            frame_curr = source.get_frame()
        else:
            frame_curr = frame_buffer.get_frame(timeout=1.0)
            
        if frame_curr is None:
            break
 
        frame_counter += 1
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        scale     = source.get_scale()
 
        if pts_prev is None or len(pts_prev) == 0:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            gray_prev = gray_curr
            continue
 
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_prev, None)
 
        if pts_curr is None or status is None or len(pts_curr) == 0:
            pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            gray_prev = gray_curr
            pts_prev  = pts_curr
            continue
 
        good_prev = pts_prev[status.ravel() == 1]
        good_curr = pts_curr[status.ravel() == 1]
 
        if len(good_curr) > 10:
            E, mask = cv2.findEssentialMat(good_curr, good_prev, source.K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                _, R, t, _ = cv2.recoverPose(E, good_curr, good_prev, source.K, mask=mask)
                if scale > 0.0:
                    delta_t       = scale * cur_R.dot(t)
                    cur_t[0, 0]  += delta_t[0, 0]
                    cur_t[2, 0]  += delta_t[2, 0]
                    if mode == "TELLO" and hasattr(source, '_last_alt_delta'):
                        cur_t[1, 0] += source._last_alt_delta
                    else:
                        cur_t[1, 0] += delta_t[1, 0]
                    cur_R = R.dot(cur_R)
 
        smooth_t = alpha * cur_t + (1.0 - alpha) * smooth_t
        traj_x.append(smooth_t[0, 0])
        traj_z.append(smooth_t[2, 0])
 
        frame_vis = frame_curr.copy()
        for pt in good_curr:
            x, y = pt.ravel()
            cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
 
        if show_local_plots:
            line.set_data(traj_x, traj_z)
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)
            cv2.imshow("VO Camera Feed", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        if shared_queue is not None:
            real_x = float(smooth_t[0, 0])
            real_y = -float(smooth_t[1, 0])
            real_z = float(smooth_t[2, 0])
 
            # Never throttle frames when running local benchmark folders
            jpeg_bytes = None
            if mode == "KITTI" or shared_queue.qsize() < 3:
                _, enc = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = enc.tobytes()
 
            try:
                shared_queue.put({
                    "frame_idx":          frame_counter,
                    "total_frames":       estimated_total_frames,
                    "estimated":          [real_x, real_y, real_z],
                    "distance_from_start": float(np.linalg.norm(smooth_t)),
                    "video_frame":        jpeg_bytes,
                }, timeout=0.2)
            except queue.Full:
                pass
 
        if len(good_curr) < 200:
            pts_curr = cv2.goodFeaturesToTrack(gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=10)
 
        gray_prev = gray_curr
        pts_prev  = pts_curr
 
    if show_local_plots:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


# --- Main Pipeline Initializer ---

def run_pipeline(mode="KITTI", data_path=None, shared_queue=None):
    if mode == "KITTI":
        print("Processing KITTI feed")
        source = KittiSource(data_path)
    elif mode == "TELLO":
        print("Processing Tello feed")
        source = TelloSource()
    elif mode == "AUTOHAWK2A":
        print("Processing Autohawk2A feed")
        source = Autohawk2ASource()
    else:
        print(f"Unknown mode: {mode}")
        return
 
    estimated_total_frames = 300
    if mode == "KITTI" and data_path and os.path.exists(data_path):
        img_dir = os.path.join(data_path, "images")
        if os.path.isdir(img_dir):
            estimated_total_frames = len(os.listdir(img_dir))
 
    show_local_plots = (shared_queue is None)
 
    # FIX: Only spawn a background thread-buffer if reading from live hardware streams
    frame_buffer = None
    if mode != "KITTI":
        frame_buffer = ThreadFrameBuffer(source, maxsize=32)
        frame_buffer.start()
 
    try:
        _vo_worker(frame_buffer, source, mode, shared_queue, show_local_plots, estimated_total_frames)
    finally:
        if frame_buffer is not None:
            frame_buffer.stop()

            
if __name__ == "__main__":
    kitti_sequence_directory = "./data/Kitti/flight_path_00"
    run_pipeline(mode="KITTI", data_path=kitti_sequence_directory)