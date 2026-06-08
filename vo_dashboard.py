import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Explicit import for 3D plotting spatial setups

# Import your pipeline backend runner
from vo_pipeline import run_pipeline 

class VODashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AUTOHAWK // Visual Odometry Control Station")
        self.root.geometry("450x650")
        
        # --- STYLING: Synthwave/Cyberpunk-inspired Palette ---
        self.bg_color = "#120e2e"       # Deep Dark Navy/Purple
        self.panel_color = "#1a153a"    # Lighter container purple
        self.accent_cyan = "#00f0ff"    # Electric Cyan
        self.accent_pink = "#ff007f"    # Hot Pink
        self.text_color = "#ffffff"     # White
        
        self.root.configure(bg=self.bg_color)
        
        # Configure custom ttk styles for a unified theme
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('.', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabelframe', background=self.panel_color, foreground=self.accent_cyan)
        self.style.configure('TLabelframe.Label', background=self.panel_color, foreground=self.accent_cyan, font=('Courier', 10, 'bold'))
        
        # Custom look for progress bar
        self.style.configure("Cyan.Horizontal.TProgressbar", troughcolor=self.bg_color, background=self.accent_cyan, thickness=15)

        # --- STATE MANAGEMENT ---
        self.vo_thread = None
        self.running = False
        self.data_queue = queue.Queue()
        self.start_time = 0
        
        # Purely tracking the actual calculated real positions
        self.estimated_vo_history = []
        
        self.setup_ui_layout()
        
    def setup_ui_layout(self):
        """Builds out the control interface widgets"""
        
        # 1. Header Banner
        header = tk.Label(self.root, text="AUTOHAWK VO CORE", font=("Courier", 18, "bold"), bg=self.bg_color, fg=self.accent_pink)
        header.pack(pady=15)
        
        # 2. Pipeline Data Source Selector Panel
        source_frame = ttk.LabelFrame(self.root, text=" 1. CHOOSE DATA STREAM ")
        source_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(source_frame, text="Active Mode:", bg=self.panel_color, fg=self.text_color).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.mode_var = tk.StringVar(value="KITTI")
        self.mode_combo = ttk.Combobox(source_frame, textvariable=self.mode_var, values=["KITTI", "TELLO", "AUTOHAWK2A"], state="readonly", width=15)
        self.mode_combo.grid(row=0, column=1, padx=10, pady=10)
        
        # 3. Hyperparameter Configuration Entry Fields
        param_frame = ttk.LabelFrame(self.root, text=" 2. TUNING HYPERPARAMETERS ")
        param_frame.pack(fill="x", padx=20, pady=10)
        
        self.param_fields = {
            "max_features": ("Max Target Features:", "500"),
            "window_size": ("LK Window Size:", "15"),
            "pyramid_level": ("Pyramid Levels:", "3"),
            "ransac_thresh": ("RANSAC Threshold:", "1.0")
        }
        self.entries = {}
        
        for idx, (key, (label_text, default_val)) in enumerate(self.param_fields.items()):
            tk.Label(param_frame, text=label_text, bg=self.panel_color, fg=self.text_color).grid(row=idx, column=0, padx=10, pady=5, sticky="w")
            entry = tk.Entry(param_frame, bg=self.bg_color, fg=self.accent_cyan, insertbackground=self.accent_cyan, borderwidth=1, relief="solid", width=12)
            entry.insert(0, default_val)
            entry.grid(row=idx, column=1, padx=10, pady=5, sticky="e")
            self.entries[key] = entry
            
        # 4. Diagnostics & Live Engine Metrics Reading Panel
        metrics_frame = ttk.LabelFrame(self.root, text=" 3. CORE TELEMETRY ")
        metrics_frame.pack(fill="x", padx=20, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(metrics_frame, length=200, mode='determinate', variable=self.progress_var, style="Cyan.Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, columnspan=2, padx=15, pady=10, sticky="ew")
        metrics_frame.columnconfigure(0, weight=1)
        
        self.lbl_status = tk.Label(metrics_frame, text="Engine Status: STANDBY", bg=self.panel_color, fg=self.text_color, font=("Courier", 10))
        self.lbl_status.grid(row=1, column=0, columnspan=2, pady=2, sticky="w", padx=10)

        self.lbl_distance = tk.Label(metrics_frame, text="Distance from Start: 0.00 m", bg=self.panel_color, fg=self.accent_pink, font=("Courier", 10, "bold"))
        self.lbl_distance.grid(row=2, column=0, columnspan=2, pady=2, sticky="w", padx=10)
        
        self.lbl_fps = tk.Label(metrics_frame, text="Processing Speed: 0.0 FPS", bg=self.panel_color, fg=self.accent_cyan, font=("Courier", 10))
        self.lbl_fps.grid(row=3, column=0, columnspan=2, pady=2, sticky="w", padx=10)

        # 5. Pipeline Primary Control Action Switches
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.btn_start = tk.Button(btn_frame, text="LAUNCH VO", font=("Courier", 11, "bold"), bg="#052e16", fg=self.accent_cyan, activebackground=self.accent_cyan, command=self.start_vo_engine, width=14, relief="flat")
        self.btn_start.grid(row=0, column=0, padx=10)
        
        self.btn_stop = tk.Button(btn_frame, text="HALT CORE", font=("Courier", 11, "bold"), bg="#450a0a", fg=self.accent_pink, state=tk.DISABLED, command=self.stop_vo_engine, width=14, relief="flat")
        self.btn_stop.grid(row=0, column=1, padx=10)

    # --- CONTROLLER METHODS ---
    
    def start_vo_engine(self):
        """Instantiates tracking queues and fires up the background pipeline thread"""
        self.running = True
        self.start_time = time.time()
        self.estimated_vo_history.clear()
        self.data_queue = queue.Queue()
        
        # Read user configurations from entries safely
        config = {
            "mode": self.mode_var.get(),
            "max_features": int(self.entries["max_features"].get()),
            "window_size": int(self.entries["window_size"].get()),
            "pyramid_level": int(self.entries["pyramid_level"].get()),
            "ransac_thresh": float(self.entries["ransac_thresh"].get())
        }
        
        # Switch button availability constraints
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Engine Status: RUNNING...", fg=self.accent_cyan)
        
        # Launch Worker Thread
        self.vo_thread = threading.Thread(target=self.worker_thread_loop, args=(config, self.data_queue))
        self.vo_thread.daemon = True
        self.vo_thread.start()
        
        # Begin checking the communication queue loop inside the main interface frame
        self.root.after(50, self.poll_queue_updates)

    def worker_thread_loop(self, config, data_q):
        """Runs isolated away from UI. Calls backend pipeline."""
        try:
            # --- FIXED: Point to your exact image sequences folder directory ---
            kitti_path = "./data/Kitti/flight_path_00" 
            
            # Passes execution down matching parameter expectations safely
            run_pipeline(mode=config["mode"], data_path=kitti_path, shared_queue=data_q)
        except Exception as e:
            print(f"Engine Thread Crash: {e}")
        finally:
            self.running = False

    def poll_queue_updates(self):
        """Drains data items from the worker thread to safely redraw widgets"""
        packet = None
        try:
            # Drain queue completely to parse the newest incoming packet data frame
            while True:
                packet = self.data_queue.get_nowait()
                if packet and "estimated" in packet: 
                    self.estimated_vo_history.append(packet["estimated"])
        except queue.Empty:
            pass  # Queue caught up entirely for this tick cycle
            
        # If we received data this loop cycle, paint it straight onto UI widgets
        if packet is not None:
            frame_idx = packet.get("frame_idx", 0)
            total_frames = packet.get("total_frames", 100)
            
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            self.progress_var.set(pct)
            
            if "distance_from_start" in packet:
                self.lbl_distance.config(text=f"Distance from Start: {packet['distance_from_start']:.2f} m")
            
            elapsed = time.time() - self.start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            self.lbl_fps.config(text=f"Processing Speed: {fps:.1f} Frames/sec")
            self.lbl_status.config(text=f"Engine Status: Frame {frame_idx}/{total_frames}")
            
        # If thread stopped processing and queue emptied out, wrap things up safely
        if not self.running and self.data_queue.empty():
            self.wrap_up_pipeline()
            return
            
        # Reschedule check loop to execute again in 50 milliseconds
        if self.running:
            self.root.after(50, self.poll_queue_updates)

    def stop_vo_engine(self):
        """Forces runtime cancellation signals to the background execution loop"""
        self.running = False
        self.lbl_status.config(text="Engine Status: ABORTED", fg=self.accent_pink)
        cv2.destroyAllWindows()
        self.wrap_up_pipeline()

    def wrap_up_pipeline(self):
        """Restores system control triggers and prints out performance figures"""
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        if len(self.estimated_vo_history) > 2:
            self.lbl_status.config(text="Engine Status: SUCCESS", fg="#22c55e")
            self.generate_trajectory_plot()
        else:
            self.lbl_status.config(text="Engine Status: STANDBY", fg=self.text_color)

    # --- TRUE 3D SPATIAL PATH PLOTTING ---

    def generate_trajectory_plot(self):
        """Generates a true 3D spatial plot of the traversed flight path"""
        est = np.array(self.estimated_vo_history)
        
        if est.ndim < 2 or len(est) < 2:
            messagebox.showinfo("Data Saved", "Flight tracking complete. Path data empty.")
            return

        # Build 3D Matplotlib Figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle("AUTOHAWK 3D FLIGHT TRAJECTORY", fontsize=12, fontweight='bold')
        
        # Plot the path line (X, Z, Y mapping for standard drone frame spatial dynamics)
        ax.plot(est[:, 0], est[:, 2], est[:, 1], color="#00f0ff", linewidth=2, label="Calculated VO Path")
        
        # Mark the takeoff point and current position point explicitly
        ax.scatter(0, 0, 0, color="green", s=100, label="Takeoff Pad (Origin)")
        ax.scatter(est[-1, 0], est[-1, 2], est[-1, 1], color="#ff007f", s=100, label="Final Drone Position")
        
        # Grid aesthetics
        ax.set_xlabel("X Position - Lateral (m)")
        ax.set_ylabel("Z Position - Depth (m)")
        ax.set_zlabel("Y Position - Altitude (m)")
        ax.grid(True, linestyle=":")
        ax.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = VODashboardApp(root)
    root.mainloop()