import tkinter as tk
from tkinter import ttk, messagebox
import multiprocessing
import queue
import time
import os
import io
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vo_pipeline import run_pipeline


class VODashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AUTOHAWK // Visual Odometry Control Station")
        self.root.geometry("900x650")

        # Synthwave palette
        self.bg_color    = "#120e2e"
        self.panel_color = "#1a153a"
        self.accent_cyan = "#00f0ff"
        self.accent_pink = "#ff007f"
        self.text_color  = "#ffffff"
        self.root.configure(bg=self.bg_color)

        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('.', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabelframe', background=self.panel_color, foreground=self.accent_cyan)
        self.style.configure('TLabelframe.Label', background=self.panel_color,
                              foreground=self.accent_cyan, font=('Courier', 10, 'bold'))
        self.style.configure("Cyan.Horizontal.TProgressbar",
                              troughcolor=self.bg_color, background=self.accent_cyan, thickness=15)

        # State
        self.vo_process   = None
        self.running      = False
        self.data_queue   = None
        self.start_time   = 0
        self.estimated_vo_history = []

        # ── Video throttle: only redraw at ~20 fps ──────────────────────────
        self._last_video_ts  = 0.0
        self._video_interval = 1.0 / 20.0   # seconds between video redraws

        # ── Canvas throttle: only redraw path when new data arrives ─────────
        self._canvas_dirty   = False
        self._last_packet    = None

        self.setup_ui_layout()

    # ------------------------------------------------------------------
    def setup_ui_layout(self):
        header = tk.Label(self.root, text="AUTOHAWK VO CORE",
                          font=("Courier", 18, "bold"),
                          bg=self.bg_color, fg=self.accent_pink)
        header.pack(pady=10)

        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=15, pady=5)

        left_panel = ttk.Frame(main_container)
        left_panel.pack(side="left", fill="both", expand=True)

        right_panel = ttk.LabelFrame(main_container, text=" LIVE CAMERA STREAM ")
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=5)

        self.video_label = tk.Label(right_panel, bg="#0d0921")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Source selector
        source_frame = ttk.LabelFrame(left_panel, text=" 1. CHOOSE DATA STREAM ")
        source_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(source_frame, text="Active Mode:",
                 bg=self.panel_color, fg=self.text_color).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.mode_var   = tk.StringVar(value="KITTI")
        self.mode_combo = ttk.Combobox(source_frame, textvariable=self.mode_var,
                                       values=["KITTI", "TELLO", "AUTOHAWK2A"],
                                       state="readonly", width=15)
        self.mode_combo.grid(row=0, column=1, padx=10, pady=10)

        # Hyperparameters
        param_frame = ttk.LabelFrame(left_panel, text=" 2. TUNING HYPERPARAMETERS ")
        param_frame.pack(fill="x", padx=5, pady=5)
        self.param_fields = {
            "max_features":  ("Max Target Features:", "500"),
            "window_size":   ("LK Window Size:",      "15"),
            "pyramid_level": ("Pyramid Levels:",      "3"),
            "ransac_thresh": ("RANSAC Threshold:",    "1.0"),
        }
        self.entries = {}
        for idx, (key, (label_text, default_val)) in enumerate(self.param_fields.items()):
            tk.Label(param_frame, text=label_text,
                     bg=self.panel_color, fg=self.text_color).grid(
                         row=idx, column=0, padx=10, pady=4, sticky="w")
            entry = tk.Entry(param_frame, bg=self.bg_color, fg=self.accent_cyan,
                             insertbackground=self.accent_cyan,
                             borderwidth=1, relief="solid", width=12)
            entry.insert(0, default_val)
            entry.grid(row=idx, column=1, padx=10, pady=4, sticky="e")
            self.entries[key] = entry

        # Telemetry
        metrics_frame = ttk.LabelFrame(left_panel, text=" 3. CORE TELEMETRY ")
        metrics_frame.pack(fill="x", padx=5, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(metrics_frame, length=200, mode='determinate',
                                            variable=self.progress_var,
                                            style="Cyan.Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, columnspan=2, padx=15, pady=8, sticky="ew")
        metrics_frame.columnconfigure(0, weight=1)

        self.lbl_status   = tk.Label(metrics_frame, text="Engine Status: STANDBY",
                                     bg=self.panel_color, fg=self.text_color, font=("Courier", 10))
        self.lbl_status.grid(row=1, column=0, columnspan=2, pady=2, sticky="w", padx=10)

        self.lbl_distance = tk.Label(metrics_frame, text="Distance from Start: 0.00 m",
                                     bg=self.panel_color, fg=self.accent_pink, font=("Courier", 10, "bold"))
        self.lbl_distance.grid(row=2, column=0, columnspan=2, pady=2, sticky="w", padx=10)

        self.lbl_fps = tk.Label(metrics_frame, text="Processing Speed: 0.0 FPS",
                                bg=self.panel_color, fg=self.accent_cyan, font=("Courier", 10))
        self.lbl_fps.grid(row=3, column=0, columnspan=2, pady=2, sticky="w", padx=10)

        # Buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=10)
        self.btn_start = tk.Button(btn_frame, text="LAUNCH VO",
                                   font=("Courier", 11, "bold"),
                                   bg="#052e16", fg=self.accent_cyan,
                                   activebackground=self.accent_cyan,
                                   command=self.start_vo_engine, width=14, relief="flat")
        self.btn_start.grid(row=0, column=0, padx=10)
        self.btn_stop = tk.Button(btn_frame, text="HALT CORE",
                                  font=("Courier", 11, "bold"),
                                  bg="#450a0a", fg=self.accent_pink,
                                  state=tk.DISABLED,
                                  command=self.stop_vo_engine, width=14, relief="flat")
        self.btn_stop.grid(row=0, column=1, padx=10)

        # Mini-map
        map_frame = ttk.LabelFrame(left_panel, text=" 4. LIVE FLIGHT TRACK (2D FLOOR PLAN) ")
        map_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.map_canvas = tk.Canvas(map_frame, bg="#0d0921",
                                    highlightthickness=1,
                                    highlightbackground=self.accent_cyan)
        self.map_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.map_canvas.bind("<Configure>", lambda e: self.clear_and_draw_grid())

    # ------------------------------------------------------------------
    # ENGINE CONTROL
    # ------------------------------------------------------------------

    def start_vo_engine(self):
        self.running = True
        self.start_time = time.time()
        self.estimated_vo_history.clear()
        self.video_label.config(image='')
        self._last_video_ts = 0.0
        self._canvas_dirty  = False

        self.data_queue = multiprocessing.Queue(maxsize=10)   # bounded — prevents memory bloat

        config = {
            "mode":          self.mode_var.get(),
            "max_features":  int(self.entries["max_features"].get()),
            "window_size":   int(self.entries["window_size"].get()),
            "pyramid_level": int(self.entries["pyramid_level"].get()),
            "ransac_thresh": float(self.entries["ransac_thresh"].get()),
        }
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Engine Status: RUNNING...", fg=self.accent_cyan)

        self.vo_process = multiprocessing.Process(
            target=worker_process_wrapper,
            args=(config, self.data_queue),
        )
        self.vo_process.start()
        self.root.after(50, self.poll_queue_updates)

    # ------------------------------------------------------------------
    # QUEUE POLLING  — only does work that actually changed
    # ------------------------------------------------------------------

    def poll_queue_updates(self):
        if self.data_queue is None:
            return

        packet = None
        frames_drained = 0

        # Drain the queue but cap at 5 per tick so we don't freeze the UI
        try:
            while frames_drained < 15:
                pkt = self.data_queue.get_nowait()
                frames_drained += 1
                if pkt and "estimated" in pkt:
                    self.estimated_vo_history.append(pkt["estimated"])
                    self._canvas_dirty = True
                packet = pkt   # keep the last one for telemetry/video
        except queue.Empty:
            pass

        if packet is not None:
            self._last_packet = packet
            self._update_telemetry(packet)

            # Video: throttle to _video_interval even if queue floods
            now = time.time()
            if (packet.get("video_frame") is not None
                    and now - self._last_video_ts >= self._video_interval):
                self._render_video_frame(packet["video_frame"])
                self._last_video_ts = now

            # Canvas: only redraw when new path data arrived
            if self._canvas_dirty:
                self.update_live_canvas_plot()
                self._canvas_dirty = False

        process_alive = self.vo_process and self.vo_process.is_alive()
        if not process_alive and self.data_queue.empty():
            self.running = False
            self.wrap_up_pipeline()
            return

        if self.running or process_alive:
            self.root.after(30, self.poll_queue_updates)

    def _update_telemetry(self, packet):
        frame_idx   = packet.get("frame_idx", 0)
        total_frames = packet.get("total_frames", 100)
        pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        self.progress_var.set(pct)
        if "distance_from_start" in packet:
            self.lbl_distance.config(
                text=f"Distance from Start: {packet['distance_from_start']:.2f} m")
        elapsed = time.time() - self.start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        self.lbl_fps.config(text=f"Processing Speed: {fps:.1f} Frames/sec")
        self.lbl_status.config(text=f"Engine Status: Frame {frame_idx}/{total_frames}")

    def _render_video_frame(self, jpeg_bytes):
        """Decode JPEG and blit to label — runs on UI thread, kept minimal."""
        try:
            image   = Image.open(io.BytesIO(jpeg_bytes))
            lbl_w   = self.video_label.winfo_width()
            lbl_h   = self.video_label.winfo_height()
            if lbl_w > 10 and lbl_h > 10:
                # BILINEAR is faster than LANCZOS and imperceptible at this size
                image = image.resize((lbl_w, lbl_h), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(image)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except Exception as e:
            print(f"Video render error: {e}")

    # ------------------------------------------------------------------
    # ENGINE TEARDOWN
    # ------------------------------------------------------------------

    def stop_vo_engine(self):
        self.running = False
        if self.vo_process and self.vo_process.is_alive():
            self.vo_process.terminate()
            self.vo_process.join()
        self.lbl_status.config(text="Engine Status: ABORTED", fg=self.accent_pink)
        self.wrap_up_pipeline()

    def wrap_up_pipeline(self):
        if self.btn_start['state'] == tk.NORMAL:
            return
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if len(self.estimated_vo_history) > 2:
            self.lbl_status.config(text="Engine Status: SUCCESS", fg="#22c55e")
            self.root.after(100, self.generate_trajectory_plot)
        else:
            self.lbl_status.config(text="Engine Status: STANDBY", fg=self.text_color)

    # ------------------------------------------------------------------
    # 3-D PLOT
    # ------------------------------------------------------------------

    def generate_trajectory_plot(self):
        est = np.array(self.estimated_vo_history)
        if est.ndim < 2 or len(est) < 2:
            messagebox.showinfo("Data Saved", "Path data empty.")
            return

        x_coords, y_coords, z_coords = est[:, 0], est[:, 1], est[:, 2]
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        fig.suptitle("AUTOHAWK // VISUAL ODOMETRY FLIGHT PATH",
                     fontsize=12, fontweight='bold', color="#00f0ff")

        colors = plt.cm.cool(np.linspace(0, 1, len(est)))
        for i in range(len(est) - 1):
            ax.plot(x_coords[i:i+2], z_coords[i:i+2], y_coords[i:i+2],
                    color=colors[i], linewidth=2.5, alpha=0.9)

        floor_level = np.min(y_coords) - 1.0
        ax.plot(x_coords, z_coords, zs=floor_level, zdir='z',
                color="#3a2f6b", linestyle="--", linewidth=1.5, alpha=0.6,
                label="Ground Projection")
        ax.scatter(x_coords[0],  z_coords[0],  y_coords[0],
                   color="#22c55e", s=120, edgecolors='w', label="Origin")
        ax.scatter(x_coords[-1], z_coords[-1], y_coords[-1],
                   color="#ff007f", s=120, edgecolors='w', label="Current Position")
        ax.plot([x_coords[-1], x_coords[-1]], [z_coords[-1], z_coords[-1]],
                [floor_level, y_coords[-1]], color="#ff007f", linestyle=":", linewidth=1.5)

        ax.set_xlabel("X – Lateral (m)",  labelpad=10, color="#ffffff")
        ax.set_ylabel("Z – Depth (m)",    labelpad=10, color="#ffffff")
        ax.set_zlabel("Y – Altitude (m)", labelpad=10, color="#ffffff")

        max_range = np.array([
            x_coords.max() - x_coords.min(),
            z_coords.max() - z_coords.min(),
            y_coords.max() - y_coords.min(),
        ]).max() / 2.0
        mid_x = (x_coords.max() + x_coords.min()) * 0.5
        mid_y = (y_coords.max() + y_coords.min()) * 0.5
        mid_z = (z_coords.max() + z_coords.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)

        ax.xaxis.set_pane_color((0.07, 0.05, 0.18, 1.0))
        ax.yaxis.set_pane_color((0.07, 0.05, 0.18, 1.0))
        ax.zaxis.set_pane_color((0.10, 0.08, 0.23, 1.0))
        ax.grid(True, linestyle=":", alpha=0.3, color="#00f0ff")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MINI-MAP
    # ------------------------------------------------------------------

    def clear_and_draw_grid(self):
        self.map_canvas.delete("all")
        w = self.map_canvas.winfo_width()
        h = self.map_canvas.winfo_height()
        self.map_canvas.create_line(0, h//2, w, h//2, fill="#1f1a4a", dash=(2, 4))
        self.map_canvas.create_line(w//2, 0, w//2, h, fill="#1f1a4a", dash=(2, 4))
        self.map_canvas.create_text(10, 10, text="ORIGIN (0,0)",
                                    fill="#22c55e", anchor="nw", font=("Courier", 8))

    def update_live_canvas_plot(self):
        if len(self.estimated_vo_history) < 2:
            return
        w  = self.map_canvas.winfo_width()
        h  = self.map_canvas.winfo_height()
        cx, cy = w // 2, h // 2
        self.clear_and_draw_grid()

        est    = np.array(self.estimated_vo_history)
        x_pts  = est[:, 0]
        z_pts  = est[:, 2]
        max_val = max(np.max(np.abs(x_pts)), np.max(np.abs(z_pts)), 0.01)
        scale_factor = (min(cx, cy) - 20) / max_val

        pixel_points = [
            (cx + int(x * scale_factor), cy - int(z * scale_factor))
            for x, z in zip(x_pts, z_pts)
        ]

        # Draw path as a single polyline — much faster than N create_line calls
        if len(pixel_points) >= 2:
            flat = [coord for pt in pixel_points for coord in pt]
            self.map_canvas.create_line(*flat, fill=self.accent_cyan, width=2)

        last_px, last_py = pixel_points[-1]
        self.map_canvas.create_oval(last_px-4, last_py-4, last_px+4, last_py+4,
                                    fill=self.accent_pink, outline="white")


# ---------------------------------------------------------------------------
# WORKER ENTRY POINT (called in subprocess)
# ---------------------------------------------------------------------------

def worker_process_wrapper(config, data_q):
    try:
        kitti_path = "./data/Kitti/flight_path_00" if config["mode"] == "KITTI" else None
        run_pipeline(mode=config["mode"], data_path=kitti_path, shared_queue=data_q)
    except Exception as e:
        print(f"Engine process crash: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app  = VODashboardApp(root)
    root.mainloop()