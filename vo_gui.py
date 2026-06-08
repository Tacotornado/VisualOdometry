import os
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

# Optional pipeline dependencies
try:
    import vo_pipeline
    VO_AVAILABLE = True
except ImportError:
    VO_AVAILABLE = False

try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False

# Global Layout Definitions
C_BG        = "#0d1117"
C_PANEL     = "#161b22"
C_BORDER    = "#30363d"
C_ACCENT    = "#1f6feb"
C_HIGHLIGHT = "#f78166"
C_SUCCESS   = "#3fb950"
C_WARNING   = "#d29922"
C_TEXT      = "#e6edf3"
C_SUBTEXT   = "#7d8590"
C_ENTRY     = "#0d1117"

F_TITLE     = ("Consolas", 13, "bold")
F_HEAD      = ("Consolas", 9,  "bold")
F_LABEL     = ("Consolas", 9)
F_VALUE     = ("Consolas", 10, "bold")
F_MONO      = ("Consolas", 8)
F_BTN       = ("Consolas", 10, "bold")


# Main Application Block

class VOApp:

    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title("Monocular Visual Odometry")
        self.root.configure(bg=C_BG)
        self.root.minsize(820, 600)

        # Execution States
        self.vo_thread  : threading.Thread | None = None
        self.running    : bool  = False
        self.pause      : bool  = False
        self.stop       : bool  = False
        self._t0        : float = 0.0

        self.folder_path  : str = ""
        self.tello_source = None

        self.Init_UI()

    
    # UI Core Builder Routines
    
    def Init_UI(self):
        # Top branding header bar
        bar = tk.Frame(self.root, bg=C_ACCENT, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="  ◈  MONOCULAR VISUAL ODOMETRY",
                 font=F_TITLE, fg=C_TEXT, bg=C_ACCENT
                 ).pack(side="left", padx=10)

        mod_text    = "VO module ✓"   if VO_AVAILABLE    else "VO module ✗ (stub mode)"
        mod_color   = C_SUCCESS       if VO_AVAILABLE    else C_WARNING
        tello_text  = "djitellopy ✓"  if TELLO_AVAILABLE else "djitellopy ✗"
        tello_color = C_SUCCESS       if TELLO_AVAILABLE else C_SUBTEXT

        tk.Label(bar, text=f"{tello_text}   {mod_text}",
                 font=F_MONO, fg=mod_color, bg=C_ACCENT
                 ).pack(side="right", padx=14)

        # Workspace panel framework
        content = tk.Frame(self.root, bg=C_BG)
        content.pack(fill="both", expand=True, padx=12, pady=10)

        left  = tk.Frame(content, bg=C_BG)
        right = tk.Frame(content, bg=C_BG)
        left .pack(side="left",  fill="both", expand=True, padx=(0, 6))
        right.pack(side="right", fill="both", expand=True, padx=(6, 0))

        self.Build_Source_Panel(left)
        self.Build_Params_Panel(left)
        self.Build_Control_Panel(right)
        self.Build_Progress_Panel(right)
        self.Build_Telemetry_Panel(right)

        # Safely trigger initial visibility state now that all panels are instantiated
        self.On_Source_Change()

        # Global system status display
        self.status_var = tk.StringVar(value="Ready — choose a data source and press Start VO")
        tk.Label(self.root, textvariable=self.status_var,
                 font=F_MONO, fg=C_SUBTEXT, bg=C_BORDER,
                 anchor="w", padx=10, pady=4
                 ).pack(fill="x", side="bottom")

    
    # Panel Modules
    
    def Build_Source_Panel(self, parent):
        frame = self.Create_Card(parent, "◈  DATA SOURCE")

        self.source_mode = tk.StringVar(value="KITTI")
        modes = [
            ("KITTI Dataset (offline)",  "KITTI"),
            ("Tello Drone (live)",        "TELLO"),
            ("AirSim (simulation)",       "AIRSIM"),
        ]
        for label, val in modes:
            tk.Radiobutton(
                frame, text=label, variable=self.source_mode, value=val,
                command=self.On_Source_Change,
                font=F_LABEL, fg=C_TEXT, bg=C_PANEL,
                selectcolor=C_ACCENT,
                activebackground=C_PANEL, activeforeground=C_TEXT,
            ).pack(anchor="w", padx=12, pady=2)

        # KITTI Sequence path controls
        self._kitti_frame = tk.Frame(frame, bg=C_PANEL)
        tk.Label(self._kitti_frame, text="Sequence folder:", font=F_LABEL,
                 fg=C_SUBTEXT, bg=C_PANEL).pack(anchor="w", padx=12)
        row = tk.Frame(self._kitti_frame, bg=C_PANEL)
        row.pack(fill="x", padx=12, pady=(0, 6))

        self.kitti_path = tk.StringVar(value="C:/data/KITTI_sequence_1/")
        tk.Entry(row, textvariable=self.kitti_path, font=F_MONO,
                 fg=C_TEXT, bg=C_ENTRY, insertbackground=C_TEXT,
                 relief="flat", bd=4
                 ).pack(side="left", fill="x", expand=True)
        tk.Button(row, text=" … ", font=F_BTN,
                  fg=C_TEXT, bg=C_ACCENT, relief="flat",
                  activebackground=C_HIGHLIGHT, activeforeground=C_TEXT,
                  command=self.Browse_Kitti
                  ).pack(side="right", padx=(4, 0))

        # Tello configuration metrics
        self._tello_frame = tk.Frame(frame, bg=C_PANEL)
        tk.Label(self._tello_frame, text="Drone status:", font=F_LABEL,
                 fg=C_SUBTEXT, bg=C_PANEL).pack(anchor="w", padx=12)
        self._tello_status = tk.StringVar(value="Not connected")
        self._tello_status_lbl = tk.Label(
            self._tello_frame, textvariable=self._tello_status,
            font=F_VALUE, fg=C_WARNING, bg=C_PANEL)
        self._tello_status_lbl.pack(anchor="w", padx=20)

        if not TELLO_AVAILABLE:
            tk.Label(self._tello_frame,
                     text="⚠  djitellopy not installed  →  pip install djitellopy",
                     font=F_MONO, fg=C_WARNING, bg=C_PANEL
                     ).pack(anchor="w", padx=20, pady=(0, 6))

        # Feature Engine Selectors
        sep = tk.Frame(frame, bg=C_BORDER, height=1)
        sep.pack(fill="x", padx=12, pady=6)
        det_row = tk.Frame(frame, bg=C_PANEL)
        det_row.pack(anchor="w", padx=12, pady=(0, 8))
        tk.Label(det_row, text="Feature detector:", font=F_LABEL,
                 fg=C_SUBTEXT, bg=C_PANEL).pack(side="left")

        self.detector_var = tk.StringVar(value="ORB")
        for name in ("ORB", "ShiTomasi"):
            tk.Radiobutton(
                det_row, text=name, variable=self.detector_var, value=name,
                font=F_LABEL, fg=C_TEXT, bg=C_PANEL,
                selectcolor=C_ACCENT,
                activebackground=C_PANEL, activeforeground=C_TEXT,
            ).pack(side="left", padx=10)

    def Build_Params_Panel(self, parent):
        frame = self.Create_Card(parent, "◈  VO PARAMETERS")

        param_defs = [
            ("n_features",    "Number of Features",     "500",  "pts"),
            ("win_size",      "Optical Flow Window",     "5",    "px"),
            ("pyr_levels",    "Pyramid Levels",          "5",    "lvl"),
            ("n_iter",        "Max Iterations",          "70",   "it"),
            ("inlier_thresh", "RANSAC Inlier Threshold", "3.0",  "px"),
        ]
        self._param_entries: dict[str, tk.Entry] = {}

        for key, label, default, unit in param_defs:
            row = tk.Frame(frame, bg=C_PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, font=F_LABEL, fg=C_SUBTEXT, bg=C_PANEL,
                     width=24, anchor="w").pack(side="left")
            e = tk.Entry(row, font=F_VALUE, fg=C_TEXT, bg=C_ENTRY,
                         insertbackground=C_TEXT, relief="flat", bd=4, width=7)
            e.insert(0, default)
            e.pack(side="left", padx=4)
            self._param_entries[key] = e
            tk.Label(row, text=unit, font=F_MONO, fg=C_SUBTEXT,
                     bg=C_PANEL).pack(side="left")

        # Reset to defaults button
        tk.Button(frame, text="↺  Reset to Defaults", font=F_MONO,
                  fg=C_SUBTEXT, bg=C_PANEL, relief="flat",
                  activebackground=C_BORDER, activeforeground=C_TEXT,
                  command=self.Reset_Params
                  ).pack(anchor="e", padx=12, pady=(4, 8))

    def Build_Control_Panel(self, parent):
        frame = self.Create_Card(parent, "◈  EXECUTION CONTROLS")
        grid = tk.Frame(frame, bg=C_PANEL)
        grid.pack(fill="x", padx=12, pady=6)

        self.start_btn  = tk.Button(grid, text="START VO", font=F_BTN,
                                    fg=C_TEXT, bg=C_SUCCESS, relief="flat",
                                    command=self.Start_Pipeline, width=14)
        self.pause_btn  = tk.Button(grid, text="PAUSE",    font=F_BTN,
                                    fg=C_TEXT, bg=C_WARNING, relief="flat",
                                    command=self.Toggle_Pause, width=14,
                                    state=tk.DISABLED)
        self.resume_btn = tk.Button(grid, text="RESUME",   font=F_BTN,
                                    fg=C_TEXT, bg=C_ACCENT, relief="flat",
                                    command=self.Toggle_Pause, width=14,
                                    state=tk.DISABLED)
        self.stop_btn   = tk.Button(grid, text="STOP",     font=F_BTN,
                                    fg=C_TEXT, bg=C_HIGHLIGHT, relief="flat",
                                    command=self.Stop_Pipeline, width=14,
                                    state=tk.DISABLED)

        self.start_btn .grid(row=0, column=0, padx=4, pady=4)
        self.pause_btn .grid(row=0, column=1, padx=4, pady=4)
        self.resume_btn.grid(row=1, column=0, padx=4, pady=4)
        self.stop_btn  .grid(row=1, column=1, padx=4, pady=4)

    def Build_Progress_Panel(self, parent):
        frame = self.Create_Card(parent, "◈  PIPELINE PROGRESS")

        self._stat_frame   = tk.StringVar(value="-")
        self._stat_fps     = tk.StringVar(value="-")
        self._stat_elapsed = tk.StringVar(value="-")

        stats = [
            ("Frame:",        self._stat_frame),
            ("FPS:",          self._stat_fps),
            ("Elapsed Time:", self._stat_elapsed),
        ]
        for label, var in stats:
            row = tk.Frame(frame, bg=C_PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, font=F_LABEL, fg=C_SUBTEXT, bg=C_PANEL,
                     width=18, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=F_VALUE, fg=C_TEXT,
                     bg=C_PANEL).pack(side="left")

    def Build_Telemetry_Panel(self, parent):
        self._telem_frame = self.Create_Card(parent, "◈  LIVE TELEMETRY (TELLO)")
        self._telem_frame.pack_forget()

        self._telem_vars: dict[str, tk.StringVar] = {}
        telems = [
            ("Battery Level:", "bat",     "%"),
            ("Barometer Alt:", "baro",    "cm"),
            ("Temperature:",   "temp",    "°C"),
            ("Speed Vector X:","speed_x", "cm/s"),
            ("Speed Vector Y:","speed_y", "cm/s"),
            ("Speed Vector Z:","speed_z", "cm/s"),
        ]
        for label, key, unit in telems:
            row = tk.Frame(self._telem_frame, bg=C_PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, font=F_LABEL, fg=C_SUBTEXT, bg=C_PANEL,
                     width=18, anchor="w").pack(side="left")
            v = tk.StringVar(value="-")
            self._telem_vars[key] = v
            tk.Label(row, textvariable=v, font=F_VALUE, fg=C_TEXT,
                     bg=C_PANEL).pack(side="left", padx=4)
            tk.Label(row, text=unit, font=F_MONO, fg=C_SUBTEXT,
                     bg=C_PANEL).pack(side="left")

    
    # Event State Handlers
   
    def On_Source_Change(self):
        mode = self.source_mode.get()
        self._kitti_frame.pack_forget()
        self._tello_frame.pack_forget()
        self._telem_frame.pack_forget()

        if mode == "KITTI":
            self._kitti_frame.pack(fill="x", pady=4)
        elif mode == "TELLO":
            self._tello_frame.pack(fill="x", pady=4)
            self._telem_frame.pack(fill="x", pady=4)  # always visible in TELLO mode

    def Browse_Kitti(self):
        p = filedialog.askdirectory(title="Select KITTI Sequence Folder")
        if p:
            self.kitti_path.set(p)

    
    # Parameter Helpers
    
    def Read_Params(self) -> dict | None:
        """Read and validate all VO parameter fields. Returns None on error."""
        try:
            return {
                "n_features":    int  (self._param_entries["n_features"].get()),
                "win_size":      int  (self._param_entries["win_size"].get()),
                "pyr_levels":    int  (self._param_entries["pyr_levels"].get()),
                "n_iter":        int  (self._param_entries["n_iter"].get()),
                "inlier_thresh": float(self._param_entries["inlier_thresh"].get()),
            }
        except ValueError:
            messagebox.showerror(
                "Invalid Parameters",
                "All parameter fields must be numeric.\n"
                "Press 'Reset to Defaults' to restore safe values.")
            return None

    def Reset_Params(self):
        """Restore all VO parameter entries to their default values."""
        defaults = {
            "n_features": "500", "win_size": "5", "pyr_levels": "5",
            "n_iter": "70", "inlier_thresh": "3.0",
        }
        for key, val in defaults.items():
            self._param_entries[key].delete(0, tk.END)
            self._param_entries[key].insert(0, val)

    
    # Pipeline Controls
   
    def Start_Pipeline(self):
        if self.running:
            return

        # 1. Validate parameter entries first
        params = self.Read_Params()
        if params is None:
            return

        # 2. Validate the data source
        mode = self.source_mode.get()
        if mode == "KITTI":
            self.folder_path = self.kitti_path.get()
            if not os.path.exists(self.folder_path):
                messagebox.showerror("Path Error",
                                     f"Folder not found:\n{self.folder_path}")
                return
        elif mode == "TELLO":
            if not TELLO_AVAILABLE:
                messagebox.showerror("Missing Library",
                                     "djitellopy is not installed.\n"
                                     "Run:  pip install djitellopy")
                return
            self.folder_path = ""
        else:
            self.folder_path = ""

        # 3. Arm state flags and update UI
        self.running = True
        self.pause   = False
        self.stop    = False
        self._t0     = time.time()

        self.Set_Buttons("running")
        self.Update_Status("Executing Odometry Pipeline…")

        if mode == "TELLO":
            self.Poll_Telemetry()

        # 4. Spin up background thread + start non-blocking queue poll
        q = queue.Queue()
        self.vo_thread = threading.Thread(
            target=self.Run_VO_Thread,
            args=(q, params, mode),
            daemon=True,
        )
        self.vo_thread.start()
        self.Poll_Queue(q)

    def Stop_Pipeline(self):
        if not self.running:
            return
        self.running = False
        self.stop    = True
        self.pause   = False
        self.Set_Buttons("idle")
        self.Reset_Progress()
        self.Update_Status("Pipeline stopped.")

    def Toggle_Pause(self):
        if not self.running:
            return
        self.pause = not self.pause
        if self.pause:
            self.Set_Buttons("paused")
            self.Update_Status("Pipeline paused.")
        else:
            self.Set_Buttons("running")
            self.Update_Status("Pipeline resumed.")

    
    # Background Thread & Queue
    
    def Run_VO_Thread(self, q: queue.Queue, params: dict, mode: str):
        """Runs the VO pipeline in a background thread (daemon)."""
        if not VO_AVAILABLE:
            # Stub: simulates 200-frame run so the UI can be tested without VO
            total = 200
            for i in range(total):
                if self.stop or not self.running:
                    break
                while self.pause:
                    time.sleep(0.05)
                time.sleep(0.04)
                elapsed = time.time() - self._t0 + 1e-9
                q.put((i, total, (i + 1) / elapsed))
            self.running = False
            return
        
        vo_pipeline.run_pipeline(
            mode=mode,
            data_path=self.folder_path,
            n_features=params["n_features"],
            win_size=params["win_size"],
            pyr_levels=params["pyr_levels"],
            n_iter=params["n_iter"],
            inlier_thresh=params["inlier_thresh"],
            detector=self.detector_var.get(),
            q=q
        )
        cv2.destroyAllWindows()
        self.running = False

        # Real pipeline call — passes all validated params and the detector choice
       # ground_truth = []
        #monocular_vo = []
       # visual_odometry.main(
           # q,
           # self.folder_path,
          #  ground_truth,
           # monocular_vo,
           # params["n_features"],
           # params["win_size"],
           # params["pyr_levels"],
           # params["n_iter"],
           # params["inlier_thresh"],
           # detector=self.detector_var.get(),
           # mode=mode,
       # )
       # cv2.destroyAllWindows()
       # self.running = False

    def Poll_Queue(self, q: queue.Queue):
        """Non-blocking queue drain using tk.after() — keeps the GUI responsive."""
        # Thread finished?
        if not (self.vo_thread and self.vo_thread.is_alive()):
            if self.running:      # ended naturally, not via Stop
                self.running = False
                self.VO_Finished()
            return

        # Consume one item if available
        try:
            frame_idx, total, fps = q.get_nowait()
            self.Update_Progress(frame_idx, total, fps)
        except queue.Empty:
            pass

        self.root.after(100, lambda: self.Poll_Queue(q))

    def VO_Finished(self):
        """Called when the VO thread exits on its own (not via Stop)."""
        self.Set_Buttons("idle")
        self.Update_Status("Pipeline finished — results shown in plot window(s).")

    
    # Progress Display
    
    def Update_Progress(self, frame_idx: int, total: int, fps: float):
        self._stat_frame  .set(f"{frame_idx + 1} / {total}")
        self._stat_fps    .set(f"{fps:.1f}")
        elapsed = time.time() - self._t0
        self._stat_elapsed.set(f"{elapsed:.1f} s")

    def Reset_Progress(self):
        self._stat_frame  .set("-")
        self._stat_fps    .set("-")
        self._stat_elapsed.set("-")

    
    # Telemetry Poll
    
    def Poll_Telemetry(self):
        if not self.running or self.source_mode.get() != "TELLO":
            return
        try:
            t = self.tello_source
            if t is None:
                raise ValueError("No Tello instance")
            self._telem_vars["bat"]    .set(str(t.get_battery()))
            self._telem_vars["baro"]   .set(str(t.get_barometer()))
            self._telem_vars["temp"]   .set(str(t.get_temperature()))
            self._telem_vars["speed_x"].set(str(t.get_velocity_x()))
            self._telem_vars["speed_y"].set(str(t.get_velocity_y()))
            self._telem_vars["speed_z"].set(str(t.get_velocity_z()))
        except Exception:
            pass
        self.root.after(500, self.Poll_Telemetry)

    
    # Core Infrastructure Helpers
    
    def Update_Status(self, msg: str):
        self.status_var.set(msg)

    def Set_Buttons(self, state: str):
        cfg = {
            #             start        pause        resume       stop
            "idle":    (tk.NORMAL,   tk.DISABLED, tk.DISABLED, tk.DISABLED),
            "running": (tk.DISABLED, tk.NORMAL,   tk.DISABLED, tk.NORMAL),
            "paused":  (tk.DISABLED, tk.DISABLED, tk.NORMAL,   tk.NORMAL),
        }[state]
        self.start_btn .config(state=cfg[0])
        self.pause_btn .config(state=cfg[1])
        self.resume_btn.config(state=cfg[2])
        self.stop_btn  .config(state=cfg[3])

    def Create_Card(self, parent, title: str) -> tk.LabelFrame:
        f = tk.LabelFrame(
            parent, text=f" {title} ", font=F_HEAD,
            fg=C_TEXT, bg=C_PANEL, bd=1, relief="solid",
            padx=6, pady=6
        )
        f.pack(fill="x", pady=6)
        return f


# System Entrypoint Execution

if __name__ == "__main__":
    app_root = tk.Tk()
    app = VOApp(app_root)
    app_root.mainloop()