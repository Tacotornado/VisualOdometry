import multiprocessing
import sys
# Import your fixed pipeline function
from vo_pipeline import run_pipeline

# Import your dashboard starter function here. For example, if using Flask:
# from web_dashboard import app
def start_dashboard_server(queue):
    """
    Worker process that boots up the web dashboard server 
    and handles incoming data packets from the VO pipeline.
    """
    print("[SYSTEM] Starting Web Dashboard Server...")
    
    # Example handling loop if your dashboard processes data via background threads:
    # If your dashboard platform requires passing the queue directly, pass it here.
    # e.g., app.run(host="0.0.0.0", port=5000)
    
    # For testing/debugging the queue bridge raw:
    while True:
        try:
            if not queue.empty():
                data = queue.get()
                # This is where your dashboard server consumes the data packet:
                # print(f"[DASHBOARD LOG] Frame: {data['frame_idx']}, Pos: {data['estimated']}")
                pass
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Multiprocessing context fix for Windows/macOS compatibility
    multiprocessing.freeze_support()
    
    # 1. Initialize the shared thread-safe communication queue
    shared_queue = multiprocessing.Queue()
    
    # 2. Select your pipeline execution mode ("KITTI", "TELLO", or "AUTOHAWK2A")
    # For testing, we default to KITTI sequence processing
    mode = "KITTI"
    kitti_path = "./data/Kitti/flight_path_00" 
    
    print(f"[SYSTEM] Initializing Visual Odometry System in {mode} mode...")

    # 3. Define the Dashboard Process
    dashboard_process = multiprocessing.Process(
        target=start_dashboard_server, 
        args=(shared_queue,)
    )
    
    # 4. Define the VO Pipeline Process
    pipeline_process = multiprocessing.Process(
        target=run_pipeline, 
        kwargs={"mode": mode, "data_path": kitti_path, "shared_queue": shared_queue}
    )
    
    # 5. Start both concurrently on separate CPU cores
    dashboard_process.start()
    pipeline_process.start()
    
    try:
        # Keep main thread alive while workers process camera tracking
        pipeline_process.join()
        dashboard_process.join()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Terminating parallel processing threads...")
        pipeline_process.terminate()
        dashboard_process.terminate()
        sys.exit(0)