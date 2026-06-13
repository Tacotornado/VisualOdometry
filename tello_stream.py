import time
import threading
import cv2
from djitellopy import Tello

width = 320
height = 240

class TelloStream:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self, url='udp://192.168.10.1:11111'):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True  # Fixed typo: was self.runnig
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
        
    def stop(self):
        self.running = False
        self.cap.release()
        
def main():
    me = Tello()
    me.connect()
    print(f"Battery: {me.get_battery()}%")
    
    me.streamoff()
    me.streamon()
    time.sleep(2)
    
    stream = TelloStream()
    stream.start()
    
    prev_time = time.time()
    
    while True:
        frame = stream.get_frame()
        if frame is None:  # Fixed: was == None
            continue
        
        img = cv2.resize(frame, (width, height))
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Fixed typo
        
        cv2.imshow("Drone Feed", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    stream.stop()
    cv2.destroyAllWindows()
    me.streamoff()
    
main()