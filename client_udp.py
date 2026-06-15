import socket
import struct
import cv2
import numpy as np

ESP_IP = "192.168.43.42"
LOCAL_IP = "0.0.0.0"
PORT = 5000
CHUNK_HDR = 6

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, PORT))
sock.settimeout(3.0)
sock.sendto(b"OPEN", (ESP_IP, PORT))
data = b""
frames = {}
while True:
    packet, _ = sock.recvfrom(1500)

    frame_id, chunk_id, total_chunks = struct.unpack("<HHH", packet[:CHUNK_HDR])
    data = packet[CHUNK_HDR:]

    if frame_id not in frames:
        frames[frame_id] = {
            "chunks": {},
            "total": total_chunks
        }
        
    frames[frame_id]["chunks"][chunk_id] = data
    if len(frames[frame_id]["chunks"]) == total_chunks:
        full = b"".join(
            frames[frame_id]["chunks"][i] for i in range(total_chunks)
        )

        img = cv2.imdecode(
            np.frombuffer(full, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is not None:
            cv2.imshow("ESP Stream", img)
        
        del frames[frame_id]

    if cv2.waitKey(1) == 27:
        break

sock.sendto(b"CLOSE", (ESP_IP, PORT))
sock.close()
cv2.destroyAllWindows()