from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2

# can be: a video path | camera index
VIDEO_PATH = "your_video_1080p.mp4" # use .etc 0 for camera

app = FastAPI()
# Change index if needed
camera = cv2.VideoCapture(VIDEO_PATH)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/ping")
def ping():
    return "pong"
