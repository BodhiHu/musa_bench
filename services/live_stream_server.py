import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2

VIDEO_PATH     = "./assets/天津道路行人_30fps.mp4" # use 0 for camera
TARGET_WIDTH   = 1920
TARGET_HEIGHT  = 1080
TARGET_FPS     = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
JPEG_QUALITY   = 50  # Lower = more compression, 30–60 recommended for streaming

encode_param   = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

def generate_frames():
    # can be video path or camera index
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
            continue

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # Compress with lower JPEG quality
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Enforce 30 FPS
        elapsed = time.time() - start_time
        time.sleep(max(0, FRAME_INTERVAL - elapsed))


app = FastAPI()

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    ret = """
    <html>
    <head>
        <title>Live Stream</title>
        <style>
            body { text-align: center; background: #111; color: white; font-family: sans-serif; }
            h1 { margin-top: 20px; }
            img { border-radius: 8px; margin-top: 20px; max-width: 95%; height: auto; box-shadow: 0 0 20px #000; }
        </style>
    </head>
    <body>
        <h1>Live Stream View</h1>
    """ + \
        f"""
        <img src="/video" style="width: {TARGET_WIDTH}px; height: {TARGET_HEIGHT}px;" />
        """ + \
    """
    </body>
    </html>
    """

    return ret

