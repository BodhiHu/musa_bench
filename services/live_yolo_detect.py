import argparse
import torch
import torch_musa
import torch_musa.cuda_compat
import cv2
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from screeninfo import get_monitors
import threading


STREAM_URL = "http://192.168.164.136:8686/video"
device = "musa:0"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detector services")
    parser.add_argument("--live-on-local", action="store_true")
    parser.add_argument("--port", type=int, default=8686)
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    return parser.parse_args()

args = parse_args()


def get_cv_wind_size():
    monitor = get_monitors()[0]  # Primary monitor
    width = monitor.width // 2
    height = monitor.height // 2
    return (width - 10, height - 10)


def mjpeg_stream_reader():
    """Yield individual JPEG frames from an MJPEG stream."""
    response = requests.get(STREAM_URL, stream=True)
    bytes_data = b""

    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b"\xff\xd8")  # Start of JPEG
        b = bytes_data.find(b"\xff\xd9")  # End of JPEG

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]

            img_array = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            yield frame


def init_model():
    model = YOLO("yolov8m.pt")
    model.fuse()
    model.model = model.model.half()
    model.model.to(device)
    return model


def live_yolo_detect_stream(model: YOLO, loop = False, pos = [0, 0], idx = 0):
    title = f"Live Yolo Detection Stream - {idx}"
    size = get_cv_wind_size()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, pos[0], pos[1])
    cv2.resizeWindow(title, size[0], size[1])

    while True:
        for frame in mjpeg_stream_reader():
            if frame is None:
                break 

            # Run YOLO object detection
            results = model.predict(
                frame, imgsz=args.imgsz, half=True,
                use_graph=False,
                preprocess_device="cpu",
                postprocess_device="cpu",
                verbose=True
            )

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Show the frame
            cv2.imshow(title, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

        if not loop:
            break

    cv2.destroyAllWindows()


def yolo_detect_video(model: YOLO):
    cap = cv2.VideoCapture("./assets/温州道路行人_30fps.mp4")

    title = "Yolo Detection Video"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, 200, 200)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow(title, annotated_frame)
            # quit on "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


model = init_model()
app = FastAPI()
lock = threading.Lock()

def yolo_detect_frames():
    for frame in mjpeg_stream_reader():
        if frame is None:
            continue

        # Run YOLO object detection
        results = model.predict(
            frame, imgsz=args.imgsz, half=True,
            use_graph=False,
            preprocess_device="cpu",
            postprocess_device="cpu",
            verbose=True
        )

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        # Yield as MJPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/video")
def video_feed():
    return StreamingResponse(yolo_detect_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>YOLO Live</title>
    </head>
    <body style="text-align:center;">
        <h2 style="">YOLO Live</h2>
        <img src="/video" style="max-width: 100%; max-height: 100%;"/>
    </body>
    </html>
    """


if __name__ == "__main__":

    if args.live_on_local:
        live_yolo_detect_stream(model, loop=True, pos=[5, 5])
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
