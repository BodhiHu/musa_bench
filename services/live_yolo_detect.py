import argparse
from datetime import datetime
from enum import Enum
import queue
import time
from typing import Dict, List
import torch
import torch_musa
import torch_musa.cuda_compat
import cv2
import os
import requests
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Response, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from screeninfo import get_monitors
from pathlib import Path
import threading
import multiprocessing as mproc
from multiprocessing import shared_memory


DEFAULT_STREAM_URL = "http://192.168.164.136:8686/video"
STREAM_URL = os.getenv("STREAM_URL", DEFAULT_STREAM_URL)
VERBOSE    = bool(os.getenv("VERBOSE", False))

FILE_PATH = Path(__file__).resolve()
FILE_DIR = FILE_PATH.parent
IMAGES_PATH = FILE_DIR / "../torch_yolo/images"

device = "musa:0"

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detector services")
    parser.add_argument("--live-on-local", action="store_true")
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

args = parse_args()
if args.verbose:
    VERBOSE = True

model_kwargs = {
    "imgsz"              : args.imgsz,
    "half"               : True,
    "device"             : device,
    "use_graph"          : False,
    "preprocess_device"  : "cpu",
    "postprocess_device" : "cpu",
    "verbose"            : VERBOSE
}


class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def print_nested_types(obj, depth=0):
    indent = "  " * depth
    print(f"{indent}{type(obj).__name__}")
    if isinstance(obj, (list, tuple)):
        for item in obj:
            print_nested_types(item, depth + 1)


def sleep_ms(ms: int | float):
    time.sleep(ms / 1000)


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
    print(f"INFO: loading model ...")
    model = YOLO("yolov8m.pt")
    model.fuse()
    model.model = model.model.half()
    model.model.to(device)
    print(f"INFO: warming up model ...")
    for _ in range(3):
        model.predict(IMAGES_PATH / "行者 - 1920x1280.jpg", **model_kwargs)

    model.predictor._lock = EmptyContextManager()
    print(f"INFO: model ready ✔")

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
                verbose=VERBOSE
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

def put_to_queue(tag: str, queue: queue.Queue, data):
    try:
        if queue.full():
            # print(f"WARNING: [{tag}] is full, will remove and skip the first frame in queue")
            try:
                queue.get_nowait()
            except:
                pass

        queue.put(data, block=False)
    except Exception as exc:
        print(f"WARNING: [{tag}] put to queue faield, ignoring:", exc)


class QueuedStream:
    next_stream_idx = 0

    def __init__(self):
        self.idx                 = QueuedStream.next_stream_idx
        self.stream_reader       = mjpeg_stream_reader()
        self.input_frames_queue  = queue.Queue(maxsize=30)
        self.inference_queue     = queue.Queue(maxsize=30)
        self.model_output_queue  = queue.Queue(maxsize=30)
        self.output_queue        = queue.Queue(maxsize=30)
        self.stop_event          = threading.Event()

        QueuedStream.next_stream_idx += 1

        print(f"INFO: created new stream = {self.idx}")


streams: Dict[int, QueuedStream] = {}
streams_lock = threading.Lock()
stop_event = threading.Event()


def add_stream():
    with streams_lock:
        stream = QueuedStream()
        streams[stream.idx] = stream

    print(f"INFO: added new stream: {stream.idx}")
    return stream


def remove_stream(stream_id: int):
    with streams_lock:
        if stream_id in streams:
            streams[stream_id].stop_event.set()
            del streams[stream_id]

    print(f"INFO: removed stream: {stream_id}")


def yolo_preprocess_worker():
    while not stop_event.is_set():
        empty = True
        with streams_lock:
            # shallow copy
            _streams = dict(streams)

        for s_idx, stream in _streams.items():

            frame = next(stream.stream_reader)

            # try:
            #     frame = stream.input_frames_queue.get(block=False)
            #     empty = False
            # except queue.Empty:
            #     continue

            if frame is None:
                continue

            print(f">>>>> PREDICT PREPROCESS")
            results = model.predict(frame, **model_kwargs, phase='preprocess')
            data = (frame, results)
            put_to_queue(f"inference_queue[{s_idx}]", stream.inference_queue, data)

        if empty:
            sleep_ms(10)


def yolo_inference_worker():
    while not stop_event.is_set():
        empty = True
        with streams_lock:
            # shallow copy
            _streams = dict(streams)

        for s_idx, stream in _streams.items():
            try:
                (frame, phase_input) = stream.inference_queue.get(block=False)
                empty = False
            except queue.Empty:
                continue

            pred_start  = time.time()
            print(f">>>>> PREDICT INFERENCE")
            results = model.predict(frame, **model_kwargs, phase_input=phase_input, phase='inference')
            pred_end = time.time()
            pred_time = pred_end - pred_start
            fps = 1 / pred_time
            data = (frame, results, fps)
            put_to_queue(f"model_output_queue[{s_idx}]", stream.model_output_queue, data)

        if empty:
            # print("WARN: all stream inference queues are empty, will wait 10ms for new requests")
            sleep_ms(10)


def yolo_postprocess_worker():
    while not stop_event.is_set():
        empty = True
        with streams_lock:
            # shallow copy
            _streams = dict(streams)

        for s_idx, stream in _streams.items():
            try:
                (frame, phase_input, fps) = stream.model_output_queue.get(block=False)
                empty = False
            except queue.Empty:
                continue

            print(f">>>>> PREDICT POSTPROCESS")
            results = model.predict(frame, **model_kwargs, phase_input=phase_input, phase='postprocess')
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Encode to JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            put_to_queue(f"output_queue[{s_idx}]", stream.output_queue, frame_bytes)

        if empty:
            sleep_ms(10)


def yolo_pipelined_stream(stream: QueuedStream):
    while not stop_event.is_set() and not stream.stop_event.is_set():
        try:
            frame_bytes = stream.output_queue.get(block=True)
        except queue.Empty:
            sleep_ms(10)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def yolo_synced_stream():
    prev_time = None
    fps = 0
    pre_endtime = None

    for frame in mjpeg_stream_reader():
        if frame is None:
            continue

        if VERBOSE and pre_endtime:
            print(f"jpeg read time: {((time.time() - pre_endtime)*1000):.2f}ms")

        kwargs = {
            "imgsz"              : args.imgsz,
            "half"               : True,
            "use_graph"          : False,
            "preprocess_device"  : "cpu",
            "postprocess_device" : "cpu",
            "verbose"            : VERBOSE
        }

        print(f">>>>> input frame: {type(frame)}")
        phase_input = None
        pre_out     = model.predict(frame, **kwargs, phase_input=phase_input, phase='preprocess')
        print(f">>>>> preprocess -> {type(pre_out)}, {type(pre_out[0])}")
        print_nested_types(pre_out)

        pred_start  = time.time()
        infer_out   = model.predict(frame, **kwargs, phase_input=pre_out, phase='inference')
        print(f">>>>> inference  -> {type(infer_out)}, {type(infer_out[0])}")
        print_nested_types(infer_out)
        pred_end    = time.time()

        results     = model.predict(frame, **kwargs, phase_input=infer_out, phase='postprocess')
        print(f">>>>> postprocs  -> {type(results)}, {type(results[0])}")

        pred_time = pred_end - pred_start
        fps = 1 / pred_time

        plot_time = time.time()
        # Draw results on frame
        annotated_frame = results[0].plot()

        if VERBOSE:
            cur_time = time.time()
            total_time = 0
            if prev_time is not None:
                total_time = cur_time - prev_time
            prev_time = cur_time
            print(f"fps = {fps:.2f}, model time = {(pred_time*1000):.2f}ms, time = {(total_time * 1000):.2f}ms")

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        print(f">>>>> final results: {type(jpeg)}, {type(frame_bytes)}")

        if VERBOSE:
            print(f"plots to jpeg time: {((time.time()-plot_time)*1000):.2f}ms")

        # Yield as MJPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        pre_endtime = time.time()


class YoloLiveMode(str, Enum):
    sync      = "sync"
    pipelined = "pipelined"

@app.get("/video/sync/{index}")
def yolo_sync(index: int = 0):
    return StreamingResponse(yolo_synced_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/pipelined/{index}")
def yolo_pipelined(index: int):

    stream = add_stream()

    # def read_frames_task(stream: QueuedStream):
    #     for frame in stream.stream_reader:
    #         if frame is None:
    #             break

    #         # frame = torch.from_numpy(frame).share_memory_()

    #         print(f">>>>> add new data to input_frames_queue[{stream.idx}]")
    #         put_to_queue(f"input_frames_queue[{stream.idx}]", stream.input_frames_queue, frame)

    # threading.Thread(target=read_frames_task, args=(stream,)).start()

    res = StreamingResponse(yolo_pipelined_stream(stream), media_type='multipart/x-mixed-replace; boundary=frame')

    async def on_close():
        remove_stream(stream_id=stream.idx)
    res.background = BackgroundTask(on_close)

    return res

@app.get("/", response_class=HTMLResponse)
def index(streams: int = Query(1), mode: YoloLiveMode = Query(YoloLiveMode.pipelined)):
    return """
    <html>
    <head>
        <title>YOLO Live</title>
        <style>
            .live-videos {
                width: 100%;
                display: flex;
                flex-direction: row;
                justify-content: flex-start;
                align-items: flex-start;
                gap: 8px;
                flex-wrap: wrap;
            }
            .live-videos img {
                max-width: calc(50% - 8px);
            }
        </style>
    """ + \
    f"""
        <script>
            var yolo_mode = "{mode}"
            var streams = {streams};
        </script>
    """ + \
    """
    </head>
    <body style="text-align:center;">
        <h2 style="">YOLO Live</h2>
        <div class="live-videos">
            """ + \
            f"""
            <img alt="image" src="/video/{mode}/0" />
            """ + \
            """
        </div>
    </body>
    <script>
        let img0 = document.querySelector('.live-videos > img')
        let new_img = img0.cloneNode()
        img0.onload = function() {
            for (let i = 1; i < streams; i++) {
                setTimeout(() => {
                    img = new_img.cloneNode()
                    img.src = `/video/${yolo_mode}/${i}`
                    img0.parentElement.insertBefore(img, img0)
                }, 3000 * i);
            }
        }
    </script>
    </html>
    """


@app.on_event("startup")
def start_threads():
    threading.Thread(target=yolo_preprocess_worker,  name="yolo_preprocess_worker",  daemon=True).start()
    threading.Thread(target=yolo_inference_worker,   name="yolo_inference_worker",   daemon=True).start()
    threading.Thread(target=yolo_postprocess_worker, name="yolo_postprocess_worker", daemon=True).start()


@app.on_event("shutdown")
def shutdown():
    stop_event.set()
    exit(0)


if __name__ == "__main__":

    if args.live_on_local:
        live_yolo_detect_stream(model, loop=True, pos=[5, 5])
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)

