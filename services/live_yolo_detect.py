import signal
import os, sys
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.abspath(f"{current_dir}/.."))
sys.path.append(os.path.abspath(f"{current_dir}/../.."))

import argparse
from datetime import datetime
from enum import Enum
import queue
import time
from typing import Dict, List
import torch
import torch_musa
import torch_musa.cuda_compat
import torch.nn.functional as nnf
import cv2
import requests
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Response, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from ultralytics.models import yolo
from screeninfo import get_monitors
from pathlib import Path
import threading
import multiprocessing as mproc
from multiprocessing import shared_memory as shm
from musa_bench.mtnn import MtnnYOLOModel


DEFAULT_STREAM_URL = "http://192.168.164.136:8686/video"
STREAM_URL = os.getenv("STREAM_URL", DEFAULT_STREAM_URL)
VERBOSE    = bool(os.getenv("VERBOSE", False))

FILE_PATH = Path(__file__).resolve()
FILE_DIR = FILE_PATH.parent
IMAGES_PATH = FILE_DIR / "../torch_yolo/images"

device = "musa:0"

DEFAULT_MP_MODE = 'process' # 'process' or 'thread'

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detector services")
    parser.add_argument("--live-on-local", action="store_true")
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--imgsz", type=int, default=640, choices=[640, 320],
                        help="GPU model input tensor size")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mp-mode", type=str, choices=["thread", "process"], default=DEFAULT_MP_MODE,
                        help="using threads or processes for parallel processing")
    return parser.parse_args()

args = parse_args()
if args.verbose:
    VERBOSE = True

if args.mp_mode == 'process':
    """use processes for multiprocessing"""
    from multiprocessing import Event, Process as Proc, Queue
    print("INFO: using multiprocessing for parallel processing")
else: # 'thread' mode
    """ use threads for multiprocessing """
    from threading import Event, Thread as Proc
    from queue import Queue
    print("INFO: using threads for parallel processing")


mproc_states_manager = mproc.Manager()

gpu_model_cfgs = {
    "imgsz"              : args.imgsz,
    "half"               : True,
    "device"             : device,
    "use_graph"          : False,
    "preprocess_device"  : "cpu",
    "postprocess_device" : "cpu",
    "verbose"            : VERBOSE
}

npu_model_cfgs = dict(gpu_model_cfgs)
npu_model_cfgs["imgsz"] = 640
npu_model_cfgs["half"] = False

cpu_model_cfgs = dict(gpu_model_cfgs)
cpu_model_cfgs["device"] = "cpu"
cpu_model_cfgs["half"] = False


class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def print_nested_types(obj, depth=0):
    indent = "  " * depth
    shape = ""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        shape = f" {obj.shape}"
    print(f"{indent}{type(obj).__name__}{shape}")
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


def mjpeg_stream_reader(tag = ""):
    """Yield individual JPEG frames from an MJPEG stream."""
    with requests.get(STREAM_URL, stream=True) as response:
        bytes_data = b""
        print(f"INFO: {tag}opened new stream connection")

        try:
            fps = 0
            last_time = None
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

                    if VERBOSE:
                        cur_time = time.time()
                        if last_time is not None:
                            delta = cur_time - last_time
                            fps = 1 / delta
                        last_time = cur_time
                        print(f"-> {tag}incoming fps  = {fps:.2f}")
        finally:
            response.close()
            print(f"INFO: {tag}closed stream connection")


def init_gpu_model():
    print(f"INFO: loading gpu model ...")
    model = YOLO("yolov8m.pt")
    model.fuse()
    model.model = model.model.half()
    model.model.to(device)

    print(f"INFO: warming up gpu model ...")
    for _ in range(1):
        model.predict(IMAGES_PATH / "行者 - 1920x1080.jpg", **gpu_model_cfgs)

    model.predictor._lock = EmptyContextManager()
    print(f"INFO: gpu model ready ✔")

    return model


def init_cpu_model():
    print(f"INFO: loading cpu model ...")
    model = YOLO("yolov8m.pt")
    model.to("cpu")
    model.fuse()

    print(f"INFO: warming up cpu model ...")
    for _ in range(1):
        model.predict(IMAGES_PATH / "行者 - 1920x1080.jpg", **cpu_model_cfgs)

    print(f"INFO: cpu model ready ✔")

    return model


def init_npu_model(model_path, core_id):
    print(f"INFO: loading npu model ...")
    npu_model = MtnnYOLOModel(model_path, core_id)
    print(f"INFO: npu model ready ✔")
    return npu_model


def live_yolo_detect_stream(loop = False, pos = [0, 0], idx = 0):
    model: YOLO = init_gpu_model()

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


def put_to_queue(tag: str, queue: Queue, data):
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


def assert_ndarray_on_shm(nd: np.ndarray):
    assert not nd.flags.owndata


class QueuedStream:

    def __init__(self, idx: int):
        self.idx                 = idx
        self.frame_reader_proc   = None
        self.stream_started      = False
        self.input_frames_queue  = Queue(maxsize=120)
        self.inference_queue     = Queue(maxsize=30)
        self.model_output_queue  = Queue(maxsize=30)
        self.output_queue        = Queue(maxsize=30)
        self.stop_event          = Event()

        self.stats               = mproc_states_manager.dict({

            "processed_frames"   : 0,
            "total_time"         : 0,
            "last_time"          : None,
            "fps"                : 0,
            "model_fps"          : 0,
            "npu_model_fps"      : 0,
            "prep_fps"           : 0,
            "post_fps"           : 0,
        })

        # self.processed_frames    = 0
        # self.total_time          = 0
        # self.last_time           = None
        # self.fps                 = 0
        # self.model_fps           = 0
        # self.npu_model_fps       = 0
        # self.prep_fps            = 0
        # self.post_fps            = 0

        print(f"INFO: created stream = {self.idx}")

    def start(self):
        if self.frame_reader_proc:
            self.stop()

        while self.stop_event.is_set():
            print(f"INFO: waiting for stream[{self.idx}] to be closed before starting ...")
            time.sleep(0.25)

        print(f"INFO: stream[{self.idx}] starting frames input worker ...")

        def frame_read_worker(idx: int, input_frames_queue: Queue, stop_event):
            signal.signal(signal.SIGINT, signal.SIG_DFL)

            while not stop_event.is_set():
                for frame in mjpeg_stream_reader(f"stream[{idx}] "):
                    if frame is None:
                        continue

                    if stop_event.is_set():
                        break

                    # put to shared memory for multi-processing
                    frame: np.ndarray = torch.from_numpy(frame).share_memory_().numpy()
                    assert_ndarray_on_shm(frame)
                    put_to_queue(f'input_frames_queue[{self.idx}]', input_frames_queue, frame)

            print(f"INFO: stream[{self.idx}] frames input worker will stop")
            stop_event.clear()

        self.frame_reader_proc = mproc.Process(
            target=frame_read_worker, name=f"frame_read_worker[{self.idx}]", daemon=True,
            kwargs={ "idx": self.idx, "input_frames_queue": self.input_frames_queue, "stop_event": self.stop_event }
        )
        self.frame_reader_proc.start()
        self.stream_started = True

    def stop(self):
        self.stop_event.set()
        if self.frame_reader_proc:
            self.frame_reader_proc.kill()
            self.frame_reader_proc = None
        self.stream_started = False

streams: Dict[int, QueuedStream] = {
    0: QueuedStream(0),
    1: QueuedStream(1),
    2: QueuedStream(2),
    3: QueuedStream(3),
    # 4: QueuedStream(4),
    # 5: QueuedStream(5),
    # 6: QueuedStream(6),
    # 7: QueuedStream(7),
}

device_stats = mproc_states_manager.dict({
    "musa:0:model_fps" : 0,
    "npu:0:model_fps"  : 0,
    "npu:1:model_fps"  : 0,
})

def active_streams():
    return {k: v for k, v in streams.items() if v.stream_started}

stop_event = Event()


def exit_on_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Exception caught in {func.__name__}]: {e}")
            traceback.print_exc()  # prints the full traceback to stderr
            os._exit(-1)
    return wrapper


def yolo_preprocess_worker(stop_event, w_streams: Dict[int, Dict]):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    model: YOLO = init_cpu_model()

    while not stop_event.is_set():
        empty = True

        for s_idx, w_stream in w_streams.items():
            start_time = time.time()

            try:
                frame: np.ndarray = w_stream["input_frames_queue"].get(block=False)
                if frame is None:
                    continue
            except queue.Empty:
                continue

            assert_ndarray_on_shm(frame)

            empty = False

            # Tensor of shape [1, 3, 192, 320]
            _t = time.time()
            # preds_tensor: torch.Tensor = model.predict(frame, phase='preprocess', **gpu_model_cfgs)[0]
            prep_tensor: torch.Tensor = model.predictor.preprocess(frame, **gpu_model_cfgs)
            assert prep_tensor.device.type == 'cpu'
            prep_tensor = prep_tensor.share_memory_()
            gpu_tensors = [prep_tensor]
            if VERBOSE:
                print(f":: model preprocess took {((time.time() - _t)*1000):.2f}ms")

            # pad for NPU #############################################################################
            """center and reshape to 640x640"""
            # **MUST RESHAPE to 640x640** for NPU inference, no limits for GPU
            # npu_tensor: torch.Tensor = model.predict(frame, phase='preprocess', **npu_model_cfgs)[0]
            # npu_tensor = npu_tensor.float()
            npu_tensor: torch.Tensor = model.predictor.preprocess(frame, **npu_model_cfgs)
            assert npu_tensor.device.type == 'cpu'
            assert npu_tensor.dtype == torch.float32
            assert npu_tensor.shape[-1] == 640
            npu_tensor = npu_tensor
            target_h, target_w = 640, 640
            h, w = npu_tensor.shape[2], npu_tensor.shape[3]
            # Compute padding
            pad_h = target_h - h  # 448
            pad_w = target_w - w  # 320
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # Apply padding: (left, right, top, bottom)
            tensor = nnf.pad(npu_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            assert tensor.device.type == 'cpu'
            npu_tensors = [tensor.contiguous().share_memory_()]
            ###########################################################################################

            data = (frame, gpu_tensors, npu_tensors)
            assert_ndarray_on_shm(frame)
            assert gpu_tensors[0].is_shared()
            assert npu_tensors[0].is_shared()
            put_to_queue(f"inference_queue[{s_idx}]", w_stream["inference_queue"], data)

            delta = time.time() - start_time
            streams[s_idx].stats["prep_fps"] = 1 / delta

        if empty:
            sleep_ms(10)

@exit_on_error
def yolo_inference_worker(stop_event, device: str, w_streams: Dict[int, Dict]):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    dev_type, dev_id = device.split(":")
    dev_id = int(dev_id)
    gpu_model = None
    npu_model = None

    if dev_type == 'npu' and npu_model is None:
        npu_model = init_npu_model("./assets/mtnn/yolov8m_quantized.nb", dev_id)

    if dev_type == 'musa' and gpu_model is None:
        gpu_model = init_gpu_model()

    while not stop_event.is_set():
        empty = True

        for s_idx, w_stream in w_streams.items():

            try:
                (frame, input_tensors, npu_input_tensors) = w_stream["inference_queue"].get(block=False)
                empty = False
            except queue.Empty:
                continue

            # assert numpy frame and input tensors are on shared memory
            assert_ndarray_on_shm(frame)
            assert input_tensors[0].is_shared()
            assert npu_input_tensors[0].is_shared()

            gpu_results = None
            npu_results = None
            if dev_type == 'musa':

                start_time = time.time()
                # results = gpu_model.predict(frame, **gpu_model_cfgs, phase_input=input_tensors, phase='inference')
                # preds_tensor: torch.Tensor = results[0][0]
                # input_tensor: torch.Tensor = results[1]
                input_tensor: torch.Tensor = input_tensors[0].to(device)
                results = gpu_model.predictor.inference(input_tensors[0].to(device))
                torch.cuda.synchronize()
                if VERBOSE:
                    print(f":: model gpu inference took {((time.time() - start_time)*1000):.2f}ms")
                preds_tensor: torch.Tensor = results[0]

                assert preds_tensor.device.type == 'musa'
                assert input_tensor.device.type == 'musa'
                gpu_results = [preds_tensor.cpu().share_memory_(), input_tensors[0]]

                streams[s_idx].stats["model_fps"] = 1 / (time.time() - start_time)
                device_stats[f"{device}:model_fps"] = streams[s_idx].stats["model_fps"]
                # print(">>>>> GPU predict results and input:")
                # print_nested_types(gpu_results)

            elif dev_type == 'npu':

                npu_start = time.time()
                npu_outs: List[np.ndarray] = npu_model(npu_input_tensors[0].numpy())
                npu_preds_tensor = torch.from_numpy(npu_outs[0])
                npu_results = [npu_preds_tensor.share_memory_(), npu_input_tensors[0]]

                streams[s_idx].stats["npu_model_fps"] = 1 / (time.time() - npu_start)
                device_stats[f"{device}:model_fps"] = streams[s_idx].stats["npu_model_fps"]

                # print(">>>>> NPU predict results and input:")
                # print_nested_types(npu_results)
                # print(">>>>> NPU predict outs:")
                # print_nested_types(npu_outs)
            else:
                print(f"ERROR: can't decide which device to run for stream[{s_idx}]: device = {device}, stream_ids = {s_idx}")
                os._exit(-1)

            data = (frame, gpu_results, npu_results)
            assert_ndarray_on_shm(frame)
            assert gpu_results[0].is_shared()
            assert gpu_results[1].is_shared()
            assert npu_results[0].is_shared()
            assert npu_results[1].is_shared()
            put_to_queue(f"model_output_queue[{s_idx}]", w_stream["model_output_queue"], data)

        if empty:
            sleep_ms(10)


def yolo_postprocess_worker(stop_event, w_streams: Dict[int, Dict]):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    model = init_cpu_model()
    processed = 0
    delta_thres = 1000 / 1000 # 80ms

    while not stop_event.is_set():
        empty = True

        total_fps = 0
        total_model_fps = sum(value for key, value in device_stats.items() if key.endswith(":model_fps"))

        for s_idx, w_stream in w_streams.items():

            start_time = time.time()
            total_fps += streams[s_idx].stats["fps"]

            try:
                (frame, gpu_results, npu_results) = w_stream["model_output_queue"].get(block=False)
                empty = False
            except queue.Empty:
                continue

            device = 'GPU' if gpu_results is not None else 'NPU'
            pred_results = gpu_results or npu_results

            assert pred_results is not None
            # assert numpy frame and input tensors are on shared memory
            assert_ndarray_on_shm(frame)
            assert pred_results[0].is_shared()
            assert pred_results[1].is_shared()

            _t = time.time()
            # results = model.predict(frame, **gpu_model_cfgs, phase_input=pred_results, phase='postprocess')
            results = model.predictor.postprocess(pred_results[0], pred_results[1], [frame])
            if VERBOSE:
                print(f":: model postprocess took {((time.time() - _t)*1000):.2f}ms")

            annotated_frame = results[0].plot()

            # fps_text = f"{device} fps: {stream.fps:.2f} total_model_fps: {total_model_fps:.2f}"
            fps_text = f"{device} gpu_fps: {streams[s_idx].stats['model_fps']:.2f} npu_fps: {streams[s_idx].stats['npu_model_fps']:.2f} total_model_fps: {total_model_fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Encode to JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue

            out_frame = torch.from_numpy(jpeg).share_memory_().numpy()
            # out_frame = jpeg.tobytes()
            assert_ndarray_on_shm(out_frame)
            put_to_queue(f"output_queue[{s_idx}]", w_stream["output_queue"], out_frame)

            # calculate fps ############################################################
            processed += 1
            if processed > 10:
                delta = 0
                cur_time = time.time()
                if streams[s_idx].stats["last_time"] is not None:
                    delta = cur_time - streams[s_idx].stats["last_time"]
                streams[s_idx].stats["last_time"] = cur_time
                streams[s_idx].stats["post_fps"] = 1 / (cur_time - start_time)
                # skip frames that comes in late due to bad network condition .etc
                if delta > 0 and delta < delta_thres:
                    # stream.total_time += delta
                    # stream.processed_frames += 1
                    # stream.fps = stream.processed_frames / stream.total_time
                    streams[s_idx].stats["fps"] = 1 / delta
                if VERBOSE:
                    print(f"<- stream[{s_idx}] [{device}] outcoming fps = {streams[s_idx].stats['fps']:.2f}, prep_fps = {streams[s_idx].stats['prep_fps']:.2f}, gpu_model_fps = {streams[s_idx].stats['model_fps']:.2f}, npu_model_fps = {streams[s_idx].stats['npu_model_fps']:.2f}, post_fps = {streams[s_idx].stats['post_fps']:.2f}")

                    if s_idx == (len(w_streams)-1):
                        print(f"total_fps = {total_fps:.2f}, total_model_fps = {total_model_fps:.2f}")
            ############################################################################

        if empty:
            sleep_ms(10)


def yolo_pipelined_stream(stream: QueuedStream):
    while not stop_event.is_set() and not stream.stop_event.is_set() and stream.stream_started:
        try:
            out_frame = stream.output_queue.get(block=True)
        except queue.Empty:
            sleep_ms(10)
            continue

        if isinstance(out_frame, np.ndarray):
            out_frame = out_frame.tobytes()

        assert isinstance(out_frame, bytes)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + out_frame + b"\r\n")


def yolo_synced_stream():
    model = init_gpu_model()

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

        phase_input = None
        pred_start  = time.time()
        pre_out     = model.predict(frame, **kwargs, phase_input=phase_input, phase='preprocess')
        pred_end    = time.time()
        pre_time    = pred_end - pred_start

        pred_start  = time.time()
        infer_out   = model.predict(frame, **kwargs, phase_input=pre_out, phase='inference')
        pred_end    = time.time()
        pred_time   = pred_end - pred_start

        pred_start  = time.time()
        results     = model.predict(frame, **kwargs, phase_input=infer_out, phase='postprocess')
        pred_end    = time.time()
        post_time   = pred_end - pred_start

        model_fps = 1 / pred_time

        plot_time = time.time()
        # Draw results on frame
        annotated_frame = results[0].plot()

        cur_time = time.time()
        delta_time = 0
        if prev_time is not None:
            delta_time = cur_time - prev_time
            fps = 1 / delta_time
        prev_time = cur_time

        if VERBOSE:
            print(f"fps = {fps:.2f}, time = {(delta_time * 1000):.2f}ms, model_fps = {model_fps:.2f}, model time = {(pred_time*1000):.2f}ms, pre_time = {pre_time*1000:.2f}, post_time = {post_time*1000:.2f}")

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        if VERBOSE:
            print(f"plots to jpeg time: {((time.time()-plot_time)*1000):.2f}ms")

        # Yield as MJPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        pre_endtime = time.time()


class YoloLiveMode(str, Enum):
    sync      = "sync"
    pipelined = "pipelined"


def init_app():

    app = FastAPI()

    @app.get("/video/sync/{index}")
    def yolo_sync(index: int = 0):
        return StreamingResponse(yolo_synced_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.get("/video/pipelined/{index}")
    def yolo_pipelined(index: int = 0):
        print(f"INFO: stream[{index}] starting ...")
        stream = streams[index]
        stream.start()
        print(f"INFO: stream[{index}] started")
        res = StreamingResponse(yolo_pipelined_stream(stream), media_type='multipart/x-mixed-replace; boundary=frame')

        # def on_close():
        #     print(f"INFO: stream[{index}] stopping ...")
        #     stream.stop()
        #     print(f"INFO: stream[{index}] stopped")
        # res.background = BackgroundTask(on_close)

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
                        img0.parentElement.append(img)
                    }, 3000 * i);
                }
            }
        </script>
        </html>
        """


    @app.on_event("startup")
    def start_workers():
        """preprocess workers"""
        w_streams = {
            stream.idx: {
                "input_frames_queue": stream.input_frames_queue,
                "inference_queue": stream.inference_queue,
            } \
            for _key, stream in streams.items()
        }
        Proc(target=yolo_preprocess_worker,  name="yolo_preprocess_worker", daemon=True, \
             kwargs={"stop_event": stop_event, "w_streams": w_streams}).start()

        """gpu inference workers"""
        w_streams = {
            key: {
                "inference_queue": stream.inference_queue,
                "model_output_queue": stream.model_output_queue
            } \
            for key, stream in streams.items() if key in [0]
        }
        Proc(target=yolo_inference_worker,   name="yolo_inference_worker:musa:0", daemon=True, \
             kwargs={"stop_event": stop_event, "device": "musa:0", "w_streams": w_streams}).start()

        """npu inference workers"""
        w_streams = {
            key: {
                "inference_queue": stream.inference_queue,
                "model_output_queue": stream.model_output_queue
            } \
            for key, stream in streams.items() if key in [0]
        }
        Proc(target=yolo_inference_worker,   name="yolo_inference_worker:npu:0", daemon=True, \
             kwargs={"stop_event": stop_event, "device": "npu:0",  "w_streams": w_streams}).start()
        Proc(target=yolo_inference_worker,   name="yolo_inference_worker:npu:1", daemon=True, \
             kwargs={"stop_event": stop_event, "device": "npu:1",  "w_streams": w_streams}).start()

        """postprocess workers"""
        w_streams = {
            key: {
                "model_output_queue": stream.model_output_queue,
                "output_queue": stream.output_queue,
            } \
            for key, stream in streams.items()
        }
        Proc(target=yolo_postprocess_worker, name="yolo_postprocess_worker", daemon=True, \
             kwargs={"stop_event": stop_event, "w_streams": w_streams}).start()


    @app.on_event("shutdown")
    def shutdown():
        stop_event.set()
        exit(0)

    return app


if __name__ == "__main__":

    if args.live_on_local:
        live_yolo_detect_stream(loop=True, pos=[5, 5])
    else:
        uvicorn.run(init_app(), host="0.0.0.0", port=args.port, reload=False)

