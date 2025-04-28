
### Start services
```bash
python live_stream_server.py
```

```bash
python live_yolo_detect.py
```

### convert video to 30fps

```bash
ffmpeg -i input.mp4 -filter:v fps=30 output_30fps.mp4
```
