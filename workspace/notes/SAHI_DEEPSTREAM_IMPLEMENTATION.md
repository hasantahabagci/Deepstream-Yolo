# SAHI in DeepStream (YOLO11) - Implementation Notes

## What was missing
- `config_preprocess_sahi.txt` was using unsupported keys (`stride-x`, `roi-width`, etc.).
- `deepstream_app_sahi.cfg` used invalid section naming (`[preprocess0]`).
- No SAHI-style cross-tile merge stage existed, so overlapping tiles could produce duplicates.

## What is now implemented
- Valid `nvdspreprocess` tiling config: [config_preprocess_sahi.txt](/home/ituarc/DeepStream-Yolo/config_preprocess_sahi.txt)
- SAHI-ready PGIE config: [config_infer_primary_yolo11_sahi.txt](/home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi.txt)
- Working Python pipeline with merge stage:
  [deepstream_sahi_app.py](/home/ituarc/DeepStream-Yolo/deepstream_sahi_app.py)
- Auto config generator for any resolution:
  [generate_sahi_preprocess_config.py](/home/ituarc/DeepStream-Yolo/generate_sahi_preprocess_config.py)

## Run
```bash
cd /home/ituarc/DeepStream-Yolo
python3 deepstream_sahi_app.py \
  --source /path/to/video.mp4 \
  --preprocess-config /home/ituarc/DeepStream-Yolo/config_preprocess_sahi.txt \
  --infer-config /home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi.txt
```

For RTSP:
```bash
python3 deepstream_sahi_app.py --source rtsp://user:pass@host:port/path
```

## Speed-first run (recommended when FPS is low)
This keeps the same 1024 model, but reduces ROI count from 7 to 4:
- overlap 0.10 instead of 0.20
- no full-frame extra ROI
- optional: disable merge for max FPS

```bash
python3 /home/ituarc/DeepStream-Yolo/deepstream_sahi_app.py \
  --source /home/ituarc/DeepStream-Yolo/balloon_test.mp4 \
  --preprocess-config /home/ituarc/DeepStream-Yolo/config_preprocess_sahi_fast.txt \
  --infer-config /home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi.txt \
  --no-display \
  --disable-merge
```

Optional high-throughput infer config (FP16 + batch4):
`/home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi_fast.txt`

## Real-time mode (recommended for live camera)
Full SAHI on every frame is usually not real-time on Orin NX. Use tracker-assisted sparse detection:
- tracker enabled
- detect every 3rd frame (`interval=2`)
- leaky queues to avoid latency growth
- merge disabled in preset

```bash
python3 /home/ituarc/DeepStream-Yolo/deepstream_sahi_app.py \
  --source rtsp://user:pass@ip:port/path \
  --preprocess-config /home/ituarc/DeepStream-Yolo/config_preprocess_sahi_fast.txt \
  --infer-config /home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi.txt \
  --realtime-preset \
  --no-display
```

For file testing:
```bash
python3 /home/ituarc/DeepStream-Yolo/deepstream_sahi_app.py \
  --source /home/ituarc/DeepStream-Yolo/balloon_test.mp4 \
  --preprocess-config /home/ituarc/DeepStream-Yolo/config_preprocess_sahi_fast.txt \
  --infer-config /home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11_sahi.txt \
  --realtime-preset \
  --no-display
```

## Generate preprocess config for your resolution
Example: 4032x2268 input, 640x640 slices, 20% overlap, include full-frame ROI.
```bash
python3 /home/ituarc/DeepStream-Yolo/generate_sahi_preprocess_config.py \
  --source-width 4032 \
  --source-height 2268 \
  --slice-width 640 \
  --slice-height 640 \
  --overlap-width-ratio 0.2 \
  --overlap-height-ratio 0.2 \
  --tensor-name input \
  --include-full-frame \
  --output /home/ituarc/DeepStream-Yolo/config_preprocess_sahi.txt
```

## FPS and quality rules
- SAHI compute cost scales with ROI count. More tiles means lower FPS.
- To keep quality high:
  - Keep overlap around `0.15-0.25`.
  - Keep merge metric as `IOS` and threshold around `0.5`.
  - Keep merge enabled (default in `deepstream_sahi_app.py`).
  - Include full-frame ROI (`--include-full-frame`) to recover large objects.
- To keep FPS high:
  - Set nvinfer `batch-size = ROI count` and rebuild engine with dynamic batch support.
  - Use FP16/INT8 if acceptable.
  - Reduce slice count (larger slices or smaller overlap) if needed.

## Important model note
- `tensor-name` in preprocess config must match the ONNX input layer name exactly.
- Current `Balloon_V11N_Last.pt.onnx` input is `input`.
