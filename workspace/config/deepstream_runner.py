#!/usr/bin/env python3
"""
DeepStream Runner Module for Visual Guidance System.

This module encapsulates the DeepStream detection and tracking pipeline,
providing a clean Python interface for the rest of the system.

Based on:
- utils/deepstream/python_test.py (existing implementation)
- NVIDIA DeepStream Python examples (deepstream-test1, deepstream-test2, etc.)
  Reference: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

Key features:
- Pipeline creation and lifecycle management
- Detection/tracking metadata extraction via probes
- Callbacks for real-time processing
- Thread-safe access to detection results
- Support for past tracking metadata (NvDCF tracker)
"""

import sys
import os
import time
import threading
import configparser
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds
from project_paths import ONNX_ENGINE_CACHE_FILE, ROOT_PGIE_CONFIG_YOLO11

# Muxer batch timeout is now calculated dynamically based on FPS
# For 59 FPS: 1_000_000 / 59 * 0.8 = ~13559 μs
# For 30 FPS: 1_000_000 / 30 * 0.8 = ~26667 μs


# =============================================================================
# Platform Detection (from NVIDIA official examples)
# =============================================================================
class PlatformInfo:
    """Platform detection utility for Jetson vs x86."""
    
    def __init__(self):
        self._is_aarch64 = os.uname().machine == 'aarch64'
    
    def is_integrated_gpu(self) -> bool:
        """Check if running on integrated GPU (Jetson)."""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return True
        except FileNotFoundError:
            return False
    
    def is_platform_aarch64(self) -> bool:
        """Check if platform is aarch64 (ARM64)."""
        return self._is_aarch64
    
    def is_jetson(self) -> bool:
        """Check if running on Jetson platform."""
        return self.is_integrated_gpu()


# =============================================================================
# Bus Call Handler (from NVIDIA official examples)
# =============================================================================
def bus_call(bus, message, loop):
    """
    GStreamer bus message handler.
    
    Based on: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/common/bus_call.py
    """
    t = message.type
    if t == Gst.MessageType.EOS:
        print("DeepStream: End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"DeepStream Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"DeepStream Error: {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct is not None:
            if struct.get_name() == 'GstBinForwarded':
                forward_msg = struct.get_value('message')
                if forward_msg and forward_msg.type == Gst.MessageType.EOS:
                    print("DeepStream: EOS from element")
    return True


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class DetectionResult:
    """Container for a single detection."""
    object_id: int  # Tracking ID
    class_id: int
    confidence: float
    left: float
    top: float
    width: float
    height: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (self.left + self.width / 2, self.top + self.height / 2)
    
    @property
    def center_x(self) -> float:
        """Get center X coordinate."""
        return self.left + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Get center Y coordinate."""
        return self.top + self.height / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'object_id': self.object_id,
            'class_id': self.class_id,
            'confidence': self.confidence,
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }


@dataclass
class TrackingHistory:
    """Container for past tracking data (from NvDCF tracker)."""
    stream_id: int = 0
    unique_id: int = 0
    class_id: int = 0
    obj_label: str = ""
    frames: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FrameDetections:
    """Container for all detections in a frame."""
    frame_number: int
    timestamp: float
    detections: List[DetectionResult] = field(default_factory=list)
    best_detection: Optional[DetectionResult] = None
    tracking_history: List[TrackingHistory] = field(default_factory=list)
    fps: float = 0.0
    num_objects: int = 0


@dataclass
class DeepStreamConfig:
    """Configuration for DeepStream pipeline."""
    # Resolution
    width: int = 1920
    height: int = 1080
    fps: int = 35
    
    # Source config
    source_type: str = "argus"  # "argus", "v4l2", "file", "appsrc"
    source_path: str = ""  # For file or v4l2
    sensor_id: int = 0  # For argus camera
    
    # Inference config
    pgie_config_path: str = str(ROOT_PGIE_CONFIG_YOLO11)
    
    # Inference parameters (override values in pgie_config_path)
    onnx_file: str = ""
    model_engine_file: str = ""
    network_mode: int = -1  # -1 = use config file value, 0=FP32, 1=INT8, 2=FP16
    num_detected_classes: int = -1
    interval: int = -1
    pre_cluster_threshold: float = -1.0
    
    # Tracker config
    enable_tracker: bool = True
    tracker_config_file: str = ""  # Path to tracker config file (INI format)
    tracker_width: int = 1920
    tracker_height: int = 1120
    tracker_gpu_id: int = 0
    tracker_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    tracker_ll_config_path: str = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml"
    enable_past_frame_meta: bool = True  # Enable past tracking metadata
    
    # Display config
    enable_display: bool = False
    
    # Recording config
    # NOTE: `enable_recording` and `output_path` are kept for backward compatibility
    # and map to annotated recording.
    enable_recording: bool = True
    enable_annotated_recording: bool = True
    enable_raw_recording: bool = False
    annotated_output_path: str = ""
    output_path: str = ""
    raw_output_path: str = ""
    bitrate: int = 4000000

    # OSD / HUD
    enable_hud: bool = True
    enable_bbox: bool = True

    # Metadata for HUD
    camera_id: str = "cam0"
    model_name: str = ""
    model_version: str = ""
    
    # Flip method (0=none, 2=180 degrees)
    flip_method: int = 2
    
    # Camera settings (for nvarguscamerasrc)
    tnr_mode: int = 0  # Temporal noise reduction
    tnr_strength: int = 0
    ee_mode: int = 0  # Edge enhancement
    ee_strength: int = 0
    
    # Use new nvstreammux
    use_new_streammux: bool = False

    def __post_init__(self) -> None:
        # Backward compatibility mapping.
        if self.output_path and not self.annotated_output_path:
            self.annotated_output_path = self.output_path
        if self.annotated_output_path and not self.output_path:
            self.output_path = self.annotated_output_path
        if self.enable_recording and not self.enable_annotated_recording:
            self.enable_annotated_recording = True


def _check_and_clean_engine_cache(onnx_path: str, engine_path: str, base_dir: str) -> None:
    """
    Check if ONNX file has changed and delete old engine file if needed.
    
    Uses a cache file to track the last used ONNX file for each engine.
    If the ONNX file has changed since last run, the engine file is deleted
    so TensorRT will rebuild it with the new model.
    
    Logic:
    1. Load cache (stores: engine_path -> last_used_onnx_path)
    2. If cache has a different ONNX for this engine -> delete engine file
    3. Save current ONNX to cache
    
    Args:
        onnx_path: Current ONNX file path (absolute)
        engine_path: Engine file path to potentially delete (absolute)
        base_dir: Base directory for cache file
    """
    import json
    
    cache_file = str(ONNX_ENGINE_CACHE_FILE)
    
    print(f"  Checking engine cache...")
    print(f"    Current ONNX: {os.path.basename(onnx_path)}")
    print(f"    Engine file: {os.path.basename(engine_path)}")
    
    # Load existing cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            cache = {}
    
    # Get last used ONNX for this engine (use basename for comparison)
    last_onnx = cache.get("last_onnx", None)
    current_onnx_name = os.path.basename(onnx_path)
    
    if last_onnx is None:
        # First run, just save current ONNX
        print(f"    First run, caching ONNX: {current_onnx_name}")
    elif last_onnx != current_onnx_name:
        # ONNX changed! Delete engine file
        print(f"    ONNX changed: {last_onnx} -> {current_onnx_name}")
        if os.path.exists(engine_path):
            print(f"    Deleting old engine file: {engine_path}")
            try:
                os.remove(engine_path)
                print(f"    Engine file deleted successfully!")
            except OSError as e:
                print(f"    Warning: Could not delete engine file: {e}")
        else:
            print(f"    Engine file not found, nothing to delete")
    else:
        print(f"    ONNX unchanged, keeping engine file")
    
    # Always update cache with current ONNX (basename only for cleaner comparison)
    cache["last_onnx"] = current_onnx_name
    cache["last_engine"] = os.path.basename(engine_path)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"    Cache updated: {cache_file}")
    except IOError as e:
        print(f"    Warning: Could not save cache file: {e}")


def create_modified_pgie_config(base_config_path: str, config: DeepStreamConfig) -> str:
    """
    Create a modified PGIE config file with overridden parameters.
    
    Reads the base config line by line, replaces only specified parameters,
    and writes to a temporary file. This preserves all original settings
    including custom-lib-path and other DeepStream-specific options.
    
    Relative paths in the original config are converted to absolute paths
    based on the original config file's directory.
    
    If the ONNX file has changed since last run, the engine file is deleted
    so TensorRT will rebuild it with the new model.
    
    Args:
        base_config_path: Path to the original config file
        config: DeepStreamConfig with override values
        
    Returns:
        Path to the modified config file
    """
    import tempfile
    
    # Get the directory of the base config for relative paths
    base_dir = os.path.dirname(base_config_path)
    
    # Build override map
    overrides = {}
    onnx_path = None
    engine_path = None
    
    if config.onnx_file:
        onnx_path = config.onnx_file if os.path.isabs(config.onnx_file) else os.path.join(base_dir, config.onnx_file)
        overrides['onnx-file'] = onnx_path
        print(f"  Override onnx-file: {onnx_path}")
        
    if config.model_engine_file:
        engine_path = config.model_engine_file if os.path.isabs(config.model_engine_file) else os.path.join(base_dir, config.model_engine_file)
        overrides['model-engine-file'] = engine_path
        print(f"  Override model-engine-file: {engine_path}")

    if config.network_mode >= 0:
        overrides['network-mode'] = str(config.network_mode)
        print(f"  Override network-mode: {config.network_mode}")
        
    if config.num_detected_classes >= 0:
        overrides['num-detected-classes'] = str(config.num_detected_classes)
        print(f"  Override num-detected-classes: {config.num_detected_classes}")
        
    if config.interval >= 0:
        overrides['interval'] = str(config.interval)
        print(f"  Override interval: {config.interval}")
        
    if config.pre_cluster_threshold >= 0:
        overrides['pre-cluster-threshold'] = str(config.pre_cluster_threshold)
        print(f"  Override pre-cluster-threshold: {config.pre_cluster_threshold}")
    
    # Keys that contain file paths and need to be converted to absolute paths
    path_keys = {'custom-lib-path', 'labelfile-path', 'int8-calib-file', 
                 'onnx-file', 'model-engine-file', 'tlt-encoded-model', 
                 'tlt-model-key', 'model-file', 'proto-file', 'mean-file'}
    
    # Read original file and replace only specified lines
    modified_lines = []
    with open(base_config_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            
            # Skip empty lines and comments - keep as-is
            if not stripped or stripped.startswith('#') or stripped.startswith('['):
                modified_lines.append(line)
                continue
            
            # Check if this line has a key=value format
            if '=' in stripped:
                key = stripped.split('=')[0].strip()
                value = stripped.split('=', 1)[1].strip()
                
                # Check if this is a key we want to override
                if key in overrides:
                    modified_lines.append(f"{key}={overrides[key]}\n")
                    continue
                
                # Convert relative paths to absolute paths for path-containing keys
                if key in path_keys and value and not value.startswith('#'):
                    if not os.path.isabs(value):
                        abs_path = os.path.join(base_dir, value)
                        modified_lines.append(f"{key}={abs_path}\n")
                        continue
            
            # Keep original line
            modified_lines.append(line)
    
    # Write to a unique temp file to avoid collisions.
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="pgie_config_",
        suffix=".txt",
        delete=False,
    ) as tf:
        tf.writelines(modified_lines)
        temp_config_path = tf.name
    
    print(f"  Modified config written to: {temp_config_path}")
    return temp_config_path


class DeepStreamRunner:
    """
    DeepStream pipeline runner for detection and tracking.
    
    This class encapsulates the GStreamer/DeepStream pipeline and provides:
    - Pipeline creation and lifecycle management
    - Detection/tracking metadata extraction
    - Callbacks for real-time processing
    - Thread-safe access to detection results
    - Support for past tracking metadata
    
    Based on official NVIDIA DeepStream Python examples:
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
    """

    def __init__(self, config: Optional[DeepStreamConfig] = None):
        """
        Initialize DeepStream runner.
        
        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        # Keep original working directory for resolving relative output paths.
        self._original_cwd = os.getcwd()
        
        self.config = config or DeepStreamConfig()
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Platform info
        self._platform = PlatformInfo()
        
        # Detection results
        self._latest_detections: Optional[FrameDetections] = None
        self._detections_lock = threading.Lock()
        self._detection_callbacks: List[Callable[[FrameDetections], None]] = []

        # HUD telemetry (from guidance/control loop)
        self._hud_state_lock = threading.Lock()
        self._hud_state: Dict[str, Any] = {}
        
        # Frame callbacks (for raw frame access)
        self._frame_callbacks: List[Callable] = []
        self._enable_frame_extraction = False
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # FPS tracking
        self._frame_count = 0
        self._start_time = 0.0
        self._fps_interval = 30

        # Keep references to key pipeline elements for dynamic linking
        self._streammux: Optional[Gst.Element] = None
        self._current_fps = 0.0
        self._appsrc_push_count = 0

        # Bus diagnostics for HUD / debugging
        self._last_bus_warning: Optional[str] = None
        self._last_bus_error: Optional[str] = None
        self._last_bus_message_ts: float = 0.0
        
        # Initialize GStreamer
        Gst.init(None)

    def update_hud_state(self, state: Dict[str, Any]) -> None:
        """Update HUD telemetry (thread-safe). Values are rendered only on annotated output."""
        if not isinstance(state, dict):
            return
        with self._hud_state_lock:
            # Shallow merge is fine; values should be JSON-like primitives/lists.
            self._hud_state.update(state)
        
    def register_detection_callback(self, callback: Callable[[FrameDetections], None]):
        """
        Register a callback to be called when new detections are available.
        
        Args:
            callback: Function that takes FrameDetections as argument.
        """
        self._detection_callbacks.append(callback)
        
    def get_latest_detections(self) -> Optional[FrameDetections]:
        """
        Get the most recent detection results (thread-safe).
        
        Returns:
            FrameDetections or None if no detections yet.
        """
        with self._detections_lock:
            return self._latest_detections

    def get_best_bbox(self) -> Optional[Dict[str, Any]]:
        """
        Get the best detection (highest confidence) as a dict.
        
        Returns:
            Detection dict with keys: left, top, width, height, confidence
            or None if no detection.
        """
        with self._detections_lock:
            if self._latest_detections and self._latest_detections.best_detection:
                return self._latest_detections.best_detection.to_dict()
        return None

    def register_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        Register a callback to be called with raw frames from the pipeline.
        
        Note: This will incur a GPU->CPU copy overhead for each frame.
        Only register if you need raw frame access.
        
        Args:
            callback: Function(frame: np.ndarray, frame_number: int)
                     frame is BGR format, same resolution as pipeline.
        """
        self._frame_callbacks.append(callback)
        self._enable_frame_extraction = True
        print(f"Frame callback registered. Frame extraction enabled.")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame (thread-safe).
        
        Note: This is only populated if frame callbacks are registered.
        
        Returns:
            numpy array (BGR, HxWxC) or None if no frame available.
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _calculate_fps(self) -> Optional[float]:
        """Calculate current FPS."""
        self._frame_count += 1
        if self._frame_count % self._fps_interval == 0:
            current_time = time.time()
            elapsed = current_time - self._start_time
            self._current_fps = self._fps_interval / elapsed if elapsed > 0 else 0
            self._start_time = current_time
            return self._current_fps
        return None

    def _extract_past_tracking_meta(self, batch_meta) -> List[TrackingHistory]:
        """
        Extract past tracking metadata from NvDCF tracker.
        
        Based on deepstream-test2 example from NVIDIA.
        """
        tracking_history = []
        
        if not self.config.enable_past_frame_meta:
            return tracking_history
        
        l_user = batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            
            if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
                try:
                    past_data_batch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                
                for misc_data_stream in pyds.NvDsTargetMiscDataBatch.list(past_data_batch):
                    for misc_data_obj in pyds.NvDsTargetMiscDataStream.list(misc_data_stream):
                        history = TrackingHistory(
                            stream_id=misc_data_stream.streamID,
                            unique_id=misc_data_obj.uniqueId,
                            class_id=misc_data_obj.classId,
                            obj_label=misc_data_obj.objLabel
                        )
                        
                        for misc_data_frame in pyds.NvDsTargetMiscDataObject.list(misc_data_obj):
                            history.frames.append({
                                'frame_num': misc_data_frame.frameNum,
                                'left': misc_data_frame.tBbox.left,
                                'top': misc_data_frame.tBbox.top,
                                'width': misc_data_frame.tBbox.width,
                                'height': misc_data_frame.tBbox.height,
                                'confidence': misc_data_frame.confidence,
                                'age': misc_data_frame.age
                            })
                        
                        tracking_history.append(history)
            
            try:
                l_user = l_user.next
            except StopIteration:
                break
        
        return tracking_history

    def _osd_probe_callback(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """
        Probe callback to extract detection/tracking metadata from the pipeline.

        Note: This probe should be placed on a pad that is always present
        (pre-OSD). It must NOT draw overlays, so the raw-recording branch
        remains overlay-free.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer")
            return Gst.PadProbeReturn.OK
        
        current_fps = self._calculate_fps()
        
        # Retrieve batch metadata
        # Note: pyds.gst_buffer_get_nvds_batch_meta() expects the C address
        # of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                # Cast l_frame.data to pyds.NvDsFrameMeta
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            detections: List[DetectionResult] = []
            best_detection: Optional[DetectionResult] = None
            max_conf = 0.0
            
            # Extract all detections
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Cast l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Disable bbox/labels at metadata level (annotated branch only).
                # This keeps the raw branch overlay-free while still allowing HUD rendering.
                if not self.config.enable_bbox:
                    try:
                        obj_meta.rect_params.border_width = 0
                        obj_meta.rect_params.has_bg_color = 0
                    except Exception:
                        pass
                    try:
                        obj_meta.text_params.display_text = ""
                        obj_meta.text_params.set_bg_clr = 0
                        obj_meta.text_params.font_params.font_size = 0
                    except Exception:
                        pass
                
                detection = DetectionResult(
                    object_id=obj_meta.object_id,
                    class_id=obj_meta.class_id,
                    confidence=obj_meta.confidence,
                    left=obj_meta.rect_params.left,
                    top=obj_meta.rect_params.top,
                    width=obj_meta.rect_params.width,
                    height=obj_meta.rect_params.height
                )
                detections.append(detection)
                
                # Track best detection (highest confidence)
                if detection.confidence > max_conf:
                    max_conf = detection.confidence
                    best_detection = detection
                
                # Optional: Set border color for visualization
                # obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 0.8)
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Extract past tracking metadata
            tracking_history = self._extract_past_tracking_meta(batch_meta)
            
            # Create frame detections container
            frame_detections = FrameDetections(
                frame_number=frame_number,
                timestamp=time.time(),
                detections=detections,
                best_detection=best_detection,
                tracking_history=tracking_history,
                fps=current_fps or self._current_fps,
                num_objects=num_rects
            )
            
            # Update latest detections (thread-safe)
            with self._detections_lock:
                self._latest_detections = frame_detections
            
            # Call registered callbacks
            for callback in self._detection_callbacks:
                try:
                    callback(frame_detections)
                except Exception as e:
                    print(f"Detection callback error: {e}")
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK

    def _hud_probe_callback(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """
        Probe callback that renders a tactical HUD overlay.

        This must be attached ONLY on the annotated branch (nvdsosd sink pad),
        so the raw recording stays 100% overlay-free.
        """
        if not self.config.enable_hud:
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        cfg = self.config
        width, height = int(cfg.width), int(cfg.height)

        # Resolve pipeline state (non-blocking).
        state_str = "UNKNOWN"
        if self.pipeline is not None:
            try:
                _, state, _ = self.pipeline.get_state(0)
                if state == Gst.State.PLAYING:
                    state_str = "PLAYING"
                elif state == Gst.State.PAUSED:
                    state_str = "PAUSED"
                elif state == Gst.State.READY:
                    state_str = "READY"
                elif state == Gst.State.NULL:
                    state_str = "NULL"
            except Exception:
                state_str = "UNKNOWN"

        # Snapshot latest stats (thread-safe).
        latest = self.get_latest_detections()
        det_count = int(latest.num_objects) if latest else 0
        best = latest.best_detection if latest else None
        best_id = int(best.object_id) if best else None
        best_cls = int(best.class_id) if best else None
        best_conf = float(best.confidence) if best else None

        # Snapshot external HUD telemetry (thread-safe).
        with self._hud_state_lock:
            ext = dict(self._hud_state)

        expected_fps = float(cfg.fps) if cfg.fps else 0.0
        fps = float(self._current_fps)

        # Alerts (time-limited).
        alert = ""
        now = time.time()
        if self._last_bus_error and (now - self._last_bus_message_ts) < 5.0:
            alert = f"INFERENCE ERROR: {self._last_bus_error}"
        elif self._last_bus_warning and (now - self._last_bus_message_ts) < 5.0:
            alert = f"WARNING: {self._last_bus_warning}"
        elif expected_fps > 0 and fps > 0 and fps < expected_fps * 0.7:
            alert = "LOW FPS"

        # Recording state indicators.
        raw_on = bool(cfg.enable_raw_recording and cfg.raw_output_path)
        ann_on = bool(cfg.enable_annotated_recording and (cfg.annotated_output_path or cfg.output_path))

        # Model info.
        model = cfg.model_name.strip() if cfg.model_name else ""
        if not model:
            candidate = cfg.onnx_file or cfg.model_engine_file or cfg.pgie_config_path
            model = os.path.basename(candidate) if candidate else "unknown"

        thr = cfg.pre_cluster_threshold if cfg.pre_cluster_threshold >= 0 else None

        # Tactical HUD layout (military-style, edge-only).
        scale = max(0.75, min(1.5, height / 1080.0))
        margin = max(10, int(18 * scale))
        pad_px = max(8, int(12 * scale))
        font_name = "Monospace"
        font_small = max(12, int(13 * scale))
        font_big = max(13, int(15 * scale))
        line_h = max(16, int(font_small * 1.35))

        panel_w = max(int(width * 0.30), 380)
        panel_lines = 4
        panel_h = pad_px * 2 + line_h * panel_lines

        x_l = margin
        x_r = max(margin, width - margin - panel_w)
        y_t = margin
        y_b = max(margin, height - margin - panel_h)

        # Per-frame meta loop (batch-size is 1, but keep generic).
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_rects = 4
            display_meta.num_labels = 0
            display_meta.num_lines = 0

            # Panel style (tactical glass + soft green borders).
            bg_rgba = (0.02, 0.05, 0.03, 0.28)
            border_rgba = (0.35, 1.00, 0.55, 0.22)

            panels = [
                (x_l, y_t),  # top-left
                (x_r, y_t),  # top-right
                (x_l, y_b),  # bottom-left
                (x_r, y_b),  # bottom-right
            ]

            for i, (px, py) in enumerate(panels):
                rp = display_meta.rect_params[i]
                rp.left = float(px)
                rp.top = float(py)
                rp.width = float(panel_w)
                rp.height = float(panel_h)
                rp.border_width = 1
                rp.border_color.set(*border_rgba)
                rp.has_bg_color = 1
                rp.bg_color.set(*bg_rgba)

            def _add_line(x1: int, y1: int, x2: int, y2: int, rgba: tuple[float, float, float, float], w: int = 1):
                idx = display_meta.num_lines
                try:
                    lp = display_meta.line_params[idx]
                except Exception:
                    return
                lp.x1 = int(x1)
                lp.y1 = int(y1)
                lp.x2 = int(x2)
                lp.y2 = int(y2)
                lp.line_width = int(w)
                lp.line_color.set(*rgba)
                display_meta.num_lines = idx + 1

            def _add_text(text: str, x: int, y: int, *, size: int, rgba: tuple[float, float, float, float]):
                idx = display_meta.num_labels
                try:
                    tp = display_meta.text_params[idx]
                except Exception:
                    return
                tp.display_text = text
                tp.x_offset = int(x)
                tp.y_offset = int(y)
                tp.font_params.font_name = font_name
                tp.font_params.font_size = int(size)
                tp.font_params.font_color.set(*rgba)
                tp.set_bg_clr = 0
                display_meta.num_labels = idx + 1

            # Colors.
            white = (1.0, 1.0, 1.0, 1.0)
            dim = (0.78, 0.92, 0.82, 0.92)
            cyan = (0.25, 0.92, 1.0, 0.95)
            green = (0.35, 1.0, 0.55, 0.98)
            amber = (1.0, 0.72, 0.25, 1.0)
            red = (1.0, 0.25, 0.25, 1.0)

            # External telemetry (from guidance loop) - based on tests/manual/python_test.py fields.
            run_id = str(ext.get("run_id", ""))[:24]
            stage = str(ext.get("stage", "TRACK"))[:16]
            drone_mode = str(ext.get("drone_mode", "N/A"))[:12]
            armed = bool(ext.get("armed", False))
            linked = bool(ext.get("connected", False))
            distance_m = ext.get("distance_m", None)
            speed_mps = ext.get("speed_mps", None)
            rpy_deg = ext.get("attitude_deg", None)  # [roll,pitch,yaw]
            body_rates = ext.get("rates_rps", None)  # [roll_rate,pitch_rate,yaw_rate]
            throttle = ext.get("throttle", None)
            kp_yaw_val = ext.get("kp_yaw", None)
            roll_gain = ext.get("filter_roll", None)
            cbf_val = ext.get("cbf", None)
            pix_err = ext.get("pixel_error_px", None)  # [ex,ey]
            virt_pix = ext.get("virt_pixel", None)     # [x,y]
            offset = ext.get("offset", None)           # [x,y,z]
            depth_virt = ext.get("depth_virt", None)
            interceptor = ext.get("interceptor_lla", None)  # [lat,lon,alt]
            target = ext.get("target_lla", None)            # [lat,lon,alt]

            # Top-left: system + camera + timestamp (UTC)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
            link_str = ("LINK" if linked else "NO-LINK")
            arm_str = ("ARM" if armed else "SAFE")
            _add_text(f"[VG] CAM:{cfg.camera_id}  {link_str}/{arm_str}", x_l + pad_px, y_t + pad_px + 0 * line_h, size=font_big, rgba=green)
            _add_text(f"UTC: {ts}  RUN:{run_id}", x_l + pad_px, y_t + pad_px + 1 * line_h, size=font_small, rgba=dim)
            d_str = f"{distance_m:6.1f}m" if isinstance(distance_m, (int, float)) else "  n/a "
            v_str = f"{speed_mps:4.1f}m/s" if isinstance(speed_mps, (int, float)) else " n/a "
            _add_text(f"MODE:{drone_mode:<8}  STG:{stage:<8}  D:{d_str}  V:{v_str}", x_l + pad_px, y_t + pad_px + 2 * line_h, size=font_small, rgba=dim)
            _add_text(f"REC RAW:{'ON' if raw_on else 'OFF'}  ANN:{'ON' if ann_on else 'OFF'}", x_l + pad_px, y_t + pad_px + 3 * line_h, size=font_small, rgba=(green if (raw_on or ann_on) else dim))

            # Top-right: FPS + pipeline state + model/threshold + alerts
            fps_text = f"FPS:{fps:.1f}/{expected_fps:.0f}" if expected_fps else f"FPS:{fps:.1f}"
            _add_text(f"{fps_text}  PIPE:{state_str}  FRM:{int(frame_meta.frame_num)}", x_r + pad_px, y_t + pad_px + 0 * line_h, size=font_big, rgba=white)
            tgt_str = "NONE"
            if best is not None and best_conf is not None:
                tgt_str = f"ID:{best_id if best_id is not None else '-'} C:{best_cls if best_cls is not None else '-'} P:{best_conf:.2f}"
            _add_text(f"DET:{det_count:<3}  TGT:{tgt_str}  BBOX:{'ON' if cfg.enable_bbox else 'OFF'}", x_r + pad_px, y_t + pad_px + 1 * line_h, size=font_small, rgba=dim)
            thr_text = f"{thr:.2f}" if thr is not None else "n/a"
            _add_text(f"MODEL:{model[:18]:<18} THR:{thr_text}", x_r + pad_px, y_t + pad_px + 2 * line_h, size=font_small, rgba=dim)
            if alert:
                alert_color = red if "ERROR" in alert else amber
                _add_text(f"ALERT: {alert[:32]}", x_r + pad_px, y_t + pad_px + 3 * line_h, size=font_small, rgba=alert_color)
            else:
                _add_text("ALERT: NONE", x_r + pad_px, y_t + pad_px + 3 * line_h, size=font_small, rgba=dim)

            # Bottom-left: guidance/control telemetry (python_test.py)
            if isinstance(pix_err, (list, tuple)) and len(pix_err) >= 2:
                _add_text(f"PIX_ERR: X:{float(pix_err[0]):>7.1f}px  Y:{float(pix_err[1]):>7.1f}px", x_l + pad_px, y_b + pad_px + 0 * line_h, size=font_big, rgba=white)
            else:
                _add_text("PIX_ERR: n/a", x_l + pad_px, y_b + pad_px + 0 * line_h, size=font_big, rgba=white)

            if isinstance(virt_pix, (list, tuple)) and len(virt_pix) >= 2:
                _add_text(f"VIRT_PX: X:{float(virt_pix[0]):>7.1f}  Y:{float(virt_pix[1]):>7.1f}", x_l + pad_px, y_b + pad_px + 1 * line_h, size=font_small, rgba=dim)
            else:
                _add_text("VIRT_PX: n/a", x_l + pad_px, y_b + pad_px + 1 * line_h, size=font_small, rgba=dim)

            if isinstance(rpy_deg, (list, tuple)) and len(rpy_deg) >= 3:
                _add_text(f"ATT_DEG: R:{float(rpy_deg[0]):>6.1f} P:{float(rpy_deg[1]):>6.1f} Y:{float(rpy_deg[2]):>6.1f}", x_l + pad_px, y_b + pad_px + 2 * line_h, size=font_small, rgba=dim)
            else:
                _add_text("ATT_DEG: n/a", x_l + pad_px, y_b + pad_px + 2 * line_h, size=font_small, rgba=dim)

            thr_cmd = f"{throttle:.3f}" if isinstance(throttle, (int, float)) else "n/a"
            kp_cmd = f"{kp_yaw_val:.3f}" if isinstance(kp_yaw_val, (int, float)) else "n/a"
            fr_cmd = f"{roll_gain:.2f}" if isinstance(roll_gain, (int, float)) else "n/a"
            _add_text(f"THR:{thr_cmd}  KP_YAW:{kp_cmd}  RGAIN:{fr_cmd}", x_l + pad_px, y_b + pad_px + 3 * line_h, size=font_small, rgba=dim)

            # Bottom-right: nav + offsets + rates (python_test.py)
            if isinstance(interceptor, (list, tuple)) and len(interceptor) >= 3:
                _add_text(f"INT: {float(interceptor[0]):.6f},{float(interceptor[1]):.6f} ALT:{float(interceptor[2]):.1f}", x_r + pad_px, y_b + pad_px + 0 * line_h, size=font_small, rgba=dim)
            else:
                _add_text("INT: n/a", x_r + pad_px, y_b + pad_px + 0 * line_h, size=font_small, rgba=dim)

            if isinstance(target, (list, tuple)) and len(target) >= 3:
                _add_text(f"TGT: {float(target[0]):.6f},{float(target[1]):.6f} ALT:{float(target[2]):.1f}", x_r + pad_px, y_b + pad_px + 1 * line_h, size=font_small, rgba=dim)
            else:
                _add_text("TGT: n/a", x_r + pad_px, y_b + pad_px + 1 * line_h, size=font_small, rgba=dim)

            off_str = "n/a"
            if isinstance(offset, (list, tuple)) and len(offset) >= 3:
                off_str = f"{float(offset[0]):.2f},{float(offset[1]):.2f},{float(offset[2]):.2f}"
            dv_str = f"{depth_virt:.2f}" if isinstance(depth_virt, (int, float)) else "n/a"
            cbf_str = f"{cbf_val:.1f}" if isinstance(cbf_val, (int, float)) else "n/a"
            _add_text(f"OFF:{off_str} D_V:{dv_str} CBF:{cbf_str}", x_r + pad_px, y_b + pad_px + 2 * line_h, size=font_small, rgba=dim)

            if isinstance(body_rates, (list, tuple)) and len(body_rates) >= 3:
                _add_text(f"RATES: R:{float(body_rates[0]):+.2f} P:{float(body_rates[1]):+.2f} Y:{float(body_rates[2]):+.2f}", x_r + pad_px, y_b + pad_px + 3 * line_h, size=font_small, rgba=dim)
            else:
                _add_text("RATES: n/a", x_r + pad_px, y_b + pad_px + 3 * line_h, size=font_small, rgba=dim)

            # Tactical symbology: vector from screen center to virtual pixel (like python_test.py).
            if isinstance(virt_pix, (list, tuple)) and len(virt_pix) >= 2:
                try:
                    cx, cy = width // 2, height // 2
                    vx, vy = int(float(virt_pix[0])), int(float(virt_pix[1]))
                    vx = max(0, min(width - 1, vx))
                    vy = max(0, min(height - 1, vy))
                    _add_line(cx, cy, vx, vy, rgba=(1.0, 0.9, 0.2, 0.55), w=max(1, int(2 * scale)))
                    cross = max(8, int(10 * scale))
                    _add_line(vx - cross, vy, vx + cross, vy, rgba=(0.35, 1.0, 0.55, 0.70), w=1)
                    _add_line(vx, vy - cross, vx, vy + cross, rgba=(0.35, 1.0, 0.55, 0.70), w=1)
                except Exception:
                    pass

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # Optional frame extraction (GPU->CPU copy). Only available on RGBA pads.
            if self._enable_frame_extraction and self._frame_callbacks:
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    import cv2

                    frame_bgr = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGR)
                    with self._frame_lock:
                        self._latest_frame = frame_bgr
                    for callback in self._frame_callbacks:
                        try:
                            callback(frame_bgr, int(frame_meta.frame_num))
                        except Exception as e:
                            print(f"Frame callback error: {e}")
                except Exception as e:
                    if not hasattr(self, "_frame_extract_error_printed"):
                        print(f"Frame extraction error (GPU->CPU): {e}")
                        self._frame_extract_error_printed = True

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _load_tracker_config(self, tracker) -> None:
        """
        Load tracker configuration from config file.
        
        Based on deepstream-test2 example from NVIDIA.
        """
        cfg = self.config
        
        if cfg.tracker_config_file and os.path.exists(cfg.tracker_config_file):
            # Load from INI-style config file
            config = configparser.ConfigParser()
            config.read(cfg.tracker_config_file)
            
            for key in config['tracker']:
                if key == 'tracker-width':
                    tracker.set_property('tracker-width', config.getint('tracker', key))
                elif key == 'tracker-height':
                    tracker.set_property('tracker-height', config.getint('tracker', key))
                elif key == 'gpu-id':
                    tracker.set_property('gpu_id', config.getint('tracker', key))
                elif key == 'll-lib-file':
                    tracker.set_property('ll-lib-file', config.get('tracker', key))
                elif key == 'll-config-file':
                    tracker.set_property('ll-config-file', config.get('tracker', key))
                elif key == 'enable-past-frame':
                    tracker.set_property('enable-past-frame', config.getboolean('tracker', key))
        else:
            # Use direct config values
            tracker.set_property('gpu-id', cfg.tracker_gpu_id)
            tracker.set_property('tracker-width', cfg.tracker_width)
            tracker.set_property('tracker-height', cfg.tracker_height)
            tracker.set_property('ll-lib-file', cfg.tracker_lib_path)
            tracker.set_property('ll-config-file', cfg.tracker_ll_config_path)
            
            # Enable past frame metadata for tracking history
            if cfg.enable_past_frame_meta:
                try:
                    tracker.set_property('enable-past-frame', True)
                except Exception:
                    pass  # Property may not exist in older versions

    def _create_element(self, factory_name: str, name: str):
        """Create a GStreamer element with error checking."""
        element = Gst.ElementFactory.make(factory_name, name)
        if not element:
            raise RuntimeError(f"Unable to create {factory_name} ({name})")
        return element

    def _create_pipeline(self) -> Gst.Pipeline:
        """
        Create the DeepStream GStreamer pipeline.
        
        Pipeline structure:
        source -> [caps_filter] -> streammux -> pgie -> [tracker] -> nvvidconv -> tee (PRE-OSD)
            -> RAW branch (no overlay): encoder -> mux -> filesink
            -> Annotated branch: RGBA -> nvosd (HUD/BBox) -> tee -> [display] / [record]
        
        Based on official NVIDIA DeepStream Python examples.
        """
        print("Creating DeepStream Pipeline")
        pipeline = Gst.Pipeline()
        if not pipeline:
            raise RuntimeError("Unable to create Pipeline")
        
        cfg = self.config
        
        # === Source ===
        if cfg.source_type == "argus":
            print("Creating nvarguscamerasrc source")
            source = self._create_element("nvarguscamerasrc", "camera-source")
            source.set_property("sensor-id", cfg.sensor_id)
            source.set_property("tnr-mode", cfg.tnr_mode)
            source.set_property("tnr-strength", cfg.tnr_strength)
            source.set_property("ee-mode", cfg.ee_mode)
            source.set_property("ee-strength", cfg.ee_strength)
        elif cfg.source_type == "v4l2":
            print("Creating v4l2src source")
            source = self._create_element("v4l2src", "camera-source")
            source.set_property("device", cfg.source_path)
        elif cfg.source_type == "file":
            # Decode container streams to NVMM surfaces
            print(f"Creating uridecodebin source: {cfg.source_path}")
            uri = cfg.source_path
            if not uri.startswith("file://"):
                uri = "file://" + os.path.abspath(uri)

            source = Gst.Bin.new("file-source-bin")
            decodebin = self._create_element("uridecodebin", "file-decodebin")
            decodebin.set_property("uri", uri)
            decodebin.connect("pad-added", self._on_file_pad_added, source)
            source.add(decodebin)
        elif cfg.source_type == "appsrc":
            print("Creating appsrc source for ROS2 integration")
            source = self._create_element("appsrc", "app-source")
            source.set_property("is-live", True)
            source.set_property("format", 3)  # GST_FORMAT_TIME
            source.set_property("block", False)
            source.set_property("max-bytes", 0)
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
            )
            source.set_property("caps", caps)
            self._appsrc = source
        else:
            raise ValueError(f"Unknown source type: {cfg.source_type}")
        
        # === CPU to GPU converter (for appsrc) ===
        appsrc_conv = None
        appsrc_caps = None
        if cfg.source_type == "appsrc":
            print("Creating nvvideoconvert for appsrc (CPU->GPU)")
            appsrc_conv = self._create_element("nvvideoconvert", "appsrc-converter")
            # Use GPU compute mode (VIC doesn't support BGR format on Jetson)
            appsrc_conv.set_property("compute-hw", 1)
            appsrc_caps = self._create_element("capsfilter", "appsrc-caps")
            caps = Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height}"
            )
            appsrc_caps.set_property("caps", caps)
        
        # === Caps filter (for camera sources) ===
        caps_filter = None
        if cfg.source_type in ("argus", "v4l2"):
            print("Creating capsfilter")
            caps_filter = self._create_element("capsfilter", "caps-filter")
            caps = Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM), width={cfg.width}, height={cfg.height}, "
                f"format=NV12, framerate={cfg.fps}/1"
            )
            caps_filter.set_property("caps", caps)
        
        # === Stream muxer ===
        print("Creating nvstreammux")
        streammux = self._create_element("nvstreammux", "stream-muxer")
        self._streammux = streammux
        
        # Only set these properties if not using new gst-nvstreammux
        if not cfg.use_new_streammux and os.environ.get('USE_NEW_NVSTREAMMUX') != 'yes':
            streammux.set_property("width", cfg.width)
            streammux.set_property("height", cfg.height)
            # Calculate batch timeout based on FPS (80% of frame interval for timing margin)
            batch_timeout_usec = int(1_000_000 / cfg.fps * 0.8) if cfg.fps > 0 else 33000
            streammux.set_property("batched-push-timeout", batch_timeout_usec)
            print(f"Streammux batch-timeout: {batch_timeout_usec}μs (fps={cfg.fps})")
        streammux.set_property("batch-size", 1)
        # File inputs are not live; camera/rtsp should stay live for lower latency.
        streammux.set_property("live-source", 0 if cfg.source_type == "file" else 1)
        
        # === Primary inference ===
        print("Creating nvinfer (primary inference)")
        pgie = self._create_element("nvinfer", "primary-nvinference-engine")
        
        # Resolve PGIE config path and normalize relative paths inside the config.
        pgie_config_path = os.path.expanduser(os.path.expandvars(cfg.pgie_config_path or ""))
        if not pgie_config_path or not os.path.exists(pgie_config_path):
            raise FileNotFoundError(
                "DeepStream PGIE config not found. "
                "Set DEEPSTREAM_YOLO_DIR or update configs/app.yaml. "
                f"Got: {cfg.pgie_config_path!r}"
            )

        # Always generate a normalized config (absolute paths), plus optional overrides.
        print("Resolving PGIE config (absolute paths + optional overrides)")
        pgie_config_path = create_modified_pgie_config(pgie_config_path, cfg)
        pgie.set_property("config-file-path", pgie_config_path)
        
        # === Tracker (optional) ===
        tracker = None
        if cfg.enable_tracker:
            print("Creating nvtracker")
            tracker = self._create_element("nvtracker", "object-tracker")
            self._load_tracker_config(tracker)
        
        # === Video converter (pre-OSD) ===
        print("Creating nvvideoconvert (pre-OSD)")
        nvvidconv = self._create_element("nvvideoconvert", "convertor-preosd")
        nvvidconv.set_property("flip-method", cfg.flip_method)
        # Use GPU compute mode (VIC doesn't support BGR format on Jetson)
        nvvidconv.set_property("compute-hw", 1)

        # === Tee (must be BEFORE any OSD/HUD for raw cleanliness) ===
        tee_preosd = self._create_element("tee", "tee-preosd")

        def _abs_out(path: str) -> str:
            if not path:
                return ""
            return path if os.path.isabs(path) else os.path.join(self._original_cwd, path)

        annotated_out = _abs_out(cfg.annotated_output_path or cfg.output_path)
        raw_out = _abs_out(cfg.raw_output_path)

        enable_raw_recording = bool(cfg.enable_raw_recording and raw_out)
        enable_annotated_recording = bool(cfg.enable_annotated_recording and annotated_out)
        enable_annotated_branch = bool(enable_annotated_recording or cfg.enable_display)

        # === RAW recording branch (no overlays) ===
        raw_q = raw_nvvidconv = raw_caps = raw_enc = raw_parser = raw_mux = raw_sink = None
        if enable_raw_recording:
            os.makedirs(os.path.dirname(raw_out), exist_ok=True)
            print(f"Creating RAW recording branch: {raw_out}")

            raw_q = self._create_element("queue", "queue-raw")
            raw_q.set_property("max-size-buffers", 3)  # Reduced for lower latency

            raw_nvvidconv = self._create_element("nvvideoconvert", "convertor-raw")
            raw_nvvidconv.set_property("compute-hw", 1)

            raw_caps = self._create_element("capsfilter", "caps-raw")
            raw_caps.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
            ))

            raw_enc = self._create_element("nvv4l2h264enc", "encoder-raw")
            raw_enc.set_property("bitrate", cfg.bitrate)
            raw_enc.set_property("preset-level", 1)  # UltraFastPreset
            raw_enc.set_property("control-rate", 1)  # Variable bitrate
            raw_enc.set_property("maxperf-enable", True)

            raw_parser = self._create_element("h264parse", "parser-raw")
            raw_mux = self._create_element("mp4mux", "mux-raw")
            raw_mux.set_property("faststart", True)

            raw_sink = self._create_element("filesink", "filesink-raw")
            raw_sink.set_property("location", raw_out)
            raw_sink.set_property("sync", False)

        # === Annotated branch (HUD/BBox are rendered here only) ===
        ann_q = ann_rgba_conv = ann_rgba_caps = nvosd = tee_postosd = None
        display_q = display_sink = None
        ann_rec_q = ann_rec_conv = ann_rec_caps = ann_enc = ann_parser = ann_mux = ann_sink = None
        if enable_annotated_branch:
            print("Creating annotated branch (OSD/HUD)")

            ann_q = self._create_element("queue", "queue-annotated")
            ann_q.set_property("max-size-buffers", 3)  # Reduced for lower latency

            ann_rgba_conv = self._create_element("nvvideoconvert", "convertor-rgba")
            ann_rgba_conv.set_property("compute-hw", 1)

            ann_rgba_caps = self._create_element("capsfilter", "caps-rgba")
            ann_rgba_caps.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=RGBA,width={cfg.width},height={cfg.height}"
            ))

            nvosd = self._create_element("nvdsosd", "onscreendisplay")

            tee_postosd = self._create_element("tee", "tee-postosd")

            # Display branch (optional)
            if cfg.enable_display:
                display_q = self._create_element("queue", "queue-display")
                display_q.set_property("max-size-buffers", 1)
                display_q.set_property("leaky", 2)  # Downstream leaky

                if self._platform.is_integrated_gpu() or self._platform.is_platform_aarch64():
                    print("Creating nv3dsink")
                    display_sink = self._create_element("nv3dsink", "nv3d-sink")
                else:
                    print("Creating nveglglessink")
                    display_sink = self._create_element("nveglglessink", "nvvideo-renderer")
                display_sink.set_property("sync", False)

            # Annotated recording branch (optional)
            if enable_annotated_recording:
                os.makedirs(os.path.dirname(annotated_out), exist_ok=True)
                print(f"Creating ANNOTATED recording branch: {annotated_out}")

                ann_rec_q = self._create_element("queue", "queue-annotated-record")
                ann_rec_q.set_property("max-size-buffers", 3)  # Reduced for lower latency

                ann_rec_conv = self._create_element("nvvideoconvert", "convertor-annotated-record")
                ann_rec_conv.set_property("compute-hw", 1)

                ann_rec_caps = self._create_element("capsfilter", "caps-annotated-record")
                ann_rec_caps.set_property("caps", Gst.Caps.from_string(
                    f"video/x-raw(memory:NVMM),format=NV12,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
                ))

                ann_enc = self._create_element("nvv4l2h264enc", "encoder-annotated")
                ann_enc.set_property("bitrate", cfg.bitrate)
                ann_enc.set_property("preset-level", 1)
                ann_enc.set_property("control-rate", 1)
                ann_enc.set_property("maxperf-enable", True)

                ann_parser = self._create_element("h264parse", "parser-annotated")
                ann_mux = self._create_element("mp4mux", "mux-annotated")
                ann_mux.set_property("faststart", True)

                ann_sink = self._create_element("filesink", "filesink-annotated")
                ann_sink.set_property("location", annotated_out)
                ann_sink.set_property("sync", False)

        # Fallback sink if no output branches enabled (keeps pipeline running)
        null_q = null_sink = None
        if not enable_raw_recording and not enable_annotated_branch:
            print("No output sinks enabled; attaching fakesink")
            null_q = self._create_element("queue", "queue-fakesink")
            null_q.set_property("max-size-buffers", 1)
            null_q.set_property("leaky", 2)
            null_sink = self._create_element("fakesink", "fakesink")
            null_sink.set_property("sync", False)
        
        # === Add elements to pipeline ===
        print("Adding elements to Pipeline")
        pipeline.add(source)
        if appsrc_conv:
            pipeline.add(appsrc_conv)
            pipeline.add(appsrc_caps)
        if caps_filter:
            pipeline.add(caps_filter)
        pipeline.add(streammux)
        pipeline.add(pgie)
        if tracker:
            pipeline.add(tracker)
        pipeline.add(nvvidconv)
        pipeline.add(tee_preosd)
        if raw_q:
            pipeline.add(raw_q)
            pipeline.add(raw_nvvidconv)
            pipeline.add(raw_caps)
            pipeline.add(raw_enc)
            pipeline.add(raw_parser)
            pipeline.add(raw_mux)
            pipeline.add(raw_sink)
        if enable_annotated_branch:
            pipeline.add(ann_q)
            pipeline.add(ann_rgba_conv)
            pipeline.add(ann_rgba_caps)
            pipeline.add(nvosd)
            pipeline.add(tee_postosd)
            if display_q:
                pipeline.add(display_q)
                pipeline.add(display_sink)
            if ann_rec_q:
                pipeline.add(ann_rec_q)
                pipeline.add(ann_rec_conv)
                pipeline.add(ann_rec_caps)
                pipeline.add(ann_enc)
                pipeline.add(ann_parser)
                pipeline.add(ann_mux)
                pipeline.add(ann_sink)
        if null_q:
            pipeline.add(null_q)
            pipeline.add(null_sink)
        
        # === Link elements ===
        print("Linking elements in Pipeline")
        
        if cfg.source_type == "file":
            # decodebin will link dynamically in _on_file_pad_added
            pass
        else:
            if appsrc_conv:
                # appsrc -> nvvideoconvert -> capsfilter -> streammux
                source.link(appsrc_conv)
                appsrc_conv.link(appsrc_caps)
                srcpad = appsrc_caps.get_static_pad("src")
            elif caps_filter:
                source.link(caps_filter)
                srcpad = caps_filter.get_static_pad("src")
            else:
                srcpad = source.get_static_pad("src")

            # Link to streammux using request pad
            sinkpad = streammux.request_pad_simple("sink_0") if hasattr(streammux, 'request_pad_simple') \
                else streammux.get_request_pad("sink_0")
            if not sinkpad:
                raise RuntimeError("Unable to get sink pad of streammux")
            if not srcpad:
                raise RuntimeError("Unable to get source pad")
            srcpad.link(sinkpad)
        
        streammux.link(pgie)
        
        if tracker:
            pgie.link(tracker)
            tracker.link(nvvidconv)
        else:
            pgie.link(nvvidconv)
        
        nvvidconv.link(tee_preosd)

        # Pre-OSD metadata probe (always-on)
        preosd_srcpad = nvvidconv.get_static_pad("src")
        if not preosd_srcpad:
            raise RuntimeError("Unable to get src pad of pre-OSD nvvideoconvert")
        preosd_srcpad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe_callback, 0)

        # RAW branch links
        if raw_q:
            tee_preosd.link(raw_q)
            raw_q.link(raw_nvvidconv)
            raw_nvvidconv.link(raw_caps)
            raw_caps.link(raw_enc)
            raw_enc.link(raw_parser)
            raw_parser.link(raw_mux)
            raw_mux.link(raw_sink)

        # Annotated branch links
        if enable_annotated_branch:
            tee_preosd.link(ann_q)
            ann_q.link(ann_rgba_conv)
            ann_rgba_conv.link(ann_rgba_caps)
            ann_rgba_caps.link(nvosd)
            nvosd.link(tee_postosd)

            # HUD probe (renders overlay only on annotated branch)
            osdsinkpad = nvosd.get_static_pad("sink")
            if not osdsinkpad:
                raise RuntimeError("Unable to get sink pad of nvosd")
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self._hud_probe_callback, 0)

            if display_q:
                tee_postosd.link(display_q)
                display_q.link(display_sink)

            if ann_rec_q:
                tee_postosd.link(ann_rec_q)
                ann_rec_q.link(ann_rec_conv)
                ann_rec_conv.link(ann_rec_caps)
                ann_rec_caps.link(ann_enc)
                ann_enc.link(ann_parser)
                ann_parser.link(ann_mux)
                ann_mux.link(ann_sink)

        # Fallback sink links
        if null_q:
            tee_preosd.link(null_q)
            null_q.link(null_sink)
        
        return pipeline

    def _on_file_pad_added(self, decodebin, pad, source_bin) -> None:
        """
        Handle dynamic pad from uridecodebin for file sources.
        Converts to NVMM/NV12 and links into streammux sink_0.
        """
        caps = pad.get_current_caps()
        if not caps:
            return
        name = caps.get_structure(0).get_name()
        if not name.startswith("video/"):
            return

        print("Linking file decodebin -> nvvideoconvert -> streammux")

        queue = self._create_element("queue", f"file-queue-{pad.get_name()}")
        conv = self._create_element("nvvideoconvert", f"file-conv-{pad.get_name()}")
        conv.set_property("compute-hw", 1)
        capsfilter = self._create_element("capsfilter", f"file-caps-{pad.get_name()}")
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
        )

        if not self.pipeline:
            return
        # Add decode chain inside the same source bin for clean linking.
        source_bin.add(queue)
        source_bin.add(conv)
        source_bin.add(capsfilter)
        queue.sync_state_with_parent()
        conv.sync_state_with_parent()
        capsfilter.sync_state_with_parent()

        pad.link(queue.get_static_pad("sink"))
        queue.link(conv)
        conv.link(capsfilter)

        if not self._streammux:
            print("Streammux not ready; cannot link file source")
            return
        sinkpad = (
            self._streammux.request_pad_simple("sink_0")
            if hasattr(self._streammux, "request_pad_simple")
            else self._streammux.get_request_pad("sink_0")
        )
        capsfilter.get_static_pad("src").link(sinkpad)

    def _run_loop(self):
        """Run the GLib main loop (in separate thread)."""
        try:
            self.loop.run()
        except Exception as e:
            print(f"DeepStream loop error: {e}")
        finally:
            self._running = False

    def _bus_call(self, bus, message, loop):
        """GStreamer bus message handler (captures warnings/errors for HUD)."""
        t = message.type

        if t == Gst.MessageType.EOS:
            print("DeepStream: End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self._last_bus_warning = str(err)
            self._last_bus_message_ts = time.time()
            print(f"DeepStream Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._last_bus_error = str(err)
            self._last_bus_message_ts = time.time()
            print(f"DeepStream Error: {err}: {debug}")
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            if struct is not None and struct.get_name() == 'GstBinForwarded':
                forward_msg = struct.get_value('message')
                if forward_msg and forward_msg.type == Gst.MessageType.EOS:
                    print("DeepStream: EOS from element")

        return True

    def start(self) -> bool:
        """
        Start the DeepStream pipeline.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            print("DeepStream pipeline already running")
            return True
        
        try:
            # Create pipeline
            self.pipeline = self._create_pipeline()
            
            # Create main loop
            self.loop = GLib.MainLoop()
            
            # Add bus message handler
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._bus_call, self.loop)
            
            # Initialize timing
            self._start_time = time.time()
            self._frame_count = 0
            self._appsrc_push_count = 0
            
            # Start pipeline
            print("Starting pipeline")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Failed to start DeepStream pipeline")
                return False
            
            self._running = True
            
            # Run loop in separate thread
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            
            print("DeepStream pipeline started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start DeepStream pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """Stop the DeepStream pipeline."""
        if not self._running:
            return
        
        print("Stopping DeepStream pipeline...")
        
        # Send EOS for clean file closure
        if self.pipeline:
            print("Sending EOS event to pipeline...")
            self.pipeline.send_event(Gst.Event.new_eos())
            time.sleep(2)  # Wait for EOS to propagate
        
        # Stop the loop
        if self.loop:
            self.loop.quit()
        
        # Wait for thread
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Set pipeline to NULL
        if self.pipeline:
            print("Setting pipeline to NULL state...")
            self.pipeline.set_state(Gst.State.NULL)
        
        self._running = False
        print("DeepStream pipeline stopped")
        
        # Print statistics
        if self._frame_count > 0:
            total_time = time.time() - self._start_time
            avg_fps = self._frame_count / total_time if total_time > 0 else 0
            print(f"Total frames processed: {self._frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    def switch_camera(self, sensor_id: int) -> bool:
        """
        Switch to a different camera sensor (for dual camera setup).
        
        This pauses the pipeline, changes the sensor_id, and resumes.
        Note: Only works with nvarguscamerasrc source type.
        
        Args:
            sensor_id: Camera sensor ID (0 or 1)
            
        Returns:
            True if successful, False otherwise.
        """
        if self.config.source_type != "argus":
            print(f"switch_camera only works with 'argus' source type, current: {self.config.source_type}")
            return False
        
        if not self._running or not self.pipeline:
            print("Pipeline not running")
            return False
        
        try:
            print(f"Switching camera to sensor_id={sensor_id}...")
            
            # Pause pipeline
            self.pipeline.set_state(Gst.State.PAUSED)
            
            # Get the camera source element
            source = self.pipeline.get_by_name("camera-source")
            if not source:
                print("Could not find camera-source element")
                return False
            
            # Change sensor ID
            current_id = source.get_property("sensor-id")
            if current_id == sensor_id:
                print(f"Already on sensor_id={sensor_id}")
                self.pipeline.set_state(Gst.State.PLAYING)
                return True
            
            source.set_property("sensor-id", sensor_id)
            self.config.sensor_id = sensor_id
            
            # Resume pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Failed to resume pipeline after camera switch")
                return False
            
            print(f"Camera switched to sensor_id={sensor_id}")
            return True
            
        except Exception as e:
            print(f"Failed to switch camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_current_sensor_id(self) -> int:
        """Get the current camera sensor ID."""
        return self.config.sensor_id

    def push_frame(self, frame) -> bool:
        """
        Push a frame to the pipeline (for appsrc mode).
        
        Args:
            frame: numpy array (BGR format, HxWxC)
            
        Returns:
            True if successful, False otherwise.
        """
        if self.config.source_type != "appsrc" or not hasattr(self, '_appsrc'):
            print("push_frame only available in appsrc mode")
            return False
        
        if not self._running:
            return False
        
        try:
            # Convert frame to GstBuffer
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Set timestamp
            frame_duration = 10**9 // self.config.fps
            buf.pts = self._appsrc_push_count * frame_duration
            buf.duration = frame_duration
            
            # Push to appsrc
            ret = self._appsrc.emit("push-buffer", buf)
            ok = ret == Gst.FlowReturn.OK
            if ok:
                self._appsrc_push_count += 1
            return ok
        except Exception as e:
            print(f"Failed to push frame: {e}")
            return False

    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps

    @property
    def frame_count(self) -> int:
        """Get total processed frame count."""
        return self._frame_count


# =============================================================================
# Convenience function for simple usage
# =============================================================================
def create_deepstream_runner(
    width: int = 1920,
    height: int = 1080,
    fps: int = 35,
    pgie_config: str = str(ROOT_PGIE_CONFIG_YOLO11),
    enable_tracker: bool = True,
    enable_display: bool = False,
    output_path: str = "",
    source_type: str = "argus"
) -> DeepStreamRunner:
    """
    Create a DeepStreamRunner with common settings.
    
    Args:
        width, height: Resolution
        fps: Frame rate
        pgie_config: Path to primary inference config
        enable_tracker: Whether to enable object tracking
        enable_display: Whether to show live display
        output_path: Path for recording (empty to disable)
        source_type: Source type ("argus", "v4l2", "file", "appsrc")
        
    Returns:
        Configured DeepStreamRunner instance.
    """
    config = DeepStreamConfig(
        width=width,
        height=height,
        fps=fps,
        source_type=source_type,
        pgie_config_path=pgie_config,
        enable_tracker=enable_tracker,
        enable_display=enable_display,
        enable_recording=bool(output_path),
        output_path=output_path
    )
    return DeepStreamRunner(config)
