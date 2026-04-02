#!/usr/bin/env python3

import os
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def load_labels(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def tensor_mode_is_input(mode: object) -> bool:
    return mode == trt.TensorIOMode.INPUT


def trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    box[0] = np.clip(box[0], 0, width - 1)
    box[1] = np.clip(box[1], 0, height - 1)
    box[2] = np.clip(box[2], 0, width - 1)
    box[3] = np.clip(box[3], 0, height - 1)
    return box


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = np.maximum(box_area + boxes_area - inter_area, 1e-6)
    return inter_area / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, top_k: int) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < top_k:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(boxes[current], boxes[remaining])
        order = remaining[overlaps < iou_threshold]

    return keep


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), line_thickness)
        label = f"{det.label} {det.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        text_y = max(y1, text_h + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 6, text_y),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


class TensorRTDetector:
    def __init__(
        self,
        engine_path: str,
        labels: Sequence[str],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        top_k: int = 100,
    ) -> None:
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TensorRT inference requires a CUDA-capable runtime.")

        self.labels = list(labels)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.device = torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        self.is_trt10 = hasattr(self.engine, "num_io_tensors")

        self.input_name, self.output_name = self._discover_io_tensors()
        self.input_dtype = self._get_tensor_dtype(self.input_name)
        self.output_dtype = self._get_tensor_dtype(self.output_name)

        self.input_shape = self._resolve_tensor_shape(self.input_name)
        if len(self.input_shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got {self.input_shape}")
        self.batch_size, self.channels, self.input_height, self.input_width = self.input_shape
        if self.batch_size != 1:
            raise ValueError(f"Only batch size 1 is supported, engine uses {self.batch_size}")
        if self.channels != 3:
            raise ValueError(f"Expected 3 input channels, engine uses {self.channels}")

        self._set_input_shape(self.input_name, self.input_shape)
        self.output_shape = self._get_context_shape(self.output_name)
        if any(dim < 0 for dim in self.output_shape):
            self.output_shape = self._resolve_tensor_shape(self.output_name)

        self.input_tensor = torch.empty(self.input_shape, dtype=self.input_dtype, device=self.device)
        self.output_tensor = torch.empty(self.output_shape, dtype=self.output_dtype, device=self.device)

        self.use_execute_v3 = hasattr(self.context, "execute_async_v3") and hasattr(
            self.context, "set_tensor_address"
        )
        if self.use_execute_v3:
            self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
            self.bindings = None
        else:
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.engine.get_binding_index(self.input_name)] = int(self.input_tensor.data_ptr())
            self.bindings[self.engine.get_binding_index(self.output_name)] = int(self.output_tensor.data_ptr())

    def _discover_io_tensors(self) -> Tuple[str, str]:
        input_names: List[str] = []
        output_names: List[str] = []

        if self.is_trt10:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if tensor_mode_is_input(self.engine.get_tensor_mode(name)):
                    input_names.append(name)
                else:
                    output_names.append(name)
        else:
            for index in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(index)
                if self.engine.binding_is_input(index):
                    input_names.append(name)
                else:
                    output_names.append(name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError(
                f"Expected one input and one output tensor, got inputs={input_names}, outputs={output_names}"
            )
        return input_names[0], output_names[0]

    def _get_tensor_dtype(self, tensor_name: str) -> torch.dtype:
        if self.is_trt10:
            return trt_dtype_to_torch(self.engine.get_tensor_dtype(tensor_name))
        index = self.engine.get_binding_index(tensor_name)
        return trt_dtype_to_torch(self.engine.get_binding_dtype(index))

    def _resolve_tensor_shape(self, tensor_name: str) -> Tuple[int, ...]:
        if self.is_trt10:
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(tensor_name))
            if all(dim > 0 for dim in shape):
                return shape
            if hasattr(self.engine, "get_tensor_profile_shape"):
                _, opt_shape, _ = self.engine.get_tensor_profile_shape(tensor_name, 0)
                return tuple(int(dim) for dim in opt_shape)
        else:
            index = self.engine.get_binding_index(tensor_name)
            shape = tuple(int(dim) for dim in self.engine.get_binding_shape(index))
            if all(dim > 0 for dim in shape):
                return shape
            min_shape, opt_shape, max_shape = self.engine.get_profile_shape(0, index)
            del min_shape, max_shape
            return tuple(int(dim) for dim in opt_shape)

        raise ValueError(f"Could not resolve tensor shape for {tensor_name}")

    def _set_input_shape(self, tensor_name: str, shape: Tuple[int, ...]) -> None:
        if self.is_trt10 and hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(tensor_name, shape)
            return
        if not self.is_trt10:
            index = self.engine.get_binding_index(tensor_name)
            self.context.set_binding_shape(index, shape)

    def _get_context_shape(self, tensor_name: str) -> Tuple[int, ...]:
        if self.is_trt10:
            return tuple(int(dim) for dim in self.context.get_tensor_shape(tensor_name))
        index = self.engine.get_binding_index(tensor_name)
        return tuple(int(dim) for dim in self.context.get_binding_shape(index))

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        image_h, image_w = image_bgr.shape[:2]
        scale = min(self.input_width / image_w, self.input_height / image_h)
        resized_w = int(round(image_w * scale))
        resized_h = int(round(image_h * scale))

        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)

        pad_left = (self.input_width - resized_w) // 2
        pad_top = (self.input_height - resized_h) // 2
        canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        if self.input_dtype == torch.float16:
            tensor = tensor.astype(np.float16)

        return tensor, scale, pad_left, pad_top

    def _postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad_left: int,
        pad_top: int,
    ) -> List[Detection]:
        image_h, image_w = original_shape
        predictions = raw_output[0] if raw_output.ndim == 3 else raw_output
        if predictions.size == 0:
            return []

        scores = predictions[:, 4]
        keep_mask = scores >= self.confidence_threshold
        if not np.any(keep_mask):
            return []

        predictions = predictions[keep_mask]
        boxes = predictions[:, :4].copy()
        scores = predictions[:, 4].astype(np.float32)
        class_ids = predictions[:, 5].astype(np.int32)

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not np.any(valid_mask):
            return []

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        kept_indices: List[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            class_keep = nms(
                boxes[class_indices],
                scores[class_indices],
                self.iou_threshold,
                self.top_k,
            )
            kept_indices.extend(class_indices[index] for index in class_keep)

        kept_indices = sorted(kept_indices, key=lambda index: float(scores[index]), reverse=True)
        kept_indices = kept_indices[: self.top_k]

        detections: List[Detection] = []
        for index in kept_indices:
            box = clip_box(boxes[index], image_w, image_h)
            class_id = int(class_ids[index])
            label = self.labels[class_id] if 0 <= class_id < len(self.labels) else f"class_{class_id}"
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    score=float(scores[index]),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return detections

    def infer(self, image_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        preprocessed, scale, pad_left, pad_top = self._preprocess(image_bgr)

        start = time.perf_counter()
        with torch.cuda.stream(self.stream):
            self.input_tensor.copy_(torch.from_numpy(preprocessed), non_blocking=False)
            if self.use_execute_v3:
                success = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            elif hasattr(self.context, "execute_async_v2"):
                success = self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
            else:
                success = self.context.execute_v2(self.bindings)
        self.stream.synchronize()
        if not success:
            raise RuntimeError("TensorRT execution failed.")

        inference_ms = (time.perf_counter() - start) * 1000.0
        raw_output = self.output_tensor.float().cpu().numpy()
        detections = self._postprocess(
            raw_output=raw_output,
            original_shape=image_bgr.shape[:2],
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return detections, inference_ms
