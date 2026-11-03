"""Детекторы объектов в кадре: ML (YOLO/ONNX), детектор цветных меток, «oracle».

Протокол: detect(frame_bgr) -> list[Detection2D]. Все координаты — в пикселях
оригинального кадра.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import cv2

from .types import Detection2D


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]: ...


def scale_to_model(width: int, height: int, model_size: int) -> tuple[int, int, float]:
    """Letterbox: сторона model_size, коэффициенты одинаковые по X/Y."""
    scale = model_size / max(width, height)
    new_w = int(round(width * scale)) // 2 * 2
    new_h = int(round(height * scale)) // 2 * 2
    pad_x = (model_size - new_w) // 2
    pad_y = (model_size - new_h) // 2
    return pad_x, pad_y, scale


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Жадный Non-Maximum Suppression; boxes: [N,4] в формате x1,y1,x2,y2."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        inter_w = np.minimum(x2[order[1:]], x2[i]) - np.maximum(x1[order[1:]], x1[i])
        inter_h = np.minimum(y2[order[1:]], y2[i]) - np.maximum(y1[order[1:]], y1[i])
        inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class YoloDetector:
    """YOLO через ultralytics (pip install ultralytics).

    ``class_names`` — список имён классов модели (например, ``["truck", "car",
    "bus"]``), транслируется в индексы после загрузки модели; ``classes`` —
    готовые индексы. ``imgsz`` — сторона входа модели (влияет на recall мелких
    объектов и на скорость: 960 даёт заметно лучший recall на 1080p).
    """

    def __init__(self, model: str = "yolov8n.pt", *, conf: float = 0.35,
                 classes: list[int] | None = None, device: str = "cpu",
                 imgsz: int = 640, class_names: list[str] | None = None,
                 agnostic_nms: bool = False):
        from ultralytics import YOLO  # ленивый импорт: без torch не ломает пакет
        self.model = YOLO(model)
        self.conf = conf
        self.device = device
        self.imgsz = int(imgsz)
        self.agnostic_nms = bool(agnostic_nms)
        names = getattr(self.model, "names", {}) or {}
        if class_names:
            name_to_id = {str(v): int(k) for k, v in names.items()}
            missing = [n for n in class_names if n not in name_to_id]
            if missing:
                raise ValueError(
                    f"в модели {model!r} нет классов {missing}; доступны: "
                    f"{sorted(name_to_id)[:20]}")
            self.classes = [name_to_id[n] for n in class_names]
        else:
            self.classes = classes

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False,
                                     classes=self.classes, device=self.device,
                                     imgsz=self.imgsz, agnostic_nms=self.agnostic_nms)
        detections: list[Detection2D] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(c) for c in box.xyxy[0]]
                cls = int(box.cls[0])
                det = Detection2D(camera="", timestamp=0.0,
                                  center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                                  score=float(box.conf[0]),
                                  bbox=(x1, y1, x2, y2), label=str(cls))
                detections.append(det)
        return detections


class OnnxDetector:
    """ONNX-детектор (onnxruntime). Выход модели: [N,5+] = x1,y1,x2,y2,score(,cls...)."""

    def __init__(self, model_path: str, *, input_size: int = 640,
                 conf: float = 0.35, output_name: str | None = None,
                 classes: list[int] | None = None):
        import onnxruntime as ort  # ленивый импорт
        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        out = self.sess.get_outputs()[0]
        self.output_name = output_name or out.name
        self.input_size = input_size
        self.conf = conf
        self.classes = classes

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        h, w = frame_bgr.shape[:2]
        pad_x, pad_y, scale = scale_to_model(w, h, self.input_size)
        resized = cv2.resize(frame_bgr, (self.input_size - 2 * pad_x,
                                         self.input_size - 2 * pad_y))
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        blob = canvas[:, :, ::-1].astype(np.float32)[None] / 255.0  # BGR->RGB
        out = self.sess.run([self.output_name],
                            {self.input_name: np.ascontiguousarray(blob)})[0]
        arr = np.asarray(out).reshape(-1, np.asarray(out).shape[-1])
        if arr.shape[1] < 5:
            return []
        boxes, scores = arr[:, :4], arr[:, 4]
        cls = arr[:, 5].astype(int) if arr.shape[1] > 5 else np.zeros(len(arr), int)
        mask = scores >= self.conf
        if self.classes is not None:
            mask &= np.isin(cls, self.classes)
        boxes, scores, cls = boxes[mask], scores[mask], cls[mask]
        keep = nms(boxes, scores)
        detections: list[Detection2D] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1 = (x1 - pad_x) / scale; x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale; y2 = (y2 - pad_y) / scale
            detections.append(Detection2D(camera="", timestamp=0.0,
                                          center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                                          score=float(scores[i]),
                                          bbox=(x1, y1, x2, y2),
                                          label=str(int(cls[i]))))
        return detections


class YoloOnnxDetector:
    """ONNX-экспорт ultralytics YOLO (``model.export(format="onnx")``).

    Зачем: на Windows без CUDA единственный быстрый GPU-путь для AMD —
    ``onnxruntime`` с ``DmlExecutionProvider`` (DirectML). Замер на
    RX 7900 XT: yolov8n @960 — 45 мс (CPU) против 5 мс (DML), ≈9×.

    Вход — ``[1, 3, S, S]`` RGB/255; выход ultralytics — ``[1, 4+nc, N]``
    (без objectness), приводится к строкам ``cx, cy, w, h, scores...``.
    Имена классов берутся из метаданных ONNX (``names``), иначе COCO.
    """

    COCO_NAMES = {2: "car", 5: "bus", 6: "train", 7: "truck"}

    def __init__(self, model_path: str, *, imgsz: int = 960, conf: float = 0.25,
                 providers: list[str] | None = None, class_names: list[str] | None = None,
                 iou: float = 0.45, pad_value: int = 114):
        import onnxruntime as ort  # ленивый импорт

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        requested = list(providers or ["CPUExecutionProvider"])
        if "CPUExecutionProvider" not in requested:
            requested.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                         providers=requested)
        self.providers = self.sess.get_providers()
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.pad_value = int(pad_value)

        names = None
        try:
            meta = self.sess.get_modelmeta().custom_metadata_map or {}
            if "names" in meta:
                import ast
                parsed = ast.literal_eval(meta["names"])
                if isinstance(parsed, dict):
                    names = {int(k): str(v) for k, v in parsed.items()}
        except Exception:
            names = None
        self.names = names or dict(self.COCO_NAMES)
        if class_names:
            name_to_id = {v: k for k, v in self.names.items()}
            missing = [n for n in class_names if n not in name_to_id]
            if missing:
                raise ValueError(f"в ONNX нет классов {missing}; доступны: "
                                 f"{sorted(name_to_id)[:20]}")
            self.classes = [name_to_id[n] for n in class_names]
        else:
            self.classes = None

    def _letterbox(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        dw = (self.imgsz - new_w) / 2.0
        dh = (self.imgsz - new_h) / 2.0
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), self.pad_value, np.uint8)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, scale, left, top

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        canvas, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        # BGR -> RGB, HWC -> CHW, [0,1], батч 1: ultralytics ждёт NCHW
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0)
        out = self.sess.run([self.output_name],
                            {self.input_name: np.ascontiguousarray(blob)})[0]
        arr = np.asarray(out)[0]
        if arr.ndim != 2:
            return []
        if arr.shape[0] < arr.shape[1]:      # [4+nc, N] -> [N, 4+nc]
            arr = arr.T
        if arr.shape[1] < 5:
            return []
        boxes_xywh = arr[:, :4]
        class_scores = arr[:, 4:]
        cls = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(arr)), cls]
        keep = scores >= self.conf
        if self.classes is not None:
            keep &= np.isin(cls, self.classes)
        if not keep.any():
            return []
        xywh = boxes_xywh[keep]
        scores = scores[keep].astype(float)
        cls = cls[keep]

        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

        detections: list[Detection2D] = []
        h, w = frame_bgr.shape[:2]
        for c in np.unique(cls):             # NMS по классам, как в ultralytics
            m = cls == c
            for i in nms(boxes[m], scores[m], self.iou):
                x1, y1, x2, y2 = boxes[m][i]
                x1 = (x1 - pad_x) / scale
                x2 = (x2 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                y2 = (y2 - pad_y) / scale
                x1, x2 = max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))
                y1, y2 = max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))
                detections.append(Detection2D(
                    camera="", timestamp=0.0,
                    center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                    score=float(scores[m][i]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    label=str(int(c))))
        return detections


class ColorBlobDetector:
    """Детектор цветных круглых меток (HSV-пороги + контуры).

    Специализированный детектор для синтетики и маркеров-наклеек: быстро,
    без ML, даёт точный центр метки.
    """

    def __init__(self, colors: list[tuple[int, int, int]], *,
                 hsv_tolerance: int = 25, min_area: float = 60.0,
                 max_area: float = 40000.0):
        """colors: список эталонных BGR цветов меток (индекс = class_id)."""
        self._ranges: list[tuple[np.ndarray, np.ndarray]] = []
        for bgr in colors:
            bgr = np.array(bgr, np.uint8)
            hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            lo = hsv.copy(); hi = hsv.copy()
            lo[0] = max(0, int(hsv[0]) - hsv_tolerance)
            hi[0] = min(179, int(hsv[0]) + hsv_tolerance)
            lo[1] = max(0, int(hsv[1]) - 70); hi[1] = 255
            lo[2] = max(0, int(hsv[2]) - 70); hi[2] = 255
            self._ranges.append((lo, hi))
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        detections: list[Detection2D] = []
        for cls, (lo, hi) in enumerate(self._ranges):
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                detections.append(Detection2D(camera="", timestamp=0.0,
                                              center=np.array([float(cx), float(cy)]),
                                              score=min(1.0, area / 1e4),
                                              bbox=(cx - radius, cy - radius, cx + radius, cy + radius),
                                              label=str(cls)))
        return detections


class OracleDetector:
    """Идеальный «детектор» по ground truth: только для численной валидации
    геометрии и трекинга (в синтетике ML-модель не знает нарисованные диски).

    Ground truth можно задать явно (``set_ground_truth``) или подвесить функцию
    ``set_ground_truth_source(fn)``, которую детектор вызывает перед каждым
    ``detect()`` — так офлайн-конвейер проецирует эталонные 3D-точки в пиксели
    каждой камеры без ручного управления.
    """

    def __init__(self, gt_by_frame: dict[int, dict[int, tuple[float, float]]] | None = None):
        self.gt_by_frame = gt_by_frame or {}
        self._pending: dict[int, tuple[float, float]] = {}
        self._source = None

    def set_ground_truth(self, gt: dict[int, tuple[float, float]]) -> None:
        self._pending = gt

    def set_ground_truth_source(self, fn) -> None:
        """``fn() -> {класс: (u, v)}`` вызывается перед каждым ``detect()``."""
        self._source = fn

    def detect(self, frame_bgr: np.ndarray) -> list[Detection2D]:
        gt = self._source() if self._source is not None else self._pending
        detections = [Detection2D(camera="", timestamp=0.0,
                                  center=np.array([u, v]), score=1.0,
                                  bbox=(u - 14, v - 14, u + 14, v + 14),
                                  label=str(cls))
                      for cls, (u, v) in gt.items()]
        return detections


def build_detector(spec: dict) -> Detector:
    """Детектор по словарю-спецификации из конфига.

    ``spec = {"kind": "blob", "colors": [[B, G, R], ...], ...}`` — цветные метки;
    ``{"kind": "oracle"}`` — эталонный детектор (только валидация);
    ``{"kind": "yolo", "model": "yolov8n.pt", "conf": 0.35, ...}`` — ultralytics;
    ``{"kind": "onnx", "model_path": "...", ...}`` — onnxruntime.
    """
    kind = str(spec.get("kind", "blob")).lower()
    if kind == "blob":
        colors = [tuple(int(c) for c in col) for col in spec["colors"]]
        return ColorBlobDetector(
            colors,
            hsv_tolerance=int(spec.get("hsv_tolerance", 25)),
            min_area=float(spec.get("min_area", 60.0)),
            max_area=float(spec.get("max_area", 40000.0)),
        )
    if kind == "oracle":
        return OracleDetector()
    if kind == "yolo":
        return YoloDetector(
            str(spec.get("model", "yolov8n.pt")),
            conf=float(spec.get("conf", 0.35)),
            classes=spec.get("classes"),
            device=str(spec.get("device", "cpu")),
            imgsz=int(spec.get("imgsz", 640)),
            class_names=spec.get("class_names"),
            agnostic_nms=bool(spec.get("agnostic_nms", False)),
        )
    if kind in ("yolo-onnx", "onnx-yolo"):
        return YoloOnnxDetector(
            str(spec.get("model_path") or spec["model"]),
            imgsz=int(spec.get("imgsz", 960)),
            conf=float(spec.get("conf", 0.25)),
            providers=spec.get("providers"),
            class_names=spec.get("class_names"),
            iou=float(spec.get("iou", 0.45)),
        )
    if kind == "onnx":
        return OnnxDetector(
            str(spec["model_path"]),
            input_size=int(spec.get("input_size", 640)),
            conf=float(spec.get("conf", 0.35)),
            output_name=spec.get("output_name"),
            classes=spec.get("classes"),
        )
    raise ValueError(f"неизвестный детектор: kind={kind!r} (blob/oracle/yolo/onnx)")
