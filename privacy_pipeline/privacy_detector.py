"""Pinned prediction-only face and number-plate detection."""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import patch

import numpy as np

from .privacy_models import (
    Detection,
    ModelLock,
    PixelBox,
    PrivacyClass,
    PrivacyConfig,
    sha256_file,
)

_MAX_FACE_FRAME_AREA_RATIO = 0.05
_MAX_PLATE_FRAME_AREA_RATIO = 0.01
_YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_YUNET_LICENSE_SHA256 = (
    "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"
)
_YUNET_SCORE_THRESHOLD = 0.30
_YUNET_STANDALONE_SCORE_THRESHOLD = 0.70
_MINIMUM_PLATE_ASPECT_RATIO = 1.5
_MAXIMUM_PLATE_ASPECT_RATIO = 6.5


@dataclass(frozen=True, slots=True)
class Tile:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        geometry = (self.x, self.y, self.width, self.height)
        if any(type(value) is not int for value in geometry):
            raise ValueError("tile coordinates must be exact integers")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("tile geometry is invalid")


class PredictionBackend(Protocol):
    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        frame_index: int,
        source: str,
        confidence: float,
    ) -> tuple[Detection, ...]: ...


class DetectorProtocol(Protocol):
    def detect(
        self, image_bgr: np.ndarray, *, frame_index: int
    ) -> tuple[Detection, ...]: ...


class BatchDetectorProtocol(Protocol):
    def detect_batch(
        self,
        images_bgr: tuple[np.ndarray, ...],
        *,
        frame_indices: tuple[int, ...],
    ) -> tuple[tuple[Detection, ...], ...]: ...


def detect_batch_frames(
    detector: DetectorProtocol,
    images_bgr: Sequence[np.ndarray],
    *,
    frame_indices: Sequence[int],
) -> tuple[tuple[Detection, ...], ...]:
    """Use a detector batch surface when present, retaining a scalar adapter."""

    images = tuple(images_bgr)
    indices = tuple(frame_indices)
    if not images or len(images) != len(indices):
        raise ValueError(
            "detector batch images and indices must be non-empty and equal"
        )
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("detector batch frame indices are invalid")
    batch_method = getattr(detector, "detect_batch", None)
    if callable(batch_method):
        result = tuple(batch_method(images, frame_indices=indices))
    else:
        result = tuple(
            detector.detect(image, frame_index=index)
            for image, index in zip(images, indices, strict=True)
        )
    if len(result) != len(indices) or any(
        any(item.frame_index != index for item in rows)
        for index, rows in zip(indices, result, strict=True)
    ):
        raise ValueError("detector batch result frame mapping differs")
    return result


class YuNetBackend(Protocol):
    def setInputSize(self, size: tuple[int, int]) -> None: ...

    def detect(self, image_bgr: np.ndarray) -> tuple[object, np.ndarray | None]: ...


def _axis_starts(length: int, edge: int, step: int) -> tuple[int, ...]:
    if length <= edge:
        return (0,)
    starts = list(range(0, length - edge + 1, step))
    final = length - edge
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def iter_tiles(
    width: int,
    height: int,
    *,
    edge: int,
    overlap_ratio: float,
) -> Iterator[Tile]:
    """Yield deterministic overlapping tiles, excluding a redundant full-frame tile."""

    if any(type(value) is not int or value <= 0 for value in (width, height, edge)):
        raise ValueError("frame and tile dimensions must be positive integers")
    if (
        type(overlap_ratio) not in {int, float}
        or isinstance(overlap_ratio, bool)
        or not math.isfinite(overlap_ratio)
        or not 0.0 < overlap_ratio < 0.5
    ):
        raise ValueError("tile overlap ratio must be within (0, 0.5)")
    if width <= edge and height <= edge:
        return
    step = max(1, round(edge * (1.0 - overlap_ratio)))
    for y in _axis_starts(height, edge, step):
        for x in _axis_starts(width, edge, step):
            yield Tile(x, y, min(edge, width - x), min(edge, height - y))


def _intersection_over_union(left: PixelBox, right: PixelBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    return intersection / (left.area + right.area - intersection)


def _detection_order(item: Detection) -> tuple[object, ...]:
    return (
        item.frame_index,
        -item.confidence,
        item.privacy_class.value,
        item.box.y1,
        item.box.x1,
        item.box.y2,
        item.box.x2,
        item.source,
    )


def _is_plausible_frame_detection(item: Detection, frame_area: int) -> bool:
    maximum_ratio = (
        _MAX_FACE_FRAME_AREA_RATIO
        if item.privacy_class is PrivacyClass.FACE
        else _MAX_PLATE_FRAME_AREA_RATIO
    )
    return item.box.area <= frame_area * maximum_ratio


def class_aware_nms(
    detections: Iterable[Detection],
    *,
    iou_threshold: float = 0.50,
) -> tuple[Detection, ...]:
    """Suppress overlapping detections only within the same privacy class."""

    if (
        type(iou_threshold) not in {int, float}
        or isinstance(iou_threshold, bool)
        or not 0.0 < iou_threshold < 1.0
    ):
        raise ValueError("NMS threshold must be within (0, 1)")
    ordered = sorted(tuple(detections), key=_detection_order)
    kept: list[Detection] = []
    for candidate in ordered:
        if any(
            candidate.frame_index == existing.frame_index
            and candidate.privacy_class is existing.privacy_class
            and _intersection_over_union(candidate.box, existing.box) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(kept)


def _temporally_matches(left: PixelBox, right: PixelBox) -> bool:
    left_x, left_y = _center(left)
    right_x, right_y = _center(right)
    maximum_width = max(left.x2 - left.x1, right.x2 - right.x1)
    maximum_height = max(left.y2 - left.y1, right.y2 - right.y1)
    minimum_area = min(left.area, right.area)
    maximum_area = max(left.area, right.area)
    return (
        abs(left_x - right_x) <= maximum_width
        and abs(left_y - right_y) <= maximum_height
        and maximum_area <= minimum_area * 4
    )


def filter_video_plate_detections(
    detections: Iterable[Detection],
    *,
    minimum_confidence: float,
) -> tuple[Detection, ...]:
    """Keep plausible plates confirmed in an immediately adjacent frame."""

    if (
        type(minimum_confidence) not in {int, float}
        or isinstance(minimum_confidence, bool)
        or not math.isfinite(minimum_confidence)
        or not 0.0 < minimum_confidence <= 1.0
    ):
        raise ValueError("minimum plate confidence must be within (0, 1]")
    items = tuple(detections)
    eligible = tuple(
        item
        for item in items
        if item.privacy_class is PrivacyClass.LICENSE_PLATE
        and item.confidence >= minimum_confidence
        and _MINIMUM_PLATE_ASPECT_RATIO
        <= (item.box.x2 - item.box.x1) / (item.box.y2 - item.box.y1)
        <= _MAXIMUM_PLATE_ASPECT_RATIO
    )
    confirmed = {
        item
        for item in eligible
        if any(
            abs(item.frame_index - other.frame_index) == 1
            and _temporally_matches(item.box, other.box)
            for other in eligible
            if other is not item
        )
    }
    return tuple(
        item
        for item in items
        if item.privacy_class is not PrivacyClass.LICENSE_PLATE or item in confirmed
    )


def _center(box: PixelBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _expanded_confirmation_box(box: PixelBox, width: int, height: int) -> PixelBox:
    center_x, center_y = _center(box)
    half_width = (box.x2 - box.x1) * 0.75
    half_height = (box.y2 - box.y1) * 0.75
    return PixelBox(
        max(0, round(center_x - half_width)),
        max(0, round(center_y - half_height)),
        min(width, round(center_x + half_width)),
        min(height, round(center_y + half_height)),
    )


def _boxes_confirm(primary: PixelBox, yunet: PixelBox, width: int, height: int) -> bool:
    primary_expanded = _expanded_confirmation_box(primary, width, height)
    yunet_expanded = _expanded_confirmation_box(yunet, width, height)
    yunet_center = _center(yunet)
    primary_center = _center(primary)
    return (
        primary_expanded.x1 <= yunet_center[0] <= primary_expanded.x2
        and primary_expanded.y1 <= yunet_center[1] <= primary_expanded.y2
    ) or (
        yunet_expanded.x1 <= primary_center[0] <= yunet_expanded.x2
        and yunet_expanded.y1 <= primary_center[1] <= yunet_expanded.y2
    )


def union_privacy_detections(
    primary: Iterable[Detection],
    yunet: Iterable[Detection],
    *,
    width: int,
    height: int,
    strong_face_confidence: float,
) -> tuple[Detection, ...]:
    """Keep primary detections and add only high-confidence YuNet faces."""

    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("union dimensions must be positive integers")
    if (
        type(strong_face_confidence) not in {int, float}
        or isinstance(strong_face_confidence, bool)
        or not 0.0 < strong_face_confidence <= 1.0
    ):
        raise ValueError("strong face confidence must be within (0, 1]")
    primary_items = tuple(primary)
    yunet_faces = tuple(yunet)
    if any(item.privacy_class is not PrivacyClass.FACE for item in yunet_faces):
        raise ValueError("YuNet union input may contain only faces")
    plates = tuple(
        item
        for item in primary_items
        if item.privacy_class is PrivacyClass.LICENSE_PLATE
    )
    strong_faces = tuple(
        item
        for item in primary_items
        if item.privacy_class is PrivacyClass.FACE
        and item.confidence >= strong_face_confidence
    )
    confirmed_weak_faces = tuple(
        item
        for item in primary_items
        if item.privacy_class is PrivacyClass.FACE
        and item.confidence < strong_face_confidence
        and any(
            _boxes_confirm(item.box, face.box, width, height) for face in yunet_faces
        )
    )
    standalone_yunet_faces = tuple(
        item
        for item in yunet_faces
        if item.confidence >= _YUNET_STANDALONE_SCORE_THRESHOLD
    )
    return class_aware_nms(
        (*plates, *strong_faces, *confirmed_weak_faces, *standalone_yunet_faces)
    )


class UnionPrivacyDetector:
    """Combine the pinned primary detector with an additive face detector."""

    def __init__(
        self,
        primary: DetectorProtocol,
        yunet: DetectorProtocol,
        *,
        strong_face_confidence: float,
    ) -> None:
        self._primary = primary
        self._yunet = yunet
        self._strong_face_confidence = strong_face_confidence

    def detect(
        self, image_bgr: np.ndarray, *, frame_index: int
    ) -> tuple[Detection, ...]:
        height, width = image_bgr.shape[:2]
        return union_privacy_detections(
            self._primary.detect(image_bgr, frame_index=frame_index),
            self._yunet.detect(image_bgr, frame_index=frame_index),
            width=width,
            height=height,
            strong_face_confidence=self._strong_face_confidence,
        )

    def detect_batch(
        self,
        images_bgr: tuple[np.ndarray, ...],
        *,
        frame_indices: tuple[int, ...],
    ) -> tuple[tuple[Detection, ...], ...]:
        primary = detect_batch_frames(
            self._primary, images_bgr, frame_indices=frame_indices
        )
        yunet = detect_batch_frames(
            self._yunet, images_bgr, frame_indices=frame_indices
        )
        return tuple(
            union_privacy_detections(
                primary_rows,
                yunet_rows,
                width=image.shape[1],
                height=image.shape[0],
                strong_face_confidence=self._strong_face_confidence,
            )
            for image, primary_rows, yunet_rows in zip(
                images_bgr, primary, yunet, strict=True
            )
        )


class YuNetFaceDetector:
    """Convert native-resolution OpenCV YuNet rows into privacy detections."""

    def __init__(self, backend: YuNetBackend) -> None:
        self._backend = backend

    def detect(
        self, image_bgr: np.ndarray, *, frame_index: int
    ) -> tuple[Detection, ...]:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
        ):
            raise ValueError("YuNet input must be an HxWx3 uint8 BGR image")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame index must be a non-negative integer")
        height, width = image_bgr.shape[:2]
        self._backend.setInputSize((width, height))
        _, rows = self._backend.detect(image_bgr)
        if rows is None:
            return ()
        result: list[Detection] = []
        for row in rows:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            x1 = max(0, min(width - 1, round(x)))
            y1 = max(0, min(height - 1, round(y)))
            x2 = max(x1 + 1, min(width, round(x + box_width)))
            y2 = max(y1 + 1, min(height, round(y + box_height)))
            detection = Detection(
                frame_index,
                PrivacyClass.FACE,
                float(row[-1]),
                PixelBox(x1, y1, x2, y2),
                "yunet",
            )
            if _is_plausible_frame_detection(detection, width * height):
                result.append(detection)
        return tuple(result)

    def detect_batch(
        self,
        images_bgr: tuple[np.ndarray, ...],
        *,
        frame_indices: tuple[int, ...],
    ) -> tuple[tuple[Detection, ...], ...]:
        return tuple(
            self.detect(image, frame_index=index)
            for image, index in zip(images_bgr, frame_indices, strict=True)
        )


def load_pretrained_yunet(model_path: Path, license_path: Path) -> YuNetFaceDetector:
    """Load only the exact reviewed OpenCV Zoo YuNet model and license."""

    if sha256_file(model_path) != _YUNET_MODEL_SHA256:
        raise ValueError("YuNet model SHA-256 differs from the reviewed artifact")
    if sha256_file(license_path) != _YUNET_LICENSE_SHA256:
        raise ValueError("YuNet license SHA-256 differs from the reviewed artifact")
    import cv2

    backend = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        _YUNET_SCORE_THRESHOLD,
        0.3,
        5000,
    )
    if (
        sha256_file(model_path) != _YUNET_MODEL_SHA256
        or sha256_file(license_path) != _YUNET_LICENSE_SHA256
    ):
        raise OSError("YuNet artifact changed while loading")
    return YuNetFaceDetector(backend)


def detect_frame(
    backend: PredictionBackend,
    image_bgr: np.ndarray,
    *,
    frame_index: int,
    config: PrivacyConfig,
) -> tuple[Detection, ...]:
    """Union a full-frame pass with translated tiled predictions."""

    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.dtype != np.uint8
    ):
        raise ValueError("detector input must be an HxWx3 uint8 BGR image")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame index must be a non-negative integer")
    if type(config) is not PrivacyConfig:
        raise ValueError("detector config must be a PrivacyConfig")
    height, width = image_bgr.shape[:2]
    frame_area = width * height
    combined: list[Detection] = []
    full = backend.predict(
        image_bgr,
        frame_index=frame_index,
        source="full",
        confidence=config.confidence,
    )
    for item in full:
        detection = Detection(
            frame_index,
            item.privacy_class,
            item.confidence,
            item.box.clip(width, height),
            "full",
        )
        if _is_plausible_frame_detection(detection, frame_area):
            combined.append(detection)
    for tile in iter_tiles(
        width,
        height,
        edge=config.tile_edge,
        overlap_ratio=config.tile_overlap_ratio,
    ):
        source = f"tile:{tile.x}:{tile.y}"
        crop = image_bgr[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        for item in backend.predict(
            crop,
            frame_index=frame_index,
            source=source,
            confidence=config.confidence,
        ):
            translated = PixelBox(
                item.box.x1 + tile.x,
                item.box.y1 + tile.y,
                item.box.x2 + tile.x,
                item.box.y2 + tile.y,
            ).clip(width, height)
            detection = Detection(
                frame_index,
                item.privacy_class,
                item.confidence,
                translated,
                source,
            )
            if _is_plausible_frame_detection(detection, frame_area):
                combined.append(detection)
    return class_aware_nms(combined)


def _normalized_class(value: object) -> PrivacyClass:
    if type(value) is not str:
        raise ValueError("model class map contains a non-string name")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "face": PrivacyClass.FACE,
        "faces": PrivacyClass.FACE,
        "license_plate": PrivacyClass.LICENSE_PLATE,
        "licence_plate": PrivacyClass.LICENSE_PLATE,
        "number_plate": PrivacyClass.LICENSE_PLATE,
        "license": PrivacyClass.LICENSE_PLATE,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("model class map contains an unsupported class") from error


class PretrainedYoloDetector:
    """Prediction-only Ultralytics adapter bound to a stable checkpoint."""

    def __init__(
        self,
        checkpoint: Path,
        lock: ModelLock,
        config: PrivacyConfig,
    ) -> None:
        if type(lock) is not ModelLock or type(config) is not PrivacyConfig:
            raise ValueError("detector requires locked model and privacy contracts")
        if sha256_file(checkpoint) != lock.checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 differs from the model lock")
        model = load_pretrained_yolo(checkpoint)
        names: Any = model.names
        if isinstance(names, dict):
            ordered_names = tuple(names[index] for index in sorted(names))
        elif isinstance(names, (list, tuple)):
            ordered_names = tuple(names)
        else:
            raise ValueError("model class map has an unsupported shape")
        resolved = tuple(_normalized_class(value) for value in ordered_names)
        if len(resolved) != 2 or set(resolved) != set(PrivacyClass):
            raise ValueError("model class map must resolve to face and license_plate")
        if sha256_file(checkpoint) != lock.checkpoint_sha256:
            raise OSError("checkpoint changed while loading the model")
        self._checkpoint = checkpoint
        self._lock = lock
        self._config = config
        self._model = model
        import torch
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._class_by_index = resolved

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        frame_index: int,
        source: str,
        confidence: float,
    ) -> tuple[Detection, ...]:
        results: Any = self._model.predict(
            source=image_bgr,
            conf=confidence,
            device=self._device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("YOLO returned an unexpected result count")
        boxes: Any = results[0].boxes
        if boxes is None:
            return ()
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        detections = []
        for coordinates_row, score, class_index_raw in zip(
            coordinates, confidences, classes, strict=True
        ):
            class_index = int(class_index_raw)
            if class_index < 0 or class_index >= len(self._class_by_index):
                raise ValueError("YOLO emitted an unknown class index")
            x1, y1, x2, y2 = (round(value) for value in coordinates_row)
            detections.append(
                Detection(
                    frame_index,
                    self._class_by_index[class_index],
                    float(score),
                    PixelBox(x1, y1, x2, y2),
                    source,
                )
            )
        return tuple(detections)

    def detect(
        self, image_bgr: np.ndarray, *, frame_index: int
    ) -> tuple[Detection, ...]:
        return detect_frame(
            self,
            image_bgr,
            frame_index=frame_index,
            config=self._config,
        )

    def detect_batch(
        self,
        images_bgr: tuple[np.ndarray, ...],
        *,
        frame_indices: tuple[int, ...],
    ) -> tuple[tuple[Detection, ...], ...]:
        """Batch all full-frame and tiled crops in one model invocation."""

        if not images_bgr or len(images_bgr) != len(frame_indices):
            raise ValueError(
                "YOLO batch images and indices must be non-empty and equal"
            )
        crops: list[np.ndarray] = []
        metadata: list[tuple[int, str, int, int, int, int, int]] = []
        positions = {
            frame_index: position for position, frame_index in enumerate(frame_indices)
        }
        if len(positions) != len(frame_indices) or any(
            type(index) is not int or index < 0 for index in frame_indices
        ):
            raise ValueError(
                "YOLO batch frame indices must be unique non-negative integers"
            )
        for image, frame_index in zip(images_bgr, frame_indices, strict=True):
            if (
                not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3
                or image.dtype != np.uint8
            ):
                raise ValueError("YOLO batch input must be HxWx3 uint8 BGR")
            height, width = image.shape[:2]
            frame_area = width * height
            crops.append(image)
            metadata.append((frame_index, "full", 0, 0, width, height, frame_area))
            for tile in iter_tiles(
                width,
                height,
                edge=self._config.tile_edge,
                overlap_ratio=self._config.tile_overlap_ratio,
            ):
                source = f"tile:{tile.x}:{tile.y}"
                crops.append(
                    image[
                        tile.y : tile.y + tile.height,
                        tile.x : tile.x + tile.width,
                    ]
                )
                metadata.append(
                    (
                        frame_index,
                        source,
                        tile.x,
                        tile.y,
                        tile.width,
                        tile.height,
                        frame_area,
                    )
                )
        results: Any = self._model.predict(
            source=crops,
            conf=self._config.confidence,
            device=self._device,
            verbose=False,
        )
        if len(results) != len(metadata):
            raise RuntimeError("YOLO returned an unexpected batch result count")
        grouped: list[list[Detection]] = [[] for _ in images_bgr]
        for result, item in zip(results, metadata, strict=True):
            (
                frame_index,
                source,
                offset_x,
                offset_y,
                crop_width,
                crop_height,
                frame_area,
            ) = item
            boxes: Any = result.boxes
            if boxes is None:
                continue
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for coordinates_row, score, class_index_raw in zip(
                coordinates, confidences, classes, strict=True
            ):
                class_index = int(class_index_raw)
                if class_index < 0 or class_index >= len(self._class_by_index):
                    raise ValueError("YOLO emitted an unknown class index")
                x1, y1, x2, y2 = (round(value) for value in coordinates_row)
                x1 = max(0, min(x1, crop_width))
                y1 = max(0, min(y1, crop_height))
                x2 = max(0, min(x2, crop_width))
                y2 = max(0, min(y2, crop_height))
                if x2 <= x1 or y2 <= y1:
                    continue
                local = PixelBox(x1, y1, x2, y2)
                translated = PixelBox(
                    local.x1 + offset_x,
                    local.y1 + offset_y,
                    local.x2 + offset_x,
                    local.y2 + offset_y,
                )
                detection = Detection(
                    frame_index,
                    self._class_by_index[class_index],
                    float(score),
                    translated,
                    source,
                )
                if _is_plausible_frame_detection(detection, frame_area):
                    grouped[positions[frame_index]].append(detection)
        return tuple(class_aware_nms(rows) for rows in grouped)


def load_pretrained_yolo(checkpoint: Path) -> Any:
    """Load the pinned legacy checkpoint under PyTorch's explicit trust switch."""

    import torch
    import ultralytics.utils as current_utils
    import ultralytics.utils.loss as current_loss
    import ultralytics.utils.tal as current_tal
    from ultralytics import YOLO

    original_load = torch.load

    def trusted_legacy_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    legacy_yolo = types.ModuleType("ultralytics.yolo")
    legacy_yolo.utils = current_utils  # type: ignore[attr-defined]
    aliases = {
        "ultralytics.yolo": legacy_yolo,
        "ultralytics.yolo.utils": current_utils,
        "ultralytics.yolo.utils.loss": current_loss,
        "ultralytics.yolo.utils.tal": current_tal,
    }
    with (
        patch.object(torch, "load", trusted_legacy_load),
        patch.dict(sys.modules, aliases),
    ):
        return YOLO(str(checkpoint), task="detect")
