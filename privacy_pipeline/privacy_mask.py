"""Deterministic temporal propagation and strong identifier blur."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .privacy_models import Detection, PixelBox, PrivacyClass, PrivacyConfig


class OpticalFlowProtocol(Protocol):
    def interpolate(
        self,
        previous: PixelBox,
        following: PixelBox,
        fraction: float,
    ) -> PixelBox: ...


@dataclass(frozen=True, slots=True)
class BlurAudit:
    box_count: int
    kernel_width: int
    kernel_height: int
    boxes_sha256: str


def _center(box: PixelBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _iou(left: PixelBox, right: PixelBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    return intersection / (left.area + right.area - intersection)


def _can_associate(left: PixelBox, right: PixelBox) -> bool:
    if _iou(left, right) >= 0.05:
        return True
    left_center = _center(left)
    right_center = _center(right)
    distance = math.dist(left_center, right_center)
    scale = max(
        left.x2 - left.x1,
        left.y2 - left.y1,
        right.x2 - right.x1,
        right.y2 - right.y1,
    )
    return distance <= scale * 2.5


def _linear_box(left: PixelBox, right: PixelBox, fraction: float) -> PixelBox:
    values = tuple(
        round(start + (end - start) * fraction)
        for start, end in zip(
            (left.x1, left.y1, left.x2, left.y2),
            (right.x1, right.y1, right.x2, right.y2),
            strict=True,
        )
    )
    return PixelBox(*values)


def _box_union(left: PixelBox, right: PixelBox) -> PixelBox:
    return PixelBox(
        min(left.x1, right.x1),
        min(left.y1, right.y1),
        max(left.x2, right.x2),
        max(left.y2, right.y2),
    )


def propagate_short_gaps(
    detections: Sequence[Detection],
    *,
    frame_count: int,
    max_gap: int,
    flow: OpticalFlowProtocol | None,
) -> tuple[Detection, ...]:
    """Interpolate matching same-class detections across bounded missing frames."""

    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame count must be a positive integer")
    if type(max_gap) is not int or not 0 <= max_gap <= 12:
        raise ValueError("maximum temporal gap must be within [0, 12]")
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for item in detections:
        if type(item) is not Detection or item.frame_index >= frame_count:
            raise ValueError("detection lies outside the frame sequence")
        by_frame[item.frame_index].append(item)
    generated: list[Detection] = list(detections)
    for left_frame in sorted(by_frame):
        for left in by_frame[left_frame]:
            right: Detection | None = None
            right_frame = left_frame
            maximum_right = min(frame_count - 1, left_frame + max_gap + 1)
            for candidate_frame in range(left_frame + 1, maximum_right + 1):
                candidates = [
                    item
                    for item in by_frame.get(candidate_frame, ())
                    if item.privacy_class is left.privacy_class
                    and _can_associate(left.box, item.box)
                ]
                if candidates:
                    right = min(
                        candidates,
                        key=lambda item: (
                            -_iou(left.box, item.box),
                            math.dist(_center(left.box), _center(item.box)),
                            item.box.y1,
                            item.box.x1,
                            item.box.y2,
                            item.box.x2,
                        ),
                    )
                    right_frame = candidate_frame
                    break
            if right is None or right_frame == left_frame + 1:
                continue
            span = right_frame - left_frame
            for frame_index in range(left_frame + 1, right_frame):
                fraction = (frame_index - left_frame) / span
                linear = _linear_box(left.box, right.box, fraction)
                propagated = linear
                source = "temporal-linear"
                if flow is not None:
                    try:
                        propagated = _box_union(
                            linear,
                            flow.interpolate(left.box, right.box, fraction),
                        )
                        source = "temporal-flow-union"
                    except Exception:
                        propagated = linear
                generated.append(
                    Detection(
                        frame_index,
                        left.privacy_class,
                        min(left.confidence, right.confidence),
                        propagated,
                        source,
                    )
                )
    unique: dict[tuple[object, ...], Detection] = {}
    for item in generated:
        key = (
            item.frame_index,
            item.privacy_class.value,
            item.box.x1,
            item.box.y1,
            item.box.x2,
            item.box.y2,
        )
        current = unique.get(key)
        if current is None or item.confidence > current.confidence:
            unique[key] = item
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.frame_index,
                item.privacy_class.value,
                item.box.y1,
                item.box.x1,
                item.box.y2,
                item.box.x2,
                -item.confidence,
            ),
        )
    )


def expanded_box(
    box: PixelBox,
    privacy_class: PrivacyClass,
    width: int,
    height: int,
    config: PrivacyConfig,
) -> PixelBox:
    """Expand a privacy region about its center and clip it to the image."""

    if type(privacy_class) is not PrivacyClass or type(config) is not PrivacyConfig:
        raise ValueError("expansion requires canonical privacy contracts")
    scale = (
        config.face_scale
        if privacy_class is PrivacyClass.FACE
        else config.plate_scale
    )
    center_x, center_y = _center(box)
    half_width = (box.x2 - box.x1) * scale / 2.0
    half_height = (box.y2 - box.y1) * scale / 2.0
    return PixelBox(
        math.floor(center_x - half_width),
        math.floor(center_y - half_height),
        math.ceil(center_x + half_width),
        math.ceil(center_y + half_height),
    ).clip(width, height)


def _odd_kernel(region_dimension: int, minimum: int, divisor: int) -> int:
    value = max(minimum, math.ceil(region_dimension / divisor))
    return value if value % 2 == 1 else value + 1


def apply_strong_blur(
    frame_bgr: np.ndarray,
    boxes: Sequence[PixelBox],
    *,
    minimum_kernel: int,
    divisor: int,
) -> tuple[np.ndarray, BlurAudit]:
    """Blur only the supplied boxes, leaving the caller's frame unchanged."""

    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
    ):
        raise ValueError("blur input must be an HxWx3 uint8 BGR image")
    if (
        type(minimum_kernel) is not int
        or minimum_kernel < 31
        or minimum_kernel % 2 != 1
    ):
        raise ValueError("minimum blur kernel must be an odd integer of at least 31")
    if type(divisor) is not int or divisor <= 0:
        raise ValueError("blur divisor must be a positive integer")
    height, width = frame_bgr.shape[:2]
    output = frame_bgr.copy()
    maximum_width = 0
    maximum_height = 0
    canonical_boxes = []
    for box in boxes:
        if type(box) is not PixelBox:
            raise ValueError("blur boxes must be PixelBox values")
        clipped = box.clip(width, height)
        region_width = clipped.x2 - clipped.x1
        region_height = clipped.y2 - clipped.y1
        kernel_width = _odd_kernel(region_width, minimum_kernel, divisor)
        kernel_height = _odd_kernel(region_height, minimum_kernel, divisor)
        roi = output[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
        output[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2] = cv2.GaussianBlur(
            roi,
            (kernel_width, kernel_height),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_REPLICATE,
        )
        maximum_width = max(maximum_width, kernel_width)
        maximum_height = max(maximum_height, kernel_height)
        canonical_boxes.append((clipped.x1, clipped.y1, clipped.x2, clipped.y2))
    payload = json.dumps(
        sorted(canonical_boxes), separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return output, BlurAudit(
        len(canonical_boxes),
        maximum_width,
        maximum_height,
        hashlib.sha256(payload).hexdigest(),
    )
