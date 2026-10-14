"""Immutable contracts for pretrained video anonymization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .safe_fs import fingerprint_regular_fd, open_regular_read

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_REPOSITORY = "varungupta31/dashcam_anonymizer"
_MODEL_CLASSES = ("face", "license_plate")
_MODEL_LICENSE = "MIT"


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_int(value: int, label: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PrivacyClass(StrEnum):
    """The only identifier classes the approved checkpoint may emit."""

    FACE = "face"
    LICENSE_PLATE = "license_plate"


@dataclass(frozen=True, slots=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if any(type(value) is not int for value in coordinates):
            raise ValueError("pixel coordinates must be exact integers")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("pixel box cannot be empty")

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def clip(self, width: int, height: int) -> PixelBox:
        _require_int(width, "image width", minimum=1)
        _require_int(height, "image height", minimum=1)
        clipped = PixelBox(
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height)),
            max(0, min(self.x2, width)),
            max(0, min(self.y2, height)),
        )
        if clipped.x2 <= clipped.x1 or clipped.y2 <= clipped.y1:
            raise ValueError("pixel box becomes empty after clipping")
        return clipped


@dataclass(frozen=True, slots=True)
class Detection:
    frame_index: int
    privacy_class: PrivacyClass
    confidence: float
    box: PixelBox
    source: str

    def __post_init__(self) -> None:
        _require_int(self.frame_index, "frame index")
        if type(self.privacy_class) is not PrivacyClass:
            raise ValueError("detection class must be a PrivacyClass")
        if (
            type(self.confidence) not in {int, float}
            or isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("detection confidence must be finite and within [0, 1]")
        if type(self.box) is not PixelBox:
            raise ValueError("detection box must be a PixelBox")
        if type(self.source) is not str or not self.source or len(self.source) > 128:
            raise ValueError("detection source must be a bounded non-empty string")


@dataclass(frozen=True, slots=True)
class ModelLock:
    repository: str
    revision: str
    checkpoint_sha256: str
    classes: tuple[str, ...]
    license_id: str

    def __post_init__(self) -> None:
        if self.repository != _MODEL_REPOSITORY:
            raise ValueError("model repository differs from the approved checkpoint")
        if type(self.revision) is not str or _COMMIT.fullmatch(self.revision) is None:
            raise ValueError("model revision must be an immutable Git commit")
        _require_sha256(self.checkpoint_sha256, "checkpoint digest")
        if type(self.classes) is not tuple or self.classes != _MODEL_CLASSES:
            raise ValueError("model class map must be exactly face and license_plate")
        if self.license_id != _MODEL_LICENSE:
            raise ValueError("model license must retain the approved MIT identifier")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    confidence: float
    confidence_grid: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
    tile_edge: int = 1280
    tile_overlap_ratio: float = 0.20
    temporal_gap_frames: int = 12
    face_scale: float = 1.50
    plate_scale: float = 1.35
    blur_divisor: int = 3
    minimum_blur_kernel: int = 31
    minimum_plate_confidence: float = 0.25
    encoder: str = "hevc_nvenc"
    encoder_preset: str = "p7"
    encoder_pixel_format: str = "yuv420p"
    encoder_cq: int = 18
    minimum_unmasked_psnr_db: float = 34.5
    minimum_unmasked_ssim: float = 0.98
    minimum_size_ratio: float = 0.80
    maximum_size_ratio: float = 1.30
    maximum_encode_attempts: int = 5

    def __post_init__(self) -> None:
        if self.confidence_grid != (0.10, 0.15, 0.20, 0.25):
            raise ValueError("confidence grid differs from the approved contract")
        numeric = (
            self.confidence,
            self.tile_overlap_ratio,
            self.face_scale,
            self.plate_scale,
            self.minimum_plate_confidence,
            self.minimum_unmasked_psnr_db,
            self.minimum_unmasked_ssim,
            self.minimum_size_ratio,
            self.maximum_size_ratio,
        )
        if any(
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("privacy numeric values must be finite real numbers")
        if self.confidence not in self.confidence_grid:
            raise ValueError("confidence must be selected from the approved grid")
        if not 0.0 < self.tile_overlap_ratio < 0.5:
            raise ValueError("tile overlap must be between zero and one half")
        _require_int(self.tile_edge, "tile edge", minimum=128)
        _require_int(self.temporal_gap_frames, "temporal gap")
        if self.temporal_gap_frames > 12:
            raise ValueError("temporal gap cannot exceed twelve frames")
        if not 1.0 < self.face_scale <= 3.0 or not 1.0 < self.plate_scale <= 3.0:
            raise ValueError("privacy expansion scales must be within (1, 3]")
        _require_int(self.blur_divisor, "blur divisor", minimum=1)
        _require_int(self.minimum_blur_kernel, "minimum blur kernel", minimum=31)
        if self.minimum_blur_kernel % 2 != 1:
            raise ValueError("minimum blur kernel must be odd")
        if (
            self.minimum_plate_confidence not in self.confidence_grid
            or self.minimum_plate_confidence < self.confidence
        ):
            raise ValueError(
                "minimum plate confidence must use the approved grid"
            )
        if self.encoder != "hevc_nvenc" or self.encoder_preset != "p7":
            raise ValueError("video encoder must be hevc_nvenc preset p7")
        if self.encoder_pixel_format not in {"yuv420p", "p010le"}:
            raise ValueError("video encoder pixel format is unsupported")
        _require_int(self.encoder_cq, "encoder CQ", minimum=1)
        if self.encoder_cq > 51:
            raise ValueError("encoder CQ cannot exceed 51")
        if self.minimum_unmasked_psnr_db <= 0:
            raise ValueError("minimum unmasked PSNR must be positive")
        if not 0.0 < self.minimum_unmasked_ssim <= 1.0:
            raise ValueError("minimum unmasked SSIM must be within (0, 1]")
        if not 0.0 < self.minimum_size_ratio <= 1.0:
            raise ValueError("minimum size ratio must be within (0, 1]")
        if not self.maximum_size_ratio >= 1.0:
            raise ValueError("maximum size ratio must be at least one")
        if self.minimum_size_ratio > self.maximum_size_ratio:
            raise ValueError("size ratio bounds are inverted")
        _require_int(self.maximum_encode_attempts, "maximum encode attempts", minimum=1)
        if self.maximum_encode_attempts > 8:
            raise ValueError("maximum encode attempts cannot exceed eight")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MediaProbe:
    width: int
    height: int
    display_aspect_num: int
    display_aspect_den: int
    frame_count: int
    relative_pts_ns: tuple[int, ...]
    duration_ns: int
    codec: str
    codec_tag: str
    byte_size: int = 1
    measured_bitrate_bps: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("width", self.width),
            ("height", self.height),
            ("display aspect numerator", self.display_aspect_num),
            ("display aspect denominator", self.display_aspect_den),
            ("frame count", self.frame_count),
            ("duration", self.duration_ns),
        ):
            _require_int(value, label, minimum=1)
        if (
            type(self.relative_pts_ns) is not tuple
            or len(self.relative_pts_ns) != self.frame_count
        ):
            raise ValueError("relative PTS count must equal frame count")
        if any(type(value) is not int or value < 0 for value in self.relative_pts_ns):
            raise ValueError("relative PTS values must be non-negative integers")
        if not self.relative_pts_ns or self.relative_pts_ns[0] != 0:
            raise ValueError("relative PTS must start at zero")
        if any(
            current <= previous
            for previous, current in zip(
                self.relative_pts_ns, self.relative_pts_ns[1:], strict=False
            )
        ):
            raise ValueError("relative PTS values must be strictly increasing")
        if self.duration_ns <= self.relative_pts_ns[-1]:
            raise ValueError("media duration must extend beyond the final PTS")
        if self.codec != "hevc" or self.codec_tag not in {"hvc1", "hev1"}:
            raise ValueError("media must be HEVC with an hvc1 or hev1 sample entry")
        _require_int(self.byte_size, "media byte size", minimum=1)
        _require_int(self.measured_bitrate_bps, "measured bitrate", minimum=1)


@dataclass(frozen=True, slots=True)
class FrameAccounting:
    frame_index: int
    final_boxes_sha256: str
    face_count: int
    plate_count: int

    def __post_init__(self) -> None:
        _require_int(self.frame_index, "frame index")
        _require_sha256(self.final_boxes_sha256, "final box-list digest")
        _require_int(self.face_count, "face count")
        _require_int(self.plate_count, "plate count")


@dataclass(frozen=True, slots=True)
class PrivacyVideoReceipt:
    input_sha256: str
    output_sha256: str
    source_probe: MediaProbe
    output_probe: MediaProbe
    model_lock_sha256: str
    privacy_config_sha256: str
    frame_accounting: tuple[FrameAccounting, ...]
    face_detections: int
    plate_detections: int
    masked_frames: int
    skipped_frames: int
    source_bytes: int = 1
    output_bytes: int = 1
    size_ratio: float = 1.0
    target_bitrate_bps: int = 1
    encode_attempts: int = 1
    unmasked_psnr_db: float = 99.0
    unmasked_ssim: float = 1.0
    changed_outside_mask_pixels: int = 0

    def __post_init__(self) -> None:
        for digest_label, digest in (
            ("input digest", self.input_sha256),
            ("output digest", self.output_sha256),
            ("model lock digest", self.model_lock_sha256),
            ("privacy config digest", self.privacy_config_sha256),
        ):
            _require_sha256(digest, digest_label)
        if (
            type(self.source_probe) is not MediaProbe
            or type(self.output_probe) is not MediaProbe
        ):
            raise ValueError("privacy receipt probes must be MediaProbe values")
        if (
            self.source_probe.width,
            self.source_probe.height,
            self.source_probe.display_aspect_num,
            self.source_probe.display_aspect_den,
            self.source_probe.frame_count,
            self.source_probe.relative_pts_ns,
        ) != (
            self.output_probe.width,
            self.output_probe.height,
            self.output_probe.display_aspect_num,
            self.output_probe.display_aspect_den,
            self.output_probe.frame_count,
            self.output_probe.relative_pts_ns,
        ):
            raise ValueError("output media differs from source timing or geometry")
        if (
            type(self.frame_accounting) is not tuple
            or len(self.frame_accounting) != self.source_probe.frame_count
        ):
            raise ValueError("one frame-accounting row is required per source frame")
        if tuple(row.frame_index for row in self.frame_accounting) != tuple(
            range(self.source_probe.frame_count)
        ):
            raise ValueError("frame-accounting indices must be contiguous")
        for row in self.frame_accounting:
            if type(row) is not FrameAccounting:
                raise ValueError("frame accounting contains an invalid row")
        for count_label, count in (
            ("face detections", self.face_detections),
            ("plate detections", self.plate_detections),
            ("masked frames", self.masked_frames),
            ("skipped frames", self.skipped_frames),
        ):
            _require_int(count, count_label)
        if self.skipped_frames != 0:
            raise ValueError("skipped frames are forbidden")
        if self.face_detections != sum(row.face_count for row in self.frame_accounting):
            raise ValueError("face detection total differs from frame accounting")
        if self.plate_detections != sum(
            row.plate_count for row in self.frame_accounting
        ):
            raise ValueError("plate detection total differs from frame accounting")
        expected_masked = sum(
            row.face_count + row.plate_count > 0 for row in self.frame_accounting
        )
        if self.masked_frames != expected_masked:
            raise ValueError("masked frame total differs from frame accounting")
        _require_int(self.source_bytes, "receipt source bytes", minimum=1)
        _require_int(self.output_bytes, "receipt output bytes", minimum=1)
        _require_int(self.target_bitrate_bps, "receipt target bitrate", minimum=1)
        _require_int(self.encode_attempts, "receipt encode attempts", minimum=1)
        _require_int(
            self.changed_outside_mask_pixels,
            "changed outside-mask pixels",
        )
        if self.source_bytes != self.source_probe.byte_size:
            raise ValueError("receipt source bytes differ from source probe")
        if self.output_bytes != self.output_probe.byte_size:
            raise ValueError("receipt output bytes differ from output probe")
        expected_ratio = self.output_bytes / self.source_bytes
        if (
            type(self.size_ratio) not in {int, float}
            or not math.isfinite(self.size_ratio)
            or not math.isclose(
                self.size_ratio, expected_ratio, rel_tol=0, abs_tol=1e-12
            )
        ):
            raise ValueError("receipt size ratio differs from byte sizes")
        if (
            type(self.unmasked_psnr_db) not in {int, float}
            or not math.isfinite(self.unmasked_psnr_db)
            or self.unmasked_psnr_db <= 0
        ):
            raise ValueError("receipt unmasked PSNR must be finite and positive")
        if (
            type(self.unmasked_ssim) not in {int, float}
            or not math.isfinite(self.unmasked_ssim)
            or not 0 <= self.unmasked_ssim <= 1
        ):
            raise ValueError("receipt unmasked SSIM must be within [0, 1]")

    @property
    def frame_accounting_sha256(self) -> str:
        return _canonical_sha256([asdict(row) for row in self.frame_accounting])

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def sha256_file(path: Path) -> str:
    """Hash one stable, canonical, single-link regular file."""

    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("file path must be canonical and absolute")
    descriptor = open_regular_read(path)
    try:
        expected = fingerprint_regular_fd(descriptor)
    finally:
        os.close(descriptor)
    current = open_regular_read(path)
    try:
        if fingerprint_regular_fd(current) != expected:
            raise OSError("file path changed while hashing")
    finally:
        os.close(current)
    return expected.sha256
