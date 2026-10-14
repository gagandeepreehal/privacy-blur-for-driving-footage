"""Image and video custom Hugging Face Inference Endpoint handler."""
import base64
import io
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from privacy_pipeline.privacy_models import ModelLock, PrivacyConfig
from privacy_pipeline.privacy_detector import PretrainedYoloDetector, UnionPrivacyDetector, load_pretrained_yunet
from privacy_pipeline.privacy_mask import expanded_box, apply_strong_blur

class EndpointHandler:
    def __init__(self, path=''):
        root = Path(path).resolve()
        metadata = json.loads((root / 'model-lock.json').read_text())
        metadata.pop('note', None)
        metadata['classes'] = tuple(metadata['classes'])
        self.config = PrivacyConfig(confidence=0.15)
        primary = PretrainedYoloDetector(root / 'best.pt', ModelLock(**metadata), self.config)
        yunet = load_pretrained_yunet(root / 'face_detection_yunet_2023mar.onnx', root / 'YUNET_LICENSE')
        self.detector = UnionPrivacyDetector(primary, yunet, strong_face_confidence=0.25)
        self.lock = threading.Lock()

    def __call__(self, data):
        encoded = data.get('inputs')
        if not isinstance(encoded, str) or len(encoded) > 28_000_000:
            raise ValueError('inputs must be a base64 media of at most 20 MB')
        if encoded.startswith(('data:image/', 'data:video/')):
            encoded = encoded.split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 20_000_000:
            raise ValueError('Media exceeds 20 MB')
        media_type = data.get('media_type', 'image')
        if media_type == 'video':
            from video import process_video
            with self.lock:
                return process_video(raw, self.detector, self.config, data.get('max_gap_frames', 12))
        if media_type != 'image':
            raise ValueError('media_type must be image or video')
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError('Image exceeds 20 megapixels')
            rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self.lock:
            detections = self.detector.detect(frame, frame_index=0)
        height, width = frame.shape[:2]
        regions = [expanded_box(d.box, d.privacy_class, width, height, self.config) for d in detections]
        blurred, _ = apply_strong_blur(frame, regions, minimum_kernel=self.config.minimum_blur_kernel, divisor=self.config.blur_divisor)
        success, png = cv2.imencode('.png', blurred)
        if not success:
            raise RuntimeError('Image encoding failed')
        return {'width': width, 'height': height, 'detections': [
            {'label': d.privacy_class.value, 'score': d.confidence,
             'box': [d.box.x1, d.box.y1, d.box.x2, d.box.y2],
             'blur_box': [b.x1, b.y1, b.x2, b.y2]} for d, b in zip(detections, regions)],
            'blurred_image': base64.b64encode(png.tobytes()).decode('ascii'), 'image_format': 'png'}
