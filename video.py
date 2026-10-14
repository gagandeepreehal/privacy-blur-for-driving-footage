"""Bounded two-pass video detection, interpolation, and H.264 export."""
import base64
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import av
from privacy_pipeline.privacy_mask import propagate_short_gaps, expanded_box, apply_strong_blur

MAX_FRAMES = 300
MAX_PIXELS = 1920 * 1080
MAX_OUTPUT_BYTES = 20_000_000

def process_video(raw, detector, config, max_gap=12):
    if type(max_gap) is not int or not 0 <= max_gap <= 12:
        raise ValueError('max_gap_frames must be an integer from 0 to 12')
    with tempfile.TemporaryDirectory(prefix='privacy-video-') as folder:
        source = Path(folder) / 'input'; target = Path(folder) / 'blurred.mp4'
        source.write_bytes(raw)
        detections, times = [], []
        geometry = None
        with av.open(str(source)) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Supply a video with exactly one video stream')
            stream = container.streams.video[0]
            rate = stream.average_rate or Fraction(30)
            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_FRAMES:
                    raise ValueError('Video exceeds 300 frames; split into shorter clips')
                current = (frame.width, frame.height)
                if frame.width * frame.height > MAX_PIXELS or frame.width % 2 or frame.height % 2:
                    raise ValueError('Video requires even dimensions and at most 1920x1080 pixels')
                if geometry is None:
                    geometry = current
                elif geometry != current:
                    raise ValueError('Changing video dimensions are unsupported')
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Video frame timestamps are required')
                timestamp = frame.pts * frame.time_base
                if times and timestamp <= times[-1]:
                    raise ValueError('Video timestamps must increase strictly')
                times.append(timestamp)
                detections.extend(detector.detect(frame.to_ndarray(format='bgr24'), frame_index=index))
        if not times:
            raise ValueError('Video contains no decodable frames')
        propagated = propagate_short_gaps(detections, frame_count=len(times), max_gap=max_gap, flow=None)
        by_frame = defaultdict(list)
        for item in propagated:
            by_frame[item.frame_index].append(item)
        summaries = []
        # Keep exact rational timestamp intervals when exporting VFR clips.
        import math
        denominator = 1
        for time in times:
            denominator = math.lcm(denominator, (time - times[0]).denominator)
        if denominator > 2_147_483_647:
            raise ValueError('Video timestamp precision is unsupported')
        time_base = Fraction(1, denominator)
        with av.open(str(source)) as input_container, av.open(str(target), mode='w') as output:
            encoder = output.add_stream('libx264', rate=rate)
            encoder.width, encoder.height = geometry
            encoder.pix_fmt = 'yuv420p'
            encoder.time_base = time_base
            encoder.codec_context.time_base = time_base
            encoder.options = {'crf': '18', 'preset': 'fast'}
            for index, frame in enumerate(input_container.decode(video=0)):
                rows = by_frame[index]
                regions = [expanded_box(row.box, row.privacy_class, *geometry, config) for row in rows]
                blurred, _ = apply_strong_blur(frame.to_ndarray(format='bgr24'), regions,
                    minimum_kernel=config.minimum_blur_kernel, divisor=config.blur_divisor)
                result = av.VideoFrame.from_ndarray(blurred, format='bgr24')
                result.pts = int((times[index] - times[0]) / time_base)
                result.time_base = time_base
                for packet in encoder.encode(result):
                    output.mux(packet)
                summaries.append({'frame_index': index, 'timestamp_seconds': float(times[index] - times[0]),
                    'detections': [{'label': row.privacy_class.value, 'score': row.confidence,
                        'source': row.source, 'box': [row.box.x1,row.box.y1,row.box.x2,row.box.y2],
                        'blur_box': [box.x1,box.y1,box.x2,box.y2]} for row,box in zip(rows,regions)]})
            for packet in encoder.encode():
                output.mux(packet)
        if target.stat().st_size > MAX_OUTPUT_BYTES:
            raise ValueError('Output exceeds 20 MB; split into shorter clips')
        interpolated = sum(row.source.startswith('temporal-') for row in propagated)
        return {'blurred_video': base64.b64encode(target.read_bytes()).decode(), 'video_format': 'mp4',
            'width': geometry[0], 'height': geometry[1], 'frame_count': len(times),
            'interpolated_detections': interpolated, 'max_gap_frames': max_gap,
            'audio_included': False, 'frames': summaries}
