import base64
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
from video import process_video
from privacy_pipeline.privacy_models import Detection, PixelBox, PrivacyClass, PrivacyConfig

class Detector:
    def detect(self, image, *, frame_index):
        if frame_index in (0, 2):
            return (Detection(frame_index, PrivacyClass.FACE, .9, PixelBox(10+frame_index*4,10,30+frame_index*4,30),'fixture'),
                    Detection(frame_index, PrivacyClass.LICENSE_PLATE, .9, PixelBox(70,40,100,50),'fixture'))
        return ()

def clip():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'input.mp4'
        with av.open(str(path),'w') as out:
            stream=out.add_stream('libx264',rate=10); stream.width=128; stream.height=64; stream.pix_fmt='yuv420p'
            stream.codec_context.time_base=Fraction(1,10)
            for index, pts in enumerate([0,1,3]):
                image=np.random.default_rng(index).integers(0,256,(64,128,3),dtype=np.uint8)
                frame=av.VideoFrame.from_ndarray(image,format='bgr24'); frame.pts=pts; frame.time_base=Fraction(1,10)
                for packet in stream.encode(frame): out.mux(packet)
            for packet in stream.encode(): out.mux(packet)
        return path.read_bytes()

class VideoTests(unittest.TestCase):
    def test_interpolates_both_classes_and_preserves_vfr_timestamps(self):
        raw=clip(); result=process_video(raw,Detector(),PrivacyConfig(confidence=.15))
        self.assertEqual(result['frame_count'],3)
        self.assertEqual(result['interpolated_detections'],2)
        rows=result['frames'][1]['detections']
        self.assertEqual({row['label'] for row in rows},{'face','license_plate'})
        self.assertTrue(all(row['source']=='temporal-linear' for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'output.mp4'; p.write_bytes(base64.b64decode(result['blurred_video']))
            with av.open(str(p)) as output:
                frames=list(output.decode(video=0))
                self.assertEqual(len(frames),3)
                times=[frame.pts*frame.time_base for frame in frames]
                self.assertEqual(times,[Fraction(0),Fraction(1,10),Fraction(3,10)])
                self.assertFalse(output.streams.audio)
    def test_disabled_gap_keeps_missing_frame_unmasked(self):
        result=process_video(clip(),Detector(),PrivacyConfig(confidence=.15),max_gap=0)
        self.assertEqual(result['interpolated_detections'],0)
        self.assertEqual(result['frames'][1]['detections'],[])
    def test_invalid_gap(self):
        for gap in [-1,13,True,1.5]:
            with self.assertRaises(ValueError): process_video(b'',Detector(),PrivacyConfig(confidence=.15),gap)

if __name__=='__main__': unittest.main()
