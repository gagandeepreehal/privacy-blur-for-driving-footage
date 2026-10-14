---
tags:
- endpoints-template
- object-detection
- privacy
pipeline_tag: object-detection
---

# Privacy Blur for Driving Footage

Custom Hugging Face Inference Endpoint handler using the recovered Dashcam Anonymizer YOLOv8 checkpoint and OpenCV YuNet. Accepts base64 images or short video clips in `inputs`; returns detections and a blurred PNG or MP4. No OCR or identity recognition. Input and output images are not written to disk by the handler.

This API supports images and short video clips. Video inference processes every frame, then fills matching same-class detection gaps using the original pipeline’s frame-based linear interpolation (up to 12 missing frames by default). Interpolation requires detections before and after the gap; it does not fill gaps at clip boundaries or implement optical flow. The pipeline retains tiled detection, plausibility filters, class-aware suppression, YuNet confirmation, expanded privacy boxes, and strong Gaussian blur. The adapter selects CUDA when available and CPU otherwise.

The recovered checkpoint is SHA-256 `c941613a63a2e54f1b4f4f1bc7dba3d2b42f905daf4513b3aeaed25b02a72236`. Its original production checksum is unavailable, so equivalence with the deleted server's exact checkpoint is not established. `model-lock.json` records the upstream source revision observed during recovery, not a claim that the separately hosted Google Drive binary is Git-versioned.

Weights are intentionally excluded from GitHub. Supply `best.pt`, `face_detection_yunet_2023mar.onnx`, and `YUNET_LICENSE` from the recovered local model folder when deploying.

## Deployment

Install `huggingface_hub` in the deployment environment and authenticate using a token with repository write access and Inference Endpoint read/manage permissions. Run `deploy.py --help` for required repository, model folder, region, and instance options. Hardware should be selected against the current HF availability and pricing before provisioning. The script creates a private model repository and a authenticated (HF token required) Endpoint, pins the uploaded revision, limits replicas to one, and requests scale-to-zero after 15 minutes. Provisioning incurs HF charges.

## Attribution and limits

YOLO checkpoint source: https://github.com/varungupta31/dashcam_anonymizer (upstream MIT license included in UPSTREAM_LICENSE). OpenCV YuNet license is included alongside its weights. Ultralytics dependencies carry their own license obligations; the upstream checkpoint's license does not replace dependency licenses.

Predictions are model-generated pseudo-labels. Missed detections remain possible, and this package does not certify anonymization completeness. Validate face and plate coverage on representative data before bulk use. The retained plausibility filters may exclude unusually large identifiers.

## Live deployment

Endpoint: https://6acfc57b9405b8c0d0c71c96.endpoints.huggingface.cloud

Public model repository: https://huggingface.co/gagandeepreehal/privacy-blur-for-driving-footage

Use `python client.py image.jpg --output blurred.png` after local HF login. Cold starts after scale-to-zero can require a retry. Local model loading, blank-image handling, invalid base64 rejection, and live face detection with a blurred PNG response passed. Plate accuracy has not been benchmarked. Deployment details are recorded in `deployment.json`.

## Video interpolation

Run `python client.py clip.mp4 --video --output blurred.mp4`. Set `--max-gap-frames 0` to disable interpolation, or choose 1–12. The API JSON uses `media_type: "video"`, base64 video bytes in `inputs`, and optional `max_gap_frames`. Responses include `blurred_video` (base64 H.264 MP4), per-frame boxes and their sources, and `interpolated_detections`.

Requests are limited to 20 MB, 300 frames, one video stream, and even dimensions with at most 1920×1080 pixels. Processing uses two decode passes with temporary input/output files, deleted after the request. Frame count and relative presentation timestamps are preserved; the original codec and audio are not retained. Outputs omit audio and source metadata. Long recordings should be split into clips; interpolation does not cross clip boundaries.

Validation: tests cover face and plate gap interpolation, disabled interpolation, invalid gap values, and variable-frame-rate timestamp preservation. Real-model video and image compatibility smoke tests also pass. Live Endpoint tests verified MP4 output, a missing-frame face interpolation, and relative output timestamps. These checks do not establish detection recall on driving footage.
