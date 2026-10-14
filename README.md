---
tags:
- endpoints-template
- object-detection
- privacy
pipeline_tag: object-detection
---

# Privacy Blur for Driving Footage

Custom Hugging Face Inference Endpoint handler using the recovered Dashcam Anonymizer YOLOv8 checkpoint and OpenCV YuNet. Accepts a base64 image in `inputs`; returns bounding boxes, confidence scores, and a base64 PNG in `blurred_image`. No OCR or identity recognition. Input and output images are not written to disk by the handler.

This is an image API. Video tracking and temporal interpolation from the original dataset pipeline are not included. The copied pipeline retains tiled detection, plausibility filters, class-aware suppression, YuNet confirmation, expanded privacy boxes, and strong Gaussian blur. The adapter selects CUDA when available and CPU otherwise.

The recovered checkpoint is SHA-256 `c941613a63a2e54f1b4f4f1bc7dba3d2b42f905daf4513b3aeaed25b02a72236`. Its original production checksum is unavailable, so equivalence with the deleted server's exact checkpoint is not established. `model-lock.json` records the upstream source revision observed during recovery, not a claim that the separately hosted Google Drive binary is Git-versioned.

Weights are intentionally excluded from GitHub. Supply `best.pt`, `face_detection_yunet_2023mar.onnx`, and `YUNET_LICENSE` from the recovered local model folder when deploying.

## Deployment

Install `huggingface_hub` in the deployment environment and authenticate using a token with repository write access and Inference Endpoint read/manage permissions. Run `deploy.py --help` for required repository, model folder, region, and instance options. Hardware should be selected against the current HF availability and pricing before provisioning. The script creates a private model repository and a authenticated (HF token required) Endpoint, pins the uploaded revision, limits replicas to one, and requests scale-to-zero after 15 minutes. Provisioning incurs HF charges.

## Attribution and limits

YOLO checkpoint source: https://github.com/varungupta31/dashcam_anonymizer (upstream MIT license included in UPSTREAM_LICENSE). OpenCV YuNet license is included alongside its weights. Ultralytics dependencies carry their own license obligations; the upstream checkpoint's license does not replace dependency licenses.

Predictions are model-generated pseudo-labels. Missed detections remain possible, and this package does not certify anonymization completeness. Validate face and plate coverage on representative data before bulk use. The retained plausibility filters may exclude unusually large identifiers.

## Live deployment

Endpoint: https://6acfc57b9405b8c0d0c71c96.endpoints.huggingface.cloud

Private model repository: https://huggingface.co/gagandeepreehal/privacy-blur-for-driving-footage

Use `python client.py image.jpg --output blurred.png` after local HF login. Cold starts after scale-to-zero can require a retry. Local model loading, blank-image handling, invalid base64 rejection, and live face detection with a blurred PNG response passed. Plate accuracy has not been benchmarked. Deployment details are recorded in `deployment.json`.
