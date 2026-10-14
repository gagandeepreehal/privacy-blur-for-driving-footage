"""Upload a private model repository and create an authenticated Endpoint."""
import argparse
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--name', default='face-plate-blurring')
    parser.add_argument('--vendor', default='aws')
    parser.add_argument('--region', required=True)
    parser.add_argument('--instance-type', required=True)
    parser.add_argument('--instance-size', required=True)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    args = parser.parse_args()
    api = HfApi()
    namespace = args.repo.split('/')[0]
    api.list_inference_endpoints(namespace=namespace)
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for item in source.iterdir():
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            if item.is_dir():
                shutil.copytree(item, staging / item.name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy2(item, staging / item.name)
        for name in ['best.pt', 'face_detection_yunet_2023mar.onnx', 'YUNET_LICENSE']:
            shutil.copy2(args.models / name, staging / name)
        api.create_repo(args.repo, repo_type='model', private=True, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, folder_path=staging, commit_message='Add face and plate blurring Endpoint')
    endpoint = api.create_inference_endpoint(name=args.name, namespace=namespace,
        repository=args.repo, revision=commit.oid, framework='pytorch', task='custom',
        accelerator=args.accelerator, vendor=args.vendor, region=args.region,
        instance_type=args.instance_type, instance_size=args.instance_size,
        type='authenticated', min_replica=0, max_replica=1, scale_to_zero_timeout=15)
    print('Endpoint requested:', endpoint.name, endpoint.status)
    print('Model revision:', commit.oid)
