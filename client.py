"""Send an image to the deployed Endpoint using the local HF login."""
import argparse
import base64
import json
import urllib.request
from pathlib import Path
from huggingface_hub import get_token

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('blurred.png'))
    parser.add_argument('--url', default='https://6acfc57b9405b8c0d0c71c96.endpoints.huggingface.cloud')
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit('Log in to Hugging Face first.')
    body = json.dumps({'inputs': base64.b64encode(args.image.read_bytes()).decode()}).encode()
    request = urllib.request.Request(args.url, data=body, headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    args.output.write_bytes(base64.b64decode(result.pop('blurred_image')))
    print(json.dumps(result, indent=2))
    print('Saved:', args.output)
