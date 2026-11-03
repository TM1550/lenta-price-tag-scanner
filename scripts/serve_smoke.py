"""Send the real DVC crop to a locally started MLflow scoring server."""

import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

import fire


def main(url="http://127.0.0.1:18081/invocations", image="data/inference_assets/example_crop.jpg"):
    encoded = base64.b64encode(Path(image).read_bytes()).decode()
    payload = {"dataframe_split": {"columns": ["image_base64"], "data": [[encoded]]}}
    request = Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    probability = float(result["predictions"][0]["probability"])
    if not 0 <= probability <= 1:
        raise ValueError("Invalid classifier output")
    print(json.dumps(result))


if __name__ == "__main__":
    fire.Fire(main)
