"""CPU YOLO MLflow adapter with image bytes as the request boundary."""

import mlflow.pyfunc


class PriceTagDetector(mlflow.pyfunc.PythonModel):
    def __init__(self, imgsz=1280, conf=0.25, iou=0.5):
        self.options = dict(imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)

    def load_context(self, context):
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        import base64
        import io
        import json

        import pandas as pd
        from PIL import Image

        if list(model_input.columns) != ["image_base64"]:
            raise ValueError("Expected only image_base64 column")
        rows = []
        for encoded in model_input["image_base64"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                result = self.model.predict(image.convert("RGB"), **self.options)[0]
            detections = [
                dict(xyxy=box[:4], confidence=box[4], class_id=int(box[5]))
                for box in result.boxes.data.cpu().tolist()
            ]
            rows.append(
                dict(detections_json=json.dumps(detections), num_detections=len(detections))
            )
        return pd.DataFrame(rows)
