"""MLflow crop-classifier adapter; HTTP requests contain image bytes, not server paths."""

import mlflow.pyfunc


class CropClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from .classifier import load_classifier

        self.model, self.transform, self.size = load_classifier(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        import base64
        import io

        import pandas as pd
        import torch
        from PIL import Image

        torch.set_num_threads(4)
        if list(model_input.columns) != ["image_base64"]:
            raise ValueError("Expected a dataframe with one image_base64 column")
        probabilities = []
        for encoded in model_input["image_base64"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                pixels = self.transform(image.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                probabilities.append(float(torch.sigmoid(self.model(pixels)).item()))
        return pd.DataFrame({"probability": probabilities})
