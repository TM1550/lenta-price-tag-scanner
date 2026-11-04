"""Load the explicit RGB classifier contract used by export and serving."""

from .constants import IMAGENET_MEAN, IMAGENET_STD


def load_classifier(checkpoint_path):
    import timm
    import torch
    from torchvision import transforms

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("color") != "RGB" or checkpoint.get("positive_class") != 1:
        raise ValueError("Expected a trained RGB/positive_class=1 checkpoint")
    model = timm.create_model(checkpoint["architecture"], pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    size = checkpoint["size"]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return model, transform, size
