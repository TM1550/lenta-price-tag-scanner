"""Conservative image transforms and measurable crop quality, without generated pixels."""

import cv2
import numpy as np
from PIL import Image


def quality(image):
    pixels = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    # Fixed short side keeps sharpness scores comparable between crop sizes.
    ratio = 160 / min(gray.shape)
    normalized = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return dict(
        min_side=min(gray.shape),
        sharpness=float(cv2.Laplacian(normalized, cv2.CV_64F).var()),
        contrast=float(gray.std()),
        white_fraction=float((gray > 250).mean()),
        black_fraction=float((gray < 5).mean()),
    )


def rejection_reasons(values, cfg):
    reasons = []
    for field, threshold in [
        ("min_side", cfg.min_side),
        ("sharpness", cfg.min_sharpness),
        ("contrast", cfg.min_contrast),
    ]:
        if values[field] < threshold:
            reasons.append(field)
    if max(values["white_fraction"], values["black_fraction"]) > cfg.max_flat_fraction:
        reasons.append("exposure")
    return reasons


def rectify(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_pixels, width_pixels = gray.shape
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        poly = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        if not 0.65 <= cv2.contourArea(poly) / (height_pixels * width_pixels) <= 0.99:
            continue
        points = poly[:, 0].astype(np.float32)
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = np.array(
            [
                points[sums.argmin()],
                points[diffs.argmin()],
                points[sums.argmax()],
                points[diffs.argmax()],
            ]
        )
        if len(np.unique(ordered, axis=0)) < 4:
            continue
        width = round(
            max(np.linalg.norm(ordered[0] - ordered[1]), np.linalg.norm(ordered[2] - ordered[3]))
        )
        height = round(
            max(np.linalg.norm(ordered[0] - ordered[3]), np.linalg.norm(ordered[1] - ordered[2]))
        )
        if min(width, height) < 24:
            continue
        target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        return cv2.warpPerspective(
            pixels,
            cv2.getPerspectiveTransform(ordered, target),
            (width, height),
            flags=cv2.INTER_CUBIC,
        )
    return pixels


def preprocess(image, mode):
    if mode == "none":
        return image
    pixels = np.asarray(image.convert("RGB"))
    if mode in ("rectify", "rectify_clahe"):
        pixels = rectify(pixels)
    if mode in ("clahe", "rectify_clahe"):
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
        pixels = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mode == "sharpen":
        pixels = cv2.addWeighted(pixels, 1.5, cv2.GaussianBlur(pixels, (0, 0), 1), -0.5, 0)
    if mode not in {"rectify", "rectify_clahe", "clahe", "sharpen"}:
        raise ValueError(f"Unknown preprocessing: {mode}")
    return Image.fromarray(pixels)
