"""Compatibility defaults for image heuristics; Hydra profiles expose tunable values."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METRIC_EPSILON = 1e-12

CROP_QUALITY_DEFAULTS = dict(short_side=160, white_threshold=250, black_threshold=5)
PRICE_LAYOUT_DEFAULTS = dict(
    min_height_fraction=0.12,
    min_top_fraction=0.40,
    min_center_x_fraction=0.25,
    max_cents_height_ratio=0.6,
    cents_top_margin_ratio=0.2,
    cents_bottom_margin_ratio=0.6,
)
EXTRACTION_DEFAULTS = dict(
    name_fraction=0.43,
    max_name_lines=5,
    saturation_threshold=70,
    brightness_threshold=70,
    red_hue_low=15,
    red_hue_high=165,
    yellow_hue_high=40,
    min_color_fraction=0.20,
)
IMAGE_PROCESSING_DEFAULTS = dict(
    canny_lower=60,
    canny_upper=150,
    closing_kernel=3,
    contour_limit=8,
    polygon_epsilon=0.025,
    min_area_fraction=0.65,
    max_area_fraction=0.99,
    min_rectified_side=24,
    clahe_clip=2.0,
    clahe_grid=4,
    sharpen_amount=0.5,
    sharpen_sigma=1.0,
    row_height_fraction=0.6,
    repetition_penalty=1.05,
    no_repeat_ngram_size=6,
)
