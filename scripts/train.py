#!/usr/bin/env python3
"""Script to train the model."""

from lenta_price_tag_scanner.train_oof import main as train


def main() -> None:
    """Run training."""
    train()


if __name__ == "__main__":
    main()
