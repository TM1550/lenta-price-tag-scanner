"""Download model and pinned custom-code snapshot for inspection before execution."""

from huggingface_hub import snapshot_download


def main():
    print(
        snapshot_download(
            repo_id="PaddlePaddle/PaddleOCR-VL",
            revision="7fa00a8c55b735ba51ba49a9058f3f9c57a99a11",
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            max_workers=2,
        )
    )


if __name__ == "__main__":
    main()
