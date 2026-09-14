"""Load local native weights with the pinned SM120 adapter, without downloading."""
import json
import os
from pathlib import Path
import sys

import yaml


def main():
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    root = Path(config["model-path"])
    model_config = json.loads((root / "config.json").read_text())
    if model_config.get("model_type") != "deepseek_v41":
        raise ValueError("Expected native deepseek-ai/DeepSeek-V4.1-Flash")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(name for name in set(index.values()) if not (root / name).is_file())
    if missing:
        raise ValueError(f"Missing model shards: {missing}")
    # The adapter specifically requires native FP8 Engram rows and UE8M0 scales.
    for layer in (1, 14):
        for suffix in ("weight", "scale"):
            if f"layers.{layer}.engram.embed.{suffix}" not in index:
                raise ValueError("Native Engram tensors missing; NVFP4 repacks are not supported")
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key or api_key == "replace-with-a-strong-api-key":
        raise ValueError("Set a non-placeholder API_KEY in .env")
    args = []
    for name, value in config.items():
        if value is True:
            args.append(f"--{name}")
        elif value is not False and value is not None:
            args.extend([f"--{name}", str(value)])
    args.extend(["--port", os.environ.get("SERVER_PORT", "20670"), "--api-key", api_key])
    os.environ["DSV41_SOURCE"] = str(root)
    print("Starting SGLang with local weights; TP8 / 400K / 50 slots requires hardware validation.", flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *args])


if __name__ == "__main__":
    main()
