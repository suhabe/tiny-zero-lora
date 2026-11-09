"""Inspect the environment without downloading or launching a model."""
import importlib.metadata
import json
import platform


def main():
    packages = ["torch", "transformers", "datasets", "trl", "peft", "accelerate", "vllm"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "versions": versions, "cuda_available": False}
    if versions["torch"]:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result.update(gpu=torch.cuda.get_device_name(), bf16_supported=torch.cuda.is_bf16_supported())
    print(json.dumps(result, indent=2))
    print("This only inspects the environment. No baseline or training has run.")
    print("Use the repo's pinned requirements and compatible CUDA drivers for GPU lessons.")


if __name__ == "__main__":
    main()
