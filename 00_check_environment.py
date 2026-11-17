from __future__ import annotations

import importlib
import platform
import sys

import project_config as config
from cxr_cascade.reporting import print_script_result


def main() -> None:
    required = [
        "numpy",
        "pandas",
        "PIL",
        "pywt",
        "psutil",
        "pydicom",
        "scipy",
        "matplotlib",
        "torch",
        "torchvision",
    ]
    versions = {}
    print(f"Python: {sys.version.split()[0]}")
    print(f"System: {platform.platform()}")
    failed = []
    for name in required:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
            print(f"[OK] {name}: {versions[name]}")
        except Exception as exc:
            failed.append(name)
            print(f"[MISSING] {name}: {exc}")
    print(f"Dataset: {config.DATASET_ROOT}")
    print(f"Dataset exists: {config.DATASET_ROOT.exists()}")
    if "torch" not in failed:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    if failed:
        raise SystemExit(f"Environment incomplete: {', '.join(failed)}")
    print("Environment check passed.")
    print_script_result(
        "00_check_environment.py",
        {
            "status": "passed",
            "python": sys.version.split()[0],
            "packages": versions,
            "dataset_exists": config.DATASET_ROOT.exists(),
        },
    )


if __name__ == "__main__":
    main()
