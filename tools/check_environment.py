"""
KINEX Environment Diagnostics & Verification Tool
Checks Python, PyTorch, TorchVision, Ultralytics, CUDA, and GPU hardware.
Outputs audit results to reports/phase4_environment.json.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_FILE = REPORTS_DIR / "phase4_environment.json"


def check_environment():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    env_data = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": None,
        "torchvision_version": None,
        "ultralytics_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "device": "CPU",
        "status": "INSPECTED",
        "notes": []
    }

    # Check torch
    try:
        import torch
        env_data["torch_version"] = torch.__version__
        env_data["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_data["gpu_name"] = torch.cuda.get_device_name(0)
            env_data["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            env_data["device"] = "GPU"
        else:
            env_data["device"] = "CPU"
    except ImportError:
        env_data["notes"].append("PyTorch is not installed in the current Python environment.")

    # Check torchvision
    try:
        import torchvision
        env_data["torchvision_version"] = torchvision.__version__
    except ImportError:
        env_data["notes"].append("TorchVision is not installed in the current Python environment.")

    # Check ultralytics
    try:
        import ultralytics
        env_data["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        env_data["notes"].append("Ultralytics is not installed in the current Python environment.")

    # Check system GPU hardware via Windows CimInstance/WMI if PyTorch didn't see CUDA
    if not env_data["cuda_available"] and platform.system() == "Windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            gpu_names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            for name in gpu_names:
                if "NVIDIA" in name.upper() or "RTX" in name.upper() or "GEFORCE" in name.upper():
                    env_data["hardware_gpu_detected"] = name
                    env_data["notes"].append(f"Physical hardware GPU detected: {name}, but torch.cuda is currently not linked/active.")
                    break
        except Exception:
            pass

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=4)

    print(f"[ENVIRONMENT AUDIT] Report saved to: {OUTPUT_FILE}")
    print(f"  Python:      {env_data['python_version'].split()[0]}")
    print(f"  Torch:       {env_data['torch_version']}")
    print(f"  TorchVision: {env_data['torchvision_version']}")
    print(f"  Ultralytics: {env_data['ultralytics_version']}")
    print(f"  CUDA:        {env_data['cuda_available']}")
    print(f"  Device:      {env_data['device']}")
    return env_data


if __name__ == "__main__":
    check_environment()
