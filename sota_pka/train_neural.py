from __future__ import annotations

import json
from pathlib import Path


def write_neural_integration_status(output_dir: Path) -> dict[str, object]:
    """Record available neural/SOTA integrations for the current environment."""
    checks = {}
    for package in ["torch", "chemprop", "torch_geometric"]:
        try:
            __import__(package)
            checks[package] = "available"
        except Exception as exc:
            checks[package] = f"unavailable: {exc}"
    cuda = {"available": False}
    if checks.get("torch") == "available":
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            cuda.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "capability": f"sm_{capability[0]}{capability[1]}",
                }
            )
            if capability >= (12, 0) and "+cu124" in torch.__version__:
                cuda["warning"] = "Stable cu124 PyTorch sees this RTX 50 GPU but may not run CUDA kernels for sm_120."
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready = all(value == "available" for value in checks.values())
    status = {
        "status": "ready" if ready else "needs_environment_setup",
        "checks": checks,
        "cuda": cuda,
        "next": "Run Chemprop/graph smoke tests; use PyTorch nightly cu128/cu129 if sm_120 CUDA kernels fail."
        if ready
        else "Install missing packages from configs/environment-wsl.yml before Chemprop/SAT4pKa/Uni-pKa runs.",
    }
    (output_dir / "neural_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
