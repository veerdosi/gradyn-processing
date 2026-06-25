from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "models" / "Cutie" / "cutie" / "model" / "big_modules.py"
    if not target.exists():
        raise SystemExit(f"Cutie source not found: {target}")
    source = target.read_text()
    patched = source.replace(
        "resnet.resnet18(pretrained=True, model_dir=resnet_model_path)",
        "resnet.resnet18(pretrained=False, model_dir=resnet_model_path)",
    ).replace(
        "resnet.resnet50(pretrained=True, model_dir=resnet_model_path)",
        "resnet.resnet50(pretrained=False, model_dir=resnet_model_path)",
    ).replace(
        "resnet.resnet18(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)",
        "resnet.resnet18(pretrained=False, extra_dim=extra_dim, model_dir=resnet_model_path)",
    ).replace(
        "resnet.resnet50(pretrained=True, extra_dim=extra_dim, model_dir=resnet_model_path)",
        "resnet.resnet50(pretrained=False, extra_dim=extra_dim, model_dir=resnet_model_path)",
    )
    target.write_text(patched)
    print("Cutie patched for deterministic offline checkpoint loading.")


if __name__ == "__main__":
    main()
