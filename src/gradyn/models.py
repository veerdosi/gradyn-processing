from __future__ import annotations

import shutil
import hashlib
import json
import zipfile
from pathlib import Path

from rich.console import Console

from .runtime import conda_environment_python, project_root, run_command

console = Console()

REPOSITORIES = {
    "mlx-sam3": (
        "https://github.com/Deekshith-Dade/mlx_sam3.git",
        "d9a92badb6000a93135e01b89cd81a54e7ff9825",
    ),
    "sam2": (
        "https://github.com/facebookresearch/sam2.git",
        "main",
    ),
    "Cutie": (
        "https://github.com/hkchengrex/Cutie.git",
        "ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a",
    ),
    "WiLoR": (
        "https://github.com/rolpotamias/WiLoR.git",
        "fcb911312a38fa8badd30d9656a167485d61b8f9",
    ),
    "depth-anything-v2": (
        "https://github.com/DepthAnything/Depth-Anything-V2.git",
        "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
    ),
}

SAM2_SMALL_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
DINOV2_MODEL_ID = "facebook/dinov2-small"


def setup_repositories() -> None:
    root = project_root() / "models"
    root.mkdir(exist_ok=True)
    for name, (url, revision) in REPOSITORIES.items():
        target = root / name
        if target.exists() and not (target / ".git" / "HEAD").exists():
            shutil.rmtree(target)
        if not target.exists():
            run_command(
                ["git", "clone", "--depth", "1", "--recursive", url, str(target)]
            )
        run_command(["git", "fetch", "--depth", "1", "origin", revision], cwd=target)
        run_command(["git", "checkout", "--detach", revision], cwd=target)
        run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=target)


def install_model_runtime_dependencies() -> None:
    root = project_root()
    objects_python = conda_environment_python("gradyn-objects")
    inference_python = conda_environment_python("gradyn-inference")
    run_command(
        [
            objects_python,
            "-m",
            "pip",
            "install",
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            "transformers>=4.57.3",
            "huggingface-hub>=0.30",
            "pillow",
            "pyarrow",
            "scipy",
        ]
    )
    run_command(
        [objects_python, "-m", "pip", "install", "-e", str(root / "models/sam2")]
    )
    run_command(
        [
            objects_python,
            "-m",
            "pip",
            "install",
            "git+https://github.com/openai/CLIP.git",
        ]
    )
    run_command(
        [
            inference_python,
            "-m",
            "pip",
            "install",
            "git+https://github.com/openai/CLIP.git",
        ]
    )


def _download(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            if destination.suffix not in {".pt", ".ckpt"} or zipfile.is_zipfile(
                destination
            ):
                return
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    run_command(
        [
            "curl",
            "-L",
            "--fail",
            "--show-error",
            "--progress-bar",
            "--retry",
            "12",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
            "-o",
            str(partial),
            url,
        ]
    )
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        actual = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {destination.name} has {actual:,} bytes; "
            f"expected {expected_bytes:,}."
        )
    partial.replace(destination)


def download_public_weights() -> None:
    root = project_root()
    downloads = [
        (
            SAM2_SMALL_URL,
            root / "models/sam2/checkpoints/sam2.1_hiera_small.pt",
            None,
        ),
        (
            "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
            root / "models/Cutie/weights/cutie-base-mega.pth",
            140_443_788,
        ),
        (
            "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
            root / "models/WiLoR/pretrained_models/detector.pt",
            None,
        ),
        (
            "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
            root / "models/WiLoR/pretrained_models/wilor_final.ckpt",
            None,
        ),
        (
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
            root
            / "models/depth-anything-v2/checkpoints/depth_anything_v2_vits.pth",
            None,
        ),
    ]
    for url, destination, expected_bytes in downloads:
        _download(url, destination, expected_bytes=expected_bytes)
    hf_models = [
        (DINOV2_MODEL_ID, root / "models/dinov2-small"),
    ]
    for model_id, local_dir in hf_models:
        if not (local_dir / "config.json").exists() or not list(
            local_dir.glob("*.safetensors")
        ):
            local_dir.mkdir(parents=True, exist_ok=True)
            code = (
                "import os; "
                "os.environ['HF_HUB_DISABLE_XET']='1'; "
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={model_id!r}, "
                f"local_dir={str(local_dir)!r})"
            )
            run_command([conda_environment_python("gradyn-objects"), "-u", "-c", code])
    console.print(
        "[green]Required public checkpoints are ready. Copy licensed "
        "MANO_RIGHT.pkl and MANO_LEFT.pkl with `gradyn models install-mano`.[/green]"
    )


def install_mano(right: Path, left: Path) -> None:
    root = project_root()
    destinations = [
        (right, root / "models/WiLoR/mano_data/MANO_RIGHT.pkl"),
        (left, root / "models/WiLoR/mano_data/MANO_LEFT.pkl"),
    ]
    for source, destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    manifest = {
        "license": "MANO model files supplied by the user under their MANO license",
        "files": {
            destination.name: hashlib.sha256(destination.read_bytes()).hexdigest()
            for _, destination in destinations
        },
    }
    (root / "models/WiLoR/mano_data/gradyn_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def verify_models() -> list[str]:
    root = project_root()
    required = [
        "models/grounding-dino-base/config.json",
        "models/dinov2-small/config.json",
        "models/clip-home/.cache/clip/ViT-B-32.pt",
        "models/sam2/sam2/__init__.py",
        "models/sam2/checkpoints/sam2.1_hiera_small.pt",
        "models/Cutie/weights/cutie-base-mega.pth",
        "models/WiLoR/pretrained_models/detector.pt",
        "models/WiLoR/pretrained_models/wilor_final.ckpt",
        "models/WiLoR/mano_data/MANO_RIGHT.pkl",
        "models/WiLoR/mano_data/MANO_LEFT.pkl",
        "models/depth-anything-v2/checkpoints/depth_anything_v2_vits.pth",
    ]
    problems: list[str] = []
    for item in required:
        path = root / item
        if not path.exists():
            problems.append(item)
            continue
        if path.suffix in {".pt", ".ckpt"} and not zipfile.is_zipfile(path):
            problems.append(f"{item} (corrupt or incomplete)")
        if (
            item.endswith("Cutie/weights/cutie-base-mega.pth")
            and path.stat().st_size != 140_443_788
        ):
            problems.append(f"{item} (wrong file size)")
    for directory in [
        root / "models/grounding-dino-base",
        root / "models/dinov2-small",
    ]:
        if directory.exists() and not (
            list(directory.glob("*.safetensors"))
            or list(directory.glob("*.bin"))
        ):
            problems.append(f"{directory.relative_to(root)}/*.safetensors")
    return problems


def model_provenance() -> dict:
    root = project_root()

    def digest(path: Path) -> str | None:
        if not path.exists():
            return None
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    files = {
        "sam2_1_small": root / "models/sam2/checkpoints/sam2.1_hiera_small.pt",
        "grounding_dino_base_config": root / "models/grounding-dino-base/config.json",
        "dinov2_small_config": root / "models/dinov2-small/config.json",
        "clip_vit_b_32": root / "models/clip-home/.cache/clip/ViT-B-32.pt",
        "cutie": root / "models/Cutie/weights/cutie-base-mega.pth",
        "wilor_detector": root / "models/WiLoR/pretrained_models/detector.pt",
        "wilor": root / "models/WiLoR/pretrained_models/wilor_final.ckpt",
        "depth_anything_v2_small": root
        / "models/depth-anything-v2/checkpoints/depth_anything_v2_vits.pth",
        "mano_right": root / "models/WiLoR/mano_data/MANO_RIGHT.pkl",
        "mano_left": root / "models/WiLoR/mano_data/MANO_LEFT.pkl",
    }
    return {
        "repositories": {
            name: {"url": url, "revision": revision}
            for name, (url, revision) in REPOSITORIES.items()
        },
        "weights_sha256": {name: digest(path) for name, path in files.items()},
    }
