from __future__ import annotations

import os
import shutil
import subprocess
import hashlib
import json
import zipfile
from pathlib import Path

from rich.console import Console

from .runtime import conda_environment_python, project_root, run_command

console = Console()

REPOSITORIES = {
    "mlx-sam3": ("https://github.com/Deekshith-Dade/mlx_sam3.git", "d9a92badb6000a93135e01b89cd81a54e7ff9825"),
    "sam2": ("https://github.com/facebookresearch/sam2.git", "2b90b9f5ceec907a1c18123530e92e794ad901a4"),
    "WiLoR": ("https://github.com/rolpotamias/WiLoR.git", "fcb911312a38fa8badd30d9656a167485d61b8f9"),
    "depth-anything-v2": (
        "https://github.com/DepthAnything/Depth-Anything-V2.git",
        "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
    ),
}

SAM3_MODEL_URL = (
    "https://huggingface.co/mlx-community/sam3-image/resolve/main/model.safetensors"
)
SAM3_MODEL_BYTES = 3_402_867_661


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
        destination.unlink()
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


def download_public_weights(*, include_qwen: bool = False) -> None:
    root = project_root()
    downloads = [
        (
            SAM3_MODEL_URL,
            root / "models/mlx-sam3/weights/sam3-image/model.safetensors",
            SAM3_MODEL_BYTES,
        ),
        (
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            root / "models/sam2/checkpoints/sam2.1_hiera_small.pt",
            None,
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
    if not include_qwen:
        console.print(
            "[green]Required explicit-object pipeline weights are ready.[/green]"
        )
        return
    qwen_dir = root / "models/qwen3-vl-4b"
    if not (qwen_dir / "config.json").exists() or not list(
        qwen_dir.glob("*.safetensors")
    ):
        qwen_dir.mkdir(parents=True, exist_ok=True)
        code = (
            "import os; "
            "os.environ['HF_HUB_DISABLE_XET']='1'; "
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='mlx-community/Qwen3-VL-4B-Instruct-4bit', "
            f"local_dir={str(qwen_dir)!r})"
        )
        run_command(
            [conda_environment_python("gradyn-vocab"), "-u", "-c", code]
        )
    console.print(
        "[yellow]Qwen3-VL and required public checkpoints are ready. "
        "Copy licensed MANO_RIGHT.pkl and MANO_LEFT.pkl with `gradyn models install-mano`.[/yellow]"
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


def verify_models(*, include_qwen: bool = False) -> list[str]:
    root = project_root()
    required = [
        *(["models/qwen3-vl-4b/config.json"] if include_qwen else []),
        "models/mlx-sam3/weights/sam3-image/model.safetensors",
        "models/sam2/checkpoints/sam2.1_hiera_small.pt",
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
            item.endswith("sam3-image/model.safetensors")
            and path.stat().st_size != SAM3_MODEL_BYTES
        ):
            problems.append(f"{item} (wrong file size)")
    if include_qwen and not list(
        (root / "models/qwen3-vl-4b").glob("*.safetensors")
    ):
        problems.append("models/qwen3-vl-4b/*.safetensors")
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
        "qwen3_vl_config": root / "models/qwen3-vl-4b/config.json",
        "sam2": root / "models/sam2/checkpoints/sam2.1_hiera_small.pt",
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
