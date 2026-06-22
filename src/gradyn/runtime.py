from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from rich.console import Console

console = Console()


class StageError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def conda_executable() -> str:
    value = os.environ.get("GRADYN_CONDA") or shutil.which("conda")
    if not value:
        raise StageError("Conda was not found. Set GRADYN_CONDA to its executable path.")
    return value


@lru_cache(maxsize=None)
def conda_environment_python(name: str) -> str:
    result = subprocess.run(
        [conda_executable(), "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    environments = json.loads(result.stdout).get("envs", [])
    suffix = f"/envs/{name}"
    for environment in environments:
        if environment == name or environment.endswith(suffix):
            python = Path(environment) / "bin" / "python"
            if python.exists():
                return str(python)
    raise StageError(f"Conda environment {name!r} was not found.")


def run_command(command: list[str], cwd: Path | None = None) -> None:
    console.print("[dim]$ " + " ".join(command) + "[/dim]")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode:
        raise StageError(f"Command failed with exit code {result.returncode}: {command}")


def run_worker(env: str, worker: str, args: list[str]) -> None:
    command = [
        conda_environment_python(env),
        "-u",
        str(project_root() / "workers" / worker),
        *args,
    ]
    env_vars = os.environ.copy()
    env_vars.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cache_root = project_root() / ".cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "ultralytics").mkdir(parents=True, exist_ok=True)
    (cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
    env_vars.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env_vars.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    env_vars.setdefault("HF_HOME", str(cache_root / "huggingface"))
    # hf-xet can report transferred chunks while leaving the destination blob at
    # zero bytes on some macOS networks. The regular HTTP path is slower to start
    # but writes model.safetensors directly and has reliable visible progress.
    env_vars.setdefault("HF_HUB_DISABLE_XET", "1")
    env_vars.setdefault("PYTHONUNBUFFERED", "1")
    env_vars.setdefault("PYTHONIOENCODING", "utf-8")
    env_vars.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    console.print(f"[bold cyan]Running {worker} in {env}[/bold cyan]")
    process = subprocess.Popen(command, cwd=project_root(), env=env_vars)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        console.print(f"[yellow]Stopping {worker}…[/yellow]")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if return_code:
        raise StageError(f"{worker} failed in Conda environment {env}")


def stage_done(work: Path, name: str, config_hash: str) -> bool:
    marker = work / f"{name}.done.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text())["config_hash"] == config_hash
    except (KeyError, json.JSONDecodeError):
        return False


def mark_stage(work: Path, name: str, config_hash: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    marker = work / f"{name}.done.json"
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"stage": name, "config_hash": config_hash}, indent=2)
    )
    os.replace(temporary, marker)
