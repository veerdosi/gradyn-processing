from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def frames(self) -> Path:
        return self.source / "frames"

    @property
    def work(self) -> Path:
        return self.root / ".work"

    @property
    def objects(self) -> Path:
        return self.root / "objects"

    @property
    def hands(self) -> Path:
        return self.root / "hands"

    @property
    def depth(self) -> Path:
        return self.root / "depth"

    @property
    def quality(self) -> Path:
        return self.root / "quality"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def create(self) -> None:
        for path in (
            self.root,
            self.source,
            self.frames,
            self.work,
            self.objects,
            self.hands,
            self.depth,
            self.quality,
            self.exports,
        ):
            path.mkdir(parents=True, exist_ok=True)
