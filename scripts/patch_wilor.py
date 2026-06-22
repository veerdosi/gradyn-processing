"""Remove WiLoR's visualization-only imports from the inference path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

utils_init = ROOT / "models/WiLoR/wilor/utils/__init__.py"
text = utils_init.read_text()
for line in (
    "from .renderer import Renderer\n",
    "from .mesh_renderer import MeshRenderer\n",
    "from .skeleton_renderer import SkeletonRenderer\n",
):
    text = text.replace(line, "")
utils_init.write_text(text)

model_file = ROOT / "models/WiLoR/wilor/models/wilor.py"
text = model_file.read_text()
text = text.replace("from ..utils import SkeletonRenderer, MeshRenderer\n", "")
text = text.replace(
    """        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None
""",
    """        self.renderer = None
        self.mesh_renderer = None
""",
)
model_file.write_text(text)

loader_file = ROOT / "models/WiLoR/wilor/models/__init__.py"
text = loader_file.read_text()
text = text.replace(
    "model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)",
    "model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)",
)
loader_file.write_text(text)

print("Applied WiLoR inference-only patch.")

