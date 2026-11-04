import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

logger = logging.getLogger("ComfyUI-NoBG")

# Local checkpoints live in `models/nobg/<name>/` (extra_model_paths.yaml can add more).
NOBG_DIR = os.path.join(folder_paths.models_dir, "nobg")
folder_paths.add_model_folder_path("nobg", NOBG_DIR)

# Hub repos offered in the loader dropdown, with the architecture each one loads as.
PRESETS = {
    "feyninc/FeyNobg": "birefnet",
    "facebook/sam3": "sam3",
}
ARCHITECTURES = ("birefnet", "sam3")
LOCAL_PREFIX = "local: "

# Never needed by nobg and often large duplicates of the safetensors weights.
DOWNLOAD_IGNORE = ["*.pt", "*.pth", "*.bin", "*.ckpt", "*.onnx", "*.onnx_data", "*.h5", "*.msgpack"]

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class NoBGModel:
    """What the loader hands downstream: the nobg model plus where it should run."""

    model: Any
    architecture: str
    source: str
    device: torch.device
    _processor: Any = field(default=None, repr=False)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self.model.default_processor()
        return self._processor

    def param_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def current_device(self) -> torch.device:
        param = next(iter(self.model.parameters()), None)
        return param.device if param is not None else torch.device("cpu")

    def to_device(self):
        if self.current_device() != self.device:
            if self.device.type != "cpu":
                # Make room next to whatever ComfyUI already holds on the GPU.
                mm.free_memory(int(self.param_bytes() * 1.2), self.device)
            self.model.to(self.device)

    def offload(self):
        offload_device = mm.unet_offload_device()
        if self.current_device() != offload_device:
            self.model.to(offload_device)
            mm.soft_empty_cache()


def _local_models() -> list[str]:
    names = []
    for root in folder_paths.get_folder_paths("nobg"):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
                names.append(LOCAL_PREFIX + name)
    return names


def _resolve_local(name: str) -> str:
    for root in folder_paths.get_folder_paths("nobg"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"nobg model folder {name!r} not found in models/nobg")


def _guess_architecture(name: str) -> str:
    if name in PRESETS:
        return PRESETS[name]
    return "sam3" if "sam3" in name.lower() else "birefnet"


def _load(architecture: str, source: str, sam3_aggregate: str):
    from nobg import BiRefNet, Sam3

    if architecture == "birefnet":
        return BiRefNet.from_pretrained(source)
    if architecture == "sam3":
        return Sam3.from_origin(source, aggregate=sam3_aggregate)
    raise ValueError(f"unknown nobg architecture {architecture!r}")


def _parse_boxes(text: str) -> list[list[float]] | None:
    """``"x1,y1,x2,y2"`` boxes, one per line or separated by ``;``."""
    boxes = []
    for chunk in text.replace(";", "\n").splitlines():
        chunk = chunk.strip().strip("[]()")
        if not chunk:
            continue
        values = [float(v) for v in chunk.replace(" ", ",").split(",") if v]
        if len(values) != 4:
            raise ValueError(f"box {chunk!r} must have 4 values: x1, y1, x2, y2")
        boxes.append(values)
    return boxes or None


class NoBGModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(PRESETS) + _local_models(), {"default": "feyninc/FeyNobg"}),
            },
        }

    RETURN_TYPES = ("NOBG_MODEL",)
    RETURN_NAMES = ("nobg_model",)
    FUNCTION = "load"
    CATEGORY = "NoBG"

    # Retain legacy Python arguments for direct callers, not as ComfyUI widgets.
    def load(self, model, architecture="auto", device="auto", dtype="auto", download_to_models_dir=False, sam3_aggregate="semantic"):
        if model.startswith(LOCAL_PREFIX):
            name = model[len(LOCAL_PREFIX):]
            source = _resolve_local(name)
        else:
            name = model
            source = model
            if download_to_models_dir:
                from huggingface_hub import snapshot_download

                source = snapshot_download(
                    model,
                    local_dir=os.path.join(NOBG_DIR, model.split("/")[-1]),
                    ignore_patterns=DOWNLOAD_IGNORE,
                )
        if architecture == "auto":
            architecture = _guess_architecture(name)

        logger.info("loading nobg %s model from %s", architecture, source)
        net = _load(architecture, source, sam3_aggregate)
        if dtype != "auto":
            net = net.to(DTYPES[dtype])
        net.eval()

        target = mm.get_torch_device() if device == "auto" else torch.device(device)
        return (NoBGModel(net, architecture, source, target),)


class NoBGRemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nobg_model": ("NOBG_MODEL",),
                "image": ("IMAGE",),
                "refine_foreground": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unmix the old background from soft edges (hair, fur) to remove halos."},
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "SAM3 only: what to cut out. Empty uses the model's default prompt.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("cutout", "foreground", "mask")
    OUTPUT_TOOLTIPS = (
        "RGBA cutout; save it with Save Image for a transparent PNG.",
        "RGB foreground colors (refined when refine_foreground is on).",
        "Alpha matte in [0, 1].",
    )
    FUNCTION = "remove"
    CATEGORY = "NoBG"

    def remove(
        self,
        nobg_model: NoBGModel,
        image,
        batch_size=1,
        refine_foreground=False,
        refine_radius=90,
        keep_model_loaded=False,
        prompt="",
        boxes="",
        score_threshold=-1.0,
    ):
        arch = nobg_model.architecture
        prompt = prompt.strip() or None
        parsed_boxes = _parse_boxes(boxes) if boxes else None

        kwargs: dict[str, Any] = {"return_type": "alpha"}
        if arch == "birefnet":
            if prompt or parsed_boxes:
                logger.warning("BiRefNet takes no prompt or boxes; ignoring them")
        elif score_threshold >= 0:
            kwargs["score_threshold"] = score_threshold

        rgb = image[..., :3]
        # nobg takes uint8 HWC arrays (via loadimg) and returns mattes at full resolution.
        frames = [(f.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy() for f in rgb]

        mattes = []
        try:
            nobg_model.to_device()
            processor = nobg_model.processor
            pbar = comfy.utils.ProgressBar(len(frames))
            for start in range(0, len(frames), batch_size):
                mm.throw_exception_if_processing_interrupted()
                chunk = frames[start : start + batch_size]
                call_kwargs = dict(kwargs)
                if arch == "birefnet":
                    out = nobg_model.model.predict(processor, chunk, batch_size=len(chunk), **call_kwargs)
                else:
                    chunk_boxes = [parsed_boxes] * len(chunk) if parsed_boxes else None
                    out = nobg_model.model.predict(
                        processor, chunk, prompt, chunk_boxes, batch_size=len(chunk), **call_kwargs
                    )
                mattes.extend(out)
                pbar.update(len(chunk))
        finally:
            if not keep_model_loaded:
                nobg_model.offload()

        mask = torch.stack([m.float().clamp(0, 1) for m in mattes]).to(rgb.device)

        foreground = rgb
        if refine_foreground:
            from nobg.utils import refine_foreground as refine

            device = mm.get_torch_device()
            refined = []
            for frame, matte in zip(rgb, mask):
                mm.throw_exception_if_processing_interrupted()
                out = refine(frame.permute(2, 0, 1).to(device), matte.to(device), r=refine_radius)
                refined.append(out.permute(1, 2, 0).to(rgb.device, rgb.dtype))
            foreground = torch.stack(refined)

        cutout = torch.cat([foreground, mask.unsqueeze(-1).to(foreground.dtype)], dim=-1)
        return (cutout, foreground, mask)


class NoBGRefineForeground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "radius": ("INT", {"default": 90, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "cutout")
    DESCRIPTION = "Estimate unmixed foreground colors for a soft mask, removing background halos on hair and fur."
    FUNCTION = "refine"
    CATEGORY = "NoBG"

    def refine(self, image, mask, radius):
        from nobg.utils import refine_foreground as refine

        rgb = image[..., :3]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and rgb.shape[0] > 1:
            mask = mask.expand(rgb.shape[0], -1, -1)
        if mask.shape[0] != rgb.shape[0]:
            raise ValueError(f"got {mask.shape[0]} masks for {rgb.shape[0]} images")

        device = mm.get_torch_device()
        refined, mattes = [], []
        for frame, matte in zip(rgb, mask):
            mm.throw_exception_if_processing_interrupted()
            out = refine(frame.permute(2, 0, 1).to(device), matte.to(device), r=radius)
            refined.append(out.permute(1, 2, 0).to(rgb.device, rgb.dtype))
            # refine_foreground resizes the mask to the image internally; match that here.
            if matte.shape != frame.shape[:2]:
                matte = torch.nn.functional.interpolate(
                    matte[None, None], size=frame.shape[:2], mode="bilinear", align_corners=False
                )[0, 0]
            mattes.append(matte.to(rgb.device, rgb.dtype))
        foreground = torch.stack(refined)
        cutout = torch.cat([foreground, torch.stack(mattes).unsqueeze(-1)], dim=-1)
        return (foreground, cutout)


NODE_CLASS_MAPPINGS = {
    "NoBGModelLoader": NoBGModelLoader,
    "NoBGRemoveBackground": NoBGRemoveBackground,
    "NoBGRefineForeground": NoBGRefineForeground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoBGModelLoader": "NoBG Model Loader",
    "NoBGRemoveBackground": "NoBG Remove Background",
    "NoBGRefineForeground": "NoBG Refine Foreground",
}
