"""Unit tests for ComfyUI-NoBG nodes.py (ComfyUI and nobg are stubbed in conftest.py)."""

import importlib.util
import os
import sys
import types
from unittest import mock

import pytest
import torch

import folder_paths
import nodes
from nodes import (
    ARCHITECTURES,
    DTYPES,
    LOCAL_PREFIX,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PRESETS,
    NoBGModel,
    NoBGModelLoader,
    NoBGRefineForeground,
    NoBGRemoveBackground,
)

NODE_CLASSES = [NoBGModelLoader, NoBGRemoveBackground, NoBGRefineForeground]


# --------------------------------------------------------------------------- helpers


class FakeNet:
    """Stands in for a nobg BiRefNet/Sam3 model."""

    def __init__(self, arch="birefnet", device="cpu"):
        self.arch = arch
        self.param = torch.nn.Parameter(torch.zeros(4, device=device))
        self.predict_calls = []
        self.to_calls = []
        self.eval_called = False
        self.processor_calls = 0

    def parameters(self):
        return iter([self.param])

    def to(self, target):
        self.to_calls.append(target)
        # Rebuild the Parameter: assigning .data across devices (e.g. cpu -> meta) is not allowed.
        self.param = torch.nn.Parameter(self.param.detach().to(target))
        return self

    def eval(self):
        self.eval_called = True
        return self

    def default_processor(self):
        self.processor_calls += 1
        return "processor"

    def predict(self, processor, images, *args, batch_size=1, **kwargs):
        self.predict_calls.append({"processor": processor, "n": len(images), "args": args, "batch_size": batch_size, "kwargs": kwargs})
        # One HxW matte per HWC uint8 frame; value encodes the frame mean so results are checkable.
        return [torch.full(img.shape[:2], 0.25) for img in images]


def make_model(arch="birefnet", device="cpu"):
    return NoBGModel(FakeNet(arch), arch, "src", torch.device(device))


def image_batch(n=2, h=8, w=6, channels=3):
    return torch.rand(n, h, w, channels)


def make_local_model(root, name, with_config=True):
    path = root / name
    path.mkdir()
    if with_config:
        (path / "config.json").write_text("{}")
    return path


# --------------------------------------------------------------------------- node metadata


def test_node_mappings_consistent():
    assert set(NODE_CLASS_MAPPINGS) == set(NODE_DISPLAY_NAME_MAPPINGS)
    assert list(NODE_CLASS_MAPPINGS.values()) == NODE_CLASSES
    for key, cls in NODE_CLASS_MAPPINGS.items():
        assert key == cls.__name__
        assert NODE_DISPLAY_NAME_MAPPINGS[key]


def test_package_exports_mappings():
    # Exercise the same package loading mechanism ComfyUI uses, even when the
    # checkout directory is not a valid Python identifier.
    name = "ComfyUI-NoBG"
    init_path = os.path.join(os.path.dirname(nodes.__file__), "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_path)
    pkg = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules):
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
        assert set(pkg.NODE_CLASS_MAPPINGS) == set(NODE_CLASS_MAPPINGS)
        assert pkg.NODE_DISPLAY_NAME_MAPPINGS == NODE_DISPLAY_NAME_MAPPINGS


@pytest.mark.parametrize("cls", NODE_CLASSES)
def test_node_common_attributes(cls):
    assert cls.CATEGORY == "NoBG"
    assert callable(getattr(cls(), cls.FUNCTION))
    assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES)
    if hasattr(cls, "OUTPUT_TOOLTIPS"):
        assert len(cls.OUTPUT_TOOLTIPS) == len(cls.RETURN_TYPES)
    inputs = cls.INPUT_TYPES()
    assert set(inputs) <= {"required", "optional", "hidden"}
    for section in inputs.values():
        for spec in section.values():
            assert isinstance(spec, tuple) and 1 <= len(spec) <= 2
            if len(spec) == 2:
                assert isinstance(spec[1], dict)


def test_loader_metadata():
    assert NoBGModelLoader.RETURN_TYPES == ("NOBG_MODEL",)
    assert NoBGModelLoader.RETURN_NAMES == ("nobg_model",)
    assert NoBGModelLoader.FUNCTION == "load"


def test_loader_input_types(nobg_paths):
    inputs = NoBGModelLoader.INPUT_TYPES()
    req, opt = inputs["required"], inputs["optional"]
    assert list(req) == ["model", "architecture", "device", "dtype"]
    assert list(opt) == ["download_to_models_dir", "sam3_aggregate"]

    models, model_opts = req["model"]
    assert models == list(PRESETS)
    assert model_opts["default"] in models
    assert req["architecture"][0] == ("auto", "birefnet", "sam3")
    assert req["architecture"][1]["default"] == "auto"
    assert req["device"][0] == ("auto", "cuda", "cpu", "mps")
    assert req["dtype"][0] == ("auto", "fp32", "fp16", "bf16")
    assert opt["download_to_models_dir"][0] == "BOOLEAN"
    assert opt["download_to_models_dir"][1]["default"] is False
    assert opt["sam3_aggregate"][0] == ("semantic", "max", "mean")
    assert opt["sam3_aggregate"][1]["default"] == "semantic"


def test_loader_input_types_lists_local_models(nobg_paths):
    make_local_model(nobg_paths.primary, "my-birefnet")
    models = NoBGModelLoader.INPUT_TYPES()["required"]["model"][0]
    assert models == list(PRESETS) + [LOCAL_PREFIX + "my-birefnet"]


def test_remove_background_metadata():
    assert NoBGRemoveBackground.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK")
    assert NoBGRemoveBackground.RETURN_NAMES == ("cutout", "foreground", "mask")
    assert NoBGRemoveBackground.FUNCTION == "remove"


def test_remove_background_input_types():
    inputs = NoBGRemoveBackground.INPUT_TYPES()
    req, opt = inputs["required"], inputs["optional"]
    assert list(req) == ["nobg_model", "image", "batch_size", "refine_foreground", "refine_radius", "keep_model_loaded"]
    assert list(opt) == ["prompt", "boxes", "score_threshold"]
    assert req["nobg_model"] == ("NOBG_MODEL",)
    assert req["image"] == ("IMAGE",)
    assert req["batch_size"][0] == "INT"
    assert req["batch_size"][1]["default"] == 1
    assert req["refine_foreground"][0] == "BOOLEAN"
    assert req["refine_radius"][0] == "INT"
    assert req["keep_model_loaded"][0] == "BOOLEAN"
    assert opt["prompt"][0] == "STRING"
    assert opt["boxes"][0] == "STRING"
    assert opt["score_threshold"][0] == "FLOAT"
    assert opt["score_threshold"][1]["default"] == -1.0


def test_refine_foreground_metadata():
    assert NoBGRefineForeground.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert NoBGRefineForeground.RETURN_NAMES == ("foreground", "cutout")
    assert NoBGRefineForeground.FUNCTION == "refine"
    inputs = NoBGRefineForeground.INPUT_TYPES()
    assert inputs["required"]["image"] == ("IMAGE",)
    assert inputs["required"]["mask"] == ("MASK",)
    assert inputs["required"]["radius"][0] == "INT"


@pytest.mark.parametrize("cls", NODE_CLASSES)
def test_int_and_float_defaults_within_bounds(cls):
    for section in cls.INPUT_TYPES().values():
        for spec in section.values():
            if len(spec) == 2 and spec[0] in ("INT", "FLOAT"):
                opts = spec[1]
                assert opts.get("min", float("-inf")) <= opts["default"] <= opts.get("max", float("inf"))


# --------------------------------------------------------------------------- module constants


def test_model_folder_registered():
    assert nodes.NOBG_DIR == os.path.join(folder_paths.models_dir, "nobg")
    assert nodes.NOBG_DIR in folder_paths.get_folder_paths("nobg")


def test_presets_use_known_architectures():
    assert set(PRESETS.values()) <= set(ARCHITECTURES)


def test_dtype_mapping():
    assert DTYPES == {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# --------------------------------------------------------------------------- local model discovery


def test_local_models_empty(nobg_paths):
    assert nodes._local_models() == []


def test_local_models_requires_config_and_sorts(nobg_paths):
    make_local_model(nobg_paths.primary, "zeta")
    make_local_model(nobg_paths.primary, "alpha")
    make_local_model(nobg_paths.primary, "no-config", with_config=False)
    (nobg_paths.primary / "stray.safetensors").write_bytes(b"")
    make_local_model(nobg_paths.extra, "extra-model")
    assert nodes._local_models() == [LOCAL_PREFIX + "alpha", LOCAL_PREFIX + "zeta", LOCAL_PREFIX + "extra-model"]


def test_local_models_skips_missing_roots(tmp_path, monkeypatch):
    present = tmp_path / "present"
    present.mkdir()
    make_local_model(present, "m")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "nobg", ([str(tmp_path / "missing"), str(present)], set()))
    assert nodes._local_models() == [LOCAL_PREFIX + "m"]


def test_resolve_local_first_root_wins(nobg_paths):
    make_local_model(nobg_paths.extra, "dup")
    first = make_local_model(nobg_paths.primary, "dup")
    assert nodes._resolve_local("dup") == str(first)


def test_resolve_local_falls_back_to_extra_root(nobg_paths):
    path = make_local_model(nobg_paths.extra, "only-extra", with_config=False)
    assert nodes._resolve_local("only-extra") == str(path)


def test_resolve_local_missing(nobg_paths):
    with pytest.raises(FileNotFoundError, match="nope"):
        nodes._resolve_local("nope")


# --------------------------------------------------------------------------- architecture detection


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("feyninc/FeyNobg", "birefnet"),
        ("facebook/sam3", "sam3"),
        ("my-SAM3-finetune", "sam3"),
        ("someone/sam3.1-hiera", "sam3"),
        ("ZhengPeng7/BiRefNet", "birefnet"),
        ("anything-else", "birefnet"),
    ],
)
def test_guess_architecture(name, expected):
    assert nodes._guess_architecture(name) == expected


def test_guess_architecture_prefers_presets(monkeypatch):
    monkeypatch.setitem(nodes.PRESETS, "org/sam3-but-birefnet", "birefnet")
    assert nodes._guess_architecture("org/sam3-but-birefnet") == "birefnet"


# --------------------------------------------------------------------------- box parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        ("   \n  ", None),
        ("1,2,3,4", [[1.0, 2.0, 3.0, 4.0]]),
        ("1, 2, 3, 4", [[1.0, 2.0, 3.0, 4.0]]),
        ("1 2 3 4", [[1.0, 2.0, 3.0, 4.0]]),
        ("[1,2,3,4]", [[1.0, 2.0, 3.0, 4.0]]),
        ("(1.5,2.5,3.5,4.5)", [[1.5, 2.5, 3.5, 4.5]]),
        ("1,2,3,4\n5,6,7,8", [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        ("1,2,3,4;5,6,7,8", [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        ("\n  1,2,3,4  \n\n;", [[1.0, 2.0, 3.0, 4.0]]),
        ("-1,0,1e2,4", [[-1.0, 0.0, 100.0, 4.0]]),
    ],
)
def test_parse_boxes(text, expected):
    assert nodes._parse_boxes(text) == expected


@pytest.mark.parametrize("text", ["1,2,3", "1,2,3,4,5", "1,2,3,4\n5,6"])
def test_parse_boxes_wrong_count(text):
    with pytest.raises(ValueError, match="must have 4 values"):
        nodes._parse_boxes(text)


def test_parse_boxes_not_a_number():
    with pytest.raises(ValueError):
        nodes._parse_boxes("a,b,c,d")


# --------------------------------------------------------------------------- _load


def test_load_birefnet(fake_nobg):
    out = nodes._load("birefnet", "some/repo", "semantic")
    fake_nobg.BiRefNet.from_pretrained.assert_called_once_with("some/repo")
    fake_nobg.Sam3.from_origin.assert_not_called()
    assert out is fake_nobg.BiRefNet.from_pretrained.return_value


def test_load_sam3(fake_nobg):
    out = nodes._load("sam3", "facebook/sam3", "max")
    fake_nobg.Sam3.from_origin.assert_called_once_with("facebook/sam3", aggregate="max")
    assert out is fake_nobg.Sam3.from_origin.return_value


def test_load_unknown(fake_nobg):
    with pytest.raises(ValueError, match="unknown nobg architecture"):
        nodes._load("u2net", "x", "semantic")


# --------------------------------------------------------------------------- NoBGModel


def test_nobg_model_processor_is_cached():
    m = make_model()
    assert m.processor == "processor"
    assert m.processor == "processor"
    assert m.model.processor_calls == 1


def test_nobg_model_param_bytes_and_device():
    m = make_model()
    assert m.param_bytes() == 4 * 4  # four float32 params
    assert m.current_device() == torch.device("cpu")


def test_nobg_model_current_device_without_params():
    m = make_model()
    m.model.parameters = lambda: iter([])
    assert m.current_device() == torch.device("cpu")


def test_nobg_model_to_device_noop_when_already_there(comfy_mocks):
    m = make_model()
    m.to_device()
    assert m.model.to_calls == []
    comfy_mocks.mm.free_memory.assert_not_called()


def test_nobg_model_to_device_frees_gpu_memory(comfy_mocks):
    m = make_model()
    m.device = torch.device("meta")  # any non-cpu device; avoids needing a real GPU
    m.to_device()
    comfy_mocks.mm.free_memory.assert_called_once_with(int(16 * 1.2), torch.device("meta"))
    assert m.model.to_calls == [torch.device("meta")]


def test_nobg_model_offload(comfy_mocks):
    m = make_model()
    m.offload()  # already on the offload device
    comfy_mocks.mm.soft_empty_cache.assert_not_called()

    comfy_mocks.mm.unet_offload_device.return_value = torch.device("meta")
    m.offload()
    assert m.model.to_calls == [torch.device("meta")]
    comfy_mocks.mm.soft_empty_cache.assert_called_once()


# --------------------------------------------------------------------------- NoBGModelLoader.load


@pytest.fixture
def fake_load(monkeypatch):
    net = FakeNet()
    loader = mock.Mock(return_value=net)
    monkeypatch.setattr(nodes, "_load", loader)
    return types.SimpleNamespace(net=net, load=loader)


def test_loader_hub_preset_auto(fake_load, comfy_mocks):
    (out,) = NoBGModelLoader().load("feyninc/FeyNobg", "auto", "auto", "auto")
    fake_load.load.assert_called_once_with("birefnet", "feyninc/FeyNobg", "semantic")
    assert isinstance(out, NoBGModel)
    assert out.architecture == "birefnet"
    assert out.source == "feyninc/FeyNobg"
    assert out.model is fake_load.net
    assert out.device == torch.device("cpu")
    assert fake_load.net.eval_called
    assert fake_load.net.to_calls == []  # dtype "auto" keeps the checkpoint dtype
    comfy_mocks.mm.get_torch_device.assert_called_once()


def test_loader_sam3_preset_passes_aggregate(fake_load):
    (out,) = NoBGModelLoader().load("facebook/sam3", "auto", "cpu", "auto", sam3_aggregate="mean")
    fake_load.load.assert_called_once_with("sam3", "facebook/sam3", "mean")
    assert out.architecture == "sam3"


@pytest.mark.parametrize(("dtype", "expected"), list(DTYPES.items()))
def test_loader_dtype(fake_load, dtype, expected):
    NoBGModelLoader().load("feyninc/FeyNobg", "auto", "cpu", dtype)
    assert fake_load.net.to_calls == [expected]
    assert fake_load.net.param.dtype == expected


def test_loader_explicit_device_and_architecture(fake_load, comfy_mocks):
    (out,) = NoBGModelLoader().load("feyninc/FeyNobg", "sam3", "cpu", "auto")
    fake_load.load.assert_called_once_with("sam3", "feyninc/FeyNobg", "semantic")
    assert out.device == torch.device("cpu")
    comfy_mocks.mm.get_torch_device.assert_not_called()


def test_loader_local_model(fake_load, nobg_paths):
    path = make_local_model(nobg_paths.extra, "my-sam3-ft")
    (out,) = NoBGModelLoader().load(LOCAL_PREFIX + "my-sam3-ft", "auto", "cpu", "auto")
    fake_load.load.assert_called_once_with("sam3", str(path), "semantic")
    assert out.source == str(path)


def test_loader_local_model_missing(fake_load, nobg_paths):
    with pytest.raises(FileNotFoundError):
        NoBGModelLoader().load(LOCAL_PREFIX + "gone", "auto", "cpu", "auto")
    fake_load.load.assert_not_called()


def test_loader_download_to_models_dir(fake_load, monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = mock.Mock(return_value="/downloaded/FeyNobg")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    (out,) = NoBGModelLoader().load("feyninc/FeyNobg", "auto", "cpu", "auto", download_to_models_dir=True)
    hub.snapshot_download.assert_called_once_with(
        "feyninc/FeyNobg",
        local_dir=os.path.join(nodes.NOBG_DIR, "FeyNobg"),
        ignore_patterns=nodes.DOWNLOAD_IGNORE,
    )
    fake_load.load.assert_called_once_with("birefnet", "/downloaded/FeyNobg", "semantic")
    assert out.source == "/downloaded/FeyNobg"


# --------------------------------------------------------------------------- NoBGRemoveBackground.remove


def run_remove(model, image, **overrides):
    kwargs = {"batch_size": 1, "refine_foreground": False, "refine_radius": 90, "keep_model_loaded": True}
    kwargs.update(overrides)
    return NoBGRemoveBackground().remove(model, image, **kwargs)


def test_remove_birefnet_outputs(comfy_mocks):
    model = make_model("birefnet")
    image = image_batch(n=3, h=8, w=6)
    cutout, foreground, mask = run_remove(model, image, batch_size=2)

    assert cutout.shape == (3, 8, 6, 4)
    assert foreground.shape == (3, 8, 6, 3)
    assert mask.shape == (3, 8, 6)
    assert torch.equal(foreground, image)
    assert torch.allclose(cutout[..., 3], mask)
    assert torch.allclose(mask, torch.full_like(mask, 0.25))

    calls = model.model.predict_calls
    assert [c["n"] for c in calls] == [2, 1]
    assert [c["batch_size"] for c in calls] == [2, 1]
    assert all(c["args"] == () and c["kwargs"] == {"return_type": "alpha"} for c in calls)
    assert all(c["processor"] == "processor" for c in calls)

    comfy_mocks.utils.ProgressBar.assert_called_once_with(3)
    assert comfy_mocks.mm.throw_exception_if_processing_interrupted.call_count == 2


def test_remove_drops_alpha_channel_and_converts_frames():
    model = make_model("birefnet")
    seen = []
    original = model.model.predict

    def spy(processor, images, *args, **kwargs):
        seen.extend(images)
        return original(processor, images, *args, **kwargs)

    model.model.predict = spy
    image = torch.tensor([[[[0.0, 0.5, 1.0, 0.3]]]])  # 1x1x1 RGBA
    cutout, foreground, _ = run_remove(model, image)
    assert foreground.shape[-1] == 3
    assert cutout.shape[-1] == 4
    assert seen[0].dtype.name == "uint8"
    assert seen[0].tolist() == [[[0, 128, 255]]]


def test_remove_birefnet_ignores_prompt_and_boxes(caplog):
    model = make_model("birefnet")
    with caplog.at_level("WARNING", logger="ComfyUI-NoBG"):
        run_remove(model, image_batch(n=1), prompt="cat", boxes="1,2,3,4", score_threshold=0.5)
    assert "ignoring" in caplog.text
    call = model.model.predict_calls[0]
    assert call["args"] == ()
    assert call["kwargs"] == {"return_type": "alpha"}  # score_threshold is SAM3-only


def test_remove_sam3_prompt_boxes_threshold():
    model = make_model("sam3")
    run_remove(model, image_batch(n=3), batch_size=2, prompt="  a cat \n", boxes="1,2,3,4;5,6,7,8", score_threshold=0.4)
    calls = model.model.predict_calls
    boxes = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert calls[0]["args"] == ("a cat", [boxes, boxes])
    assert calls[1]["args"] == ("a cat", [boxes])
    assert all(c["kwargs"] == {"return_type": "alpha", "score_threshold": 0.4} for c in calls)


def test_remove_sam3_defaults():
    model = make_model("sam3")
    run_remove(model, image_batch(n=1), prompt="   ", boxes="", score_threshold=-1.0)
    call = model.model.predict_calls[0]
    assert call["args"] == (None, None)
    assert call["kwargs"] == {"return_type": "alpha"}


def test_remove_bad_boxes_fail_before_inference():
    model = make_model("sam3")
    with pytest.raises(ValueError):
        run_remove(model, image_batch(n=1), boxes="1,2,3")
    assert model.model.predict_calls == []


@pytest.mark.parametrize(("keep", "offloaded"), [(True, False), (False, True)])
def test_remove_keep_model_loaded(keep, offloaded):
    model = make_model()
    with mock.patch.object(NoBGModel, "offload") as offload:
        run_remove(model, image_batch(n=1), keep_model_loaded=keep)
    assert offload.called is offloaded


def test_remove_offloads_even_on_error():
    model = make_model()
    model.model.predict = mock.Mock(side_effect=RuntimeError("boom"))
    with mock.patch.object(NoBGModel, "offload") as offload, pytest.raises(RuntimeError, match="boom"):
        run_remove(model, image_batch(n=1), keep_model_loaded=False)
    offload.assert_called_once()


def test_remove_offloads_when_processor_creation_fails():
    model = make_model()
    model.model.default_processor = mock.Mock(side_effect=RuntimeError("processor setup failed"))
    with mock.patch.object(NoBGModel, "offload") as offload, pytest.raises(RuntimeError, match="processor setup failed"):
        run_remove(model, image_batch(n=1), keep_model_loaded=False)
    offload.assert_called_once()


def test_remove_refine_foreground(fake_nobg):
    model = make_model()
    image = image_batch(n=2)
    cutout, foreground, mask = run_remove(model, image, refine_foreground=True, refine_radius=33)
    assert torch.allclose(foreground, image * 0.5)
    assert torch.allclose(cutout[..., :3], foreground)
    assert torch.allclose(cutout[..., 3], mask)
    refine = fake_nobg.utils.refine_foreground
    assert refine.call_count == 2
    assert all(c.kwargs == {"r": 33} for c in refine.call_args_list)


# --------------------------------------------------------------------------- NoBGRefineForeground.refine


def test_refine_node_batch(fake_nobg):
    image = image_batch(n=2, h=8, w=6, channels=4)
    mask = torch.rand(2, 8, 6)
    foreground, cutout = NoBGRefineForeground().refine(image, mask, 12)
    assert foreground.shape == (2, 8, 6, 3)
    assert cutout.shape == (2, 8, 6, 4)
    assert torch.allclose(foreground, image[..., :3] * 0.5)
    assert torch.allclose(cutout[..., 3], mask)
    assert all(c.kwargs == {"r": 12} for c in fake_nobg.utils.refine_foreground.call_args_list)


def test_refine_node_2d_mask_broadcasts(fake_nobg):
    image = image_batch(n=3)
    mask = torch.rand(8, 6)
    foreground, cutout = NoBGRefineForeground().refine(image, mask, 90)
    assert cutout.shape == (3, 8, 6, 4)
    for i in range(3):
        assert torch.allclose(cutout[i, ..., 3], mask)


def test_refine_node_resizes_mask(fake_nobg):
    image = image_batch(n=1, h=8, w=6)
    mask = torch.ones(1, 4, 3)
    _, cutout = NoBGRefineForeground().refine(image, mask, 90)
    assert cutout.shape == (1, 8, 6, 4)
    assert torch.allclose(cutout[..., 3], torch.ones(1, 8, 6))


def test_refine_node_mismatched_batch(fake_nobg):
    with pytest.raises(ValueError, match="got 2 masks for 3 images"):
        NoBGRefineForeground().refine(image_batch(n=3), torch.rand(2, 8, 6), 90)
