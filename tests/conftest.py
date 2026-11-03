"""Stub out the ComfyUI modules nodes.py imports at module load, so it can be tested standalone."""

import os
import sys
import tempfile
import types
from unittest import mock

import pytest
import torch

# Add the project root directory to Python path
# This allows the tests to import the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODELS_DIR = os.path.join(tempfile.gettempdir(), "comfyui-nobg-tests-models")


def _install_stubs():
    # folder_paths: a tiny registry mirroring the ComfyUI API that nodes.py uses.
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = MODELS_DIR
    folder_paths.folder_names_and_paths = {}

    def add_model_folder_path(folder_name, full_folder_path, is_default=False):
        paths = folder_paths.folder_names_and_paths.setdefault(folder_name, ([], set()))[0]
        if full_folder_path not in paths:
            if is_default:
                paths.insert(0, full_folder_path)
            else:
                paths.append(full_folder_path)

    def get_folder_paths(folder_name):
        return list(folder_paths.folder_names_and_paths.get(folder_name, ([], set()))[0])

    folder_paths.add_model_folder_path = add_model_folder_path
    folder_paths.get_folder_paths = get_folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")
    comfy.model_management = model_management
    comfy.utils = utils
    _reset_comfy_mocks(model_management, utils)

    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = utils


def _reset_comfy_mocks(model_management, utils):
    model_management.get_torch_device = mock.Mock(return_value=torch.device("cpu"))
    model_management.unet_offload_device = mock.Mock(return_value=torch.device("cpu"))
    model_management.free_memory = mock.Mock()
    model_management.soft_empty_cache = mock.Mock()
    model_management.throw_exception_if_processing_interrupted = mock.Mock()
    utils.ProgressBar = mock.MagicMock()


_install_stubs()


@pytest.fixture(autouse=True)
def comfy_mocks():
    """Fresh call-tracking mocks for comfy.model_management / comfy.utils in every test."""
    mm = sys.modules["comfy.model_management"]
    utils = sys.modules["comfy.utils"]
    _reset_comfy_mocks(mm, utils)
    return types.SimpleNamespace(mm=mm, utils=utils)


@pytest.fixture
def nobg_paths(tmp_path, monkeypatch):
    """Point the "nobg" model folder at two fresh tmp dirs (default + an extra_model_paths one)."""
    import folder_paths

    primary = tmp_path / "models" / "nobg"
    extra = tmp_path / "extra" / "nobg"
    primary.mkdir(parents=True)
    extra.mkdir(parents=True)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "nobg", ([str(primary), str(extra)], set()))
    return types.SimpleNamespace(primary=primary, extra=extra)


@pytest.fixture
def fake_nobg(monkeypatch):
    """A fake ``nobg`` package so nothing is downloaded or loaded."""
    nobg = types.ModuleType("nobg")
    nobg.__path__ = []
    nobg.BiRefNet = mock.Mock(name="BiRefNet")
    nobg.Sam3 = mock.Mock(name="Sam3")
    nobg_utils = types.ModuleType("nobg.utils")

    def refine_foreground(image, mask, r=90):
        # CHW image in, CHW image out; scale so tests can tell refined from raw.
        assert image.ndim == 3 and image.shape[0] == 3
        return image * 0.5

    nobg_utils.refine_foreground = mock.Mock(side_effect=refine_foreground)
    nobg.utils = nobg_utils
    monkeypatch.setitem(sys.modules, "nobg", nobg)
    monkeypatch.setitem(sys.modules, "nobg.utils", nobg_utils)
    return nobg
