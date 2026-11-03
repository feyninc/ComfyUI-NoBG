"""Checks for the end-to-end harness's output validator (no ComfyUI required)."""

import io

from PIL import Image
import pytest

from run_workflow import run_additional_checks, validate_png


def encode(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def test_validate_png_accepts_real_transparency():
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    image.putpixel((0, 0), (10, 20, 30, 0))
    assert validate_png(encode(image), (2, 2)) == {
        "width": 2, "height": 2, "mode": "RGBA", "alpha_range": [0, 255],
    }


@pytest.mark.parametrize("alpha", [0, 128, 255])
def test_validate_png_rejects_constant_alpha(alpha):
    with pytest.raises(RuntimeError, match="foreground and background"):
        validate_png(encode(Image.new("RGBA", (2, 2), (10, 20, 30, alpha))), (2, 2))


def test_validate_png_rejects_rgb():
    with pytest.raises(RuntimeError, match="Expected RGBA PNG"):
        validate_png(encode(Image.new("RGB", (2, 2))), (2, 2))


def test_validate_png_rejects_wrong_dimensions():
    with pytest.raises(RuntimeError, match="Expected RGBA PNG"):
        validate_png(encode(Image.new("RGBA", (2, 2))), (3, 3))


def test_validate_png_rejects_other_formats():
    with pytest.raises(RuntimeError, match="Expected RGBA PNG"):
        validate_png(encode(Image.new("RGBA", (2, 2)), "TIFF"), (2, 2))


def test_extended_runner_checks_batch_and_both_refiners(tmp_path):
    baseline = Image.new("RGBA", (2, 2), (100, 100, 100, 128))
    refined = Image.new("RGBA", (2, 2), (90, 90, 90, 128))
    calls = []

    def execute(graph, label, expected_count=1):
        calls.append((label, graph, expected_count))
        return [baseline.copy()] * expected_count if "batch" in label else [refined.copy()]

    workflow = {"3": {"inputs": {}}, "4": {"inputs": {}}}
    run_additional_checks(workflow, execute, baseline, True, False, (2, 2), tmp_path)
    assert len(calls) == 3
    assert calls[0][1]["3"]["inputs"]["keep_model_loaded"] is False
    assert calls[0][2] == 2
    assert calls[1][1]["3"]["inputs"]["refine_foreground"] is True
    assert calls[2][1]["5"]["class_type"] == "NoBGRefineForeground"
    assert (tmp_path / "extended-validation.json").exists()


def test_sam3_runner_rejects_ignored_prompts(tmp_path):
    baseline = Image.new("RGBA", (2, 2), (100, 100, 100, 128))
    workflow = {"2": {"inputs": {}}, "3": {"inputs": {}}}
    with pytest.raises(RuntimeError, match="nearly identical masks"):
        run_additional_checks(workflow, lambda *args: [baseline.copy()], baseline, False, True, (2, 2), tmp_path)
