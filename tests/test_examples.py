"""Keep template metadata and wiring in sync with the registered nodes."""

import json
from pathlib import Path
import tomllib

from PIL import Image
import pytest

from nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "example_workflows").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_template_registration_widgets_and_links(path):
    workflow = json.loads(path.read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for node in nodes.values():
        if node["type"] not in NODE_CLASS_MAPPINGS:
            continue
        assert node["properties"]["cnr_id"] == project["name"]
        cls = NODE_CLASS_MAPPINGS[node["type"]]
        widgets = [
            spec for section in cls.INPUT_TYPES().values() for spec in section.values()
            if isinstance(spec[0], (list, tuple)) or spec[0] in ("INT", "FLOAT", "BOOLEAN", "STRING")
        ]
        assert len(widgets) == len(node["widgets_values"])
        for spec, value in zip(widgets, node["widgets_values"]):
            if isinstance(spec[0], (list, tuple)):
                assert value in spec[0]
    for link_id, source_id, source_slot, target_id, target_slot, data_type in workflow["links"]:
        source, target = nodes[source_id], nodes[target_id]
        assert source["outputs"][source_slot]["type"] == data_type
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["type"] == data_type
        assert target["inputs"][target_slot]["link"] == link_id
    with Image.open(path.with_suffix(".jpg")) as thumbnail:
        assert thumbnail.format == "JPEG"


def test_registry_icon_meets_size_requirements():
    with Image.open(ROOT / "assets" / "icon.png") as image:
        assert image.width == image.height
        assert image.width <= 400


def test_api_workflow_inputs_match_registered_schema():
    workflow = json.loads((ROOT / "tests" / "workflows" / "NoBG_remove_background_api.json").read_text())
    for node in workflow.values():
        cls = NODE_CLASS_MAPPINGS.get(node["class_type"])
        if cls is None:
            continue
        schema = cls.INPUT_TYPES()
        allowed = {name for section in schema.values() for name in section}
        assert set(node["inputs"]) <= allowed
        assert set(schema["required"]) <= set(node["inputs"])
