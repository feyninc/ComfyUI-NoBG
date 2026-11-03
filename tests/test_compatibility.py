import pytest

from check_compatibility import breaking_changes, node_contracts


def test_extract_node_contracts():
    assert node_contracts('''
class Loader:
    RETURN_TYPES = ("MODEL",)
NODE_CLASS_MAPPINGS = {"Loader": Loader}
''') == {"Loader": ("MODEL",)}


@pytest.mark.parametrize("source", [
    "NODE_CLASS_MAPPINGS = build_mappings()",
    'NODE_CLASS_MAPPINGS = {"Missing": Missing}',
    'class Node: pass\nNODE_CLASS_MAPPINGS = {"Node": Node}',
    'class Node: RETURN_TYPES = ["IMAGE"]\nNODE_CLASS_MAPPINGS = {"Node": Node}',
    "NODE_CLASS_MAPPINGS = {}",
])
def test_unsupported_contracts_fail_explicitly(source):
    with pytest.raises(ValueError):
        node_contracts(source)


def test_adding_outputs_and_nodes_is_compatible():
    assert breaking_changes({"A": ("IMAGE",)}, {"A": ("IMAGE", "MASK"), "B": ("MODEL",)}) == []


@pytest.mark.parametrize("current", [{}, {"A": ()}, {"A": ("MASK", "IMAGE")}])
def test_removing_or_reordering_outputs_is_breaking(current):
    assert breaking_changes({"A": ("IMAGE", "MASK")}, current)
