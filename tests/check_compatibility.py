"""Compare this repository's static node registrations/return types without imports.

Deliberately fail on dynamic declarations rather than silently skipping checks.
This catches the removed-node and reordered/removed-output breaks checked by
node-diff without executing either checkout or needing a ComfyUI installation.
"""

import argparse
import ast
from pathlib import Path


def node_contracts(source):
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    registrations = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in node.targets
        ):
            registrations = node.value
    if not isinstance(registrations, ast.Dict):
        raise ValueError("NODE_CLASS_MAPPINGS must be a static dictionary")
    contracts = {}
    for key, value in zip(registrations.keys, registrations.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or not isinstance(value, ast.Name):
            raise ValueError("Node registrations must map literal names to local classes")
        if value.id not in classes:
            raise ValueError(f"Registered class {value.id!r} is not defined in nodes.py")
        return_types = None
        for node in classes[value.id].body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RETURN_TYPES" for target in node.targets
            ):
                return_types = ast.literal_eval(node.value)
        if not isinstance(return_types, tuple) or not all(isinstance(item, str) for item in return_types):
            raise ValueError(f"{value.id}.RETURN_TYPES must be a literal tuple of strings")
        contracts[key.value] = return_types
    if not contracts:
        raise ValueError("NODE_CLASS_MAPPINGS is empty")
    return contracts


def breaking_changes(base, current):
    changes = []
    for name, outputs in base.items():
        if name not in current:
            changes.append(f"{name}: node removed")
        elif current[name][:len(outputs)] != outputs:
            changes.append(f"{name}: return types changed from {outputs} to {current[name]}")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    args = parser.parse_args()
    current = node_contracts((args.current / "nodes.py").read_text(encoding="utf-8"))
    if not (args.base / "nodes.py").exists() and not (args.base / "__init__.py").exists():
        print("Initial node package: no base node contracts to compare")
        return
    base = node_contracts((args.base / "nodes.py").read_text(encoding="utf-8"))
    changes = breaking_changes(base, current)
    if changes:
        raise SystemExit("Breaking changes detected:\n" + "\n".join(changes))
    print("No breaking changes detected")


if __name__ == "__main__":
    main()
