"""Resolve archived artifact locations without rewriting scientific records."""

import ast
import json
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "experiments/output_path_migration_20261108.json"


def output_mappings():
    return json.loads(MIGRATION.read_text(encoding="utf-8"))["mappings"]


def translate_relative_path(value, mappings=None):
    """Map complete directory prefixes; never alter similarly named siblings."""
    value = str(value).replace("\\", "/")
    if (value.startswith("/") or PureWindowsPath(value).drive
            or ".." in value.split("/")):
        raise ValueError("Expected a repository-relative path without traversal")
    mappings = output_mappings() if mappings is None else mappings
    for old, new in sorted(mappings.items(), key=lambda pair: -len(pair[0])):
        if value == old or value.startswith(old + "/"):
            return new + value[len(old):]
    return value


def resolve_recorded_path(value, root=ROOT):
    """Resolve a current or historical path beneath the named repository."""
    root = Path(root).resolve()
    path = Path(value)
    if path.is_absolute():
        try:
            value = path.resolve().relative_to(root).as_posix()
        except ValueError:
            raise ValueError("Recorded path is outside this repository")
    translated = translate_relative_path(value)
    destination = (root / translated).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise ValueError("Mapped path escapes the repository")
    return destination


def translate_script_paths(source, mappings=None):
    """Adapt archived path literals in memory, preserving the source file.

    Handle both 'outputs/old' literals and Path chains such as
    ROOT / 'outputs' / 'old'. No training, network or reward expression changes.
    """
    mappings = output_mappings() if mappings is None else mappings

    class Paths(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.op, ast.Div):
                return node
            components, base = [], node
            while (isinstance(base, ast.BinOp) and isinstance(base.op, ast.Div)
                   and isinstance(base.right, ast.Constant) and isinstance(base.right.value, str)):
                components.insert(0, base.right.value)
                base = base.left
            if components and isinstance(base, ast.Name) and base.id == "ROOT":
                relative = "/".join(components)
                mapped = translate_relative_path(relative, mappings)
                if mapped != relative:
                    return ast.copy_location(ast.BinOp(left=base, op=ast.Div(),
                                                       right=ast.Constant(value=mapped)), node)
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for old, new in mappings.items():
                    if node.value == old or node.value.startswith(old + "/"):
                        return ast.copy_location(ast.Constant(value=new + node.value[len(old):]), node)
            return node

    tree = Paths().visit(ast.parse(source))
    return ast.fix_missing_locations(tree)
