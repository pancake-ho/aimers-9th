from __future__ import annotations

import ast
import builtins
import symtable
import unittest
from dataclasses import fields
from pathlib import Path

from src.config import NeuralConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NeuralConfigAlignmentTests(unittest.TestCase):
    def test_no_undefined_global_names(self) -> None:
        source = (PROJECT_ROOT / "src" / "neural.py").read_text(encoding="utf-8")
        table = symtable.symtable(source, "src/neural.py", "exec")
        module_defined = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
        }
        global_references: set[str] = set()

        def collect(current) -> None:
            global_references.update(
                symbol.get_name()
                for symbol in current.get_symbols()
                if symbol.is_global() and symbol.is_referenced()
            )
            for child in current.get_children():
                collect(child)

        collect(table)
        undefined = global_references - module_defined - set(dir(builtins))
        self.assertEqual(undefined, set())

    def test_qualified_stdlib_modules_are_imported(self) -> None:
        source = (PROJECT_ROOT / "src" / "neural.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        qualified_modules = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"gc", "math", "random"}
        }

        self.assertIn("math", qualified_modules)
        self.assertEqual(qualified_modules - imported_modules, set())

    def test_all_config_attributes_are_declared(self) -> None:
        source = (PROJECT_ROOT / "src" / "neural.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared = {field.name for field in fields(NeuralConfig)}
        referenced = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "config"
        }

        self.assertTrue(referenced)
        self.assertEqual(referenced - declared, set())

    def test_backend_probe_only_replaces_declared_config_fields(self) -> None:
        source = (PROJECT_ROOT / "src" / "neural.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared = {field.name for field in fields(NeuralConfig)}

        replaced = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "replace":
                continue
            replaced.update(keyword.arg for keyword in node.keywords if keyword.arg)

        self.assertTrue(replaced)
        self.assertEqual(replaced - declared, set())

    def test_rejected_tabm_contract_is_absent(self) -> None:
        source = (PROJECT_ROOT / "src" / "neural.py").read_text(encoding="utf-8")
        self.assertNotIn("tabm_k", source)
        self.assertNotIn('"tabm"', source)


if __name__ == "__main__":
    unittest.main()
