import importlib

import pytest


@pytest.mark.parametrize("name", ["harness", "backends", "stages", "frontends"])
def test_subpackage_imports(name):
    importlib.import_module(f"k0ntrol.{name}")
