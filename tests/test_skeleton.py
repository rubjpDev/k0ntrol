import importlib
import tomllib
from pathlib import Path

import pytest

from k0ntrol import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["memory", "tools", "tracker", "frontends", "workflow"])
def test_subpackage_imports(name):
    importlib.import_module(f"k0ntrol.{name}")


@pytest.mark.parametrize(
    "name", ["harness", "backends", "stages", "modespec", "modes"]
)
def test_retired_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"k0ntrol.{name}")


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_0_1_0_dev0():
    assert __version__ == "0.1.0.dev0"
    assert _pyproject()["project"]["version"] == "0.1.0.dev0"


def test_runtime_dependencies_are_the_approved_closed_list():
    dependencies = _pyproject()["project"]["dependencies"]
    assert dependencies == [
        "pyyaml>=6",
        "rich>=13",
        "langgraph>=1.2.17,<1.3",
        "langchain>=1.4.7,<1.5",
        "langgraph-checkpoint-sqlite>=3.1.2,<3.2",
    ]


def test_provider_and_langfuse_extras():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert set(extras) == {
        "ollama",
        "openai",
        "anthropic",
        "openrouter",
        "aws",
        "google",
        "langfuse",
    }
    assert extras["langfuse"] == ["langfuse>=5.0.4,<6"]
