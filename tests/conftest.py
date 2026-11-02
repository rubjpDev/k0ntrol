"""Suite-wide fixtures (ADR 0009)."""

import pytest


@pytest.fixture(autouse=True)
def _no_plugin_autoload_in_nested_pytest(monkeypatch):
    """A nested ``python -m pytest`` skips entry-point plugins: they cost
    seconds to compile under ``run_command``'s fresh bytecode cache.

    Per test, never in ``os.environ`` at import time: xdist workers inherit
    the environment and would start without xdist and pytest-cov.
    """
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
