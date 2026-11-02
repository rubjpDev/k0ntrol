"""Suite-wide test environment set by ``tests/conftest.py`` (ADR 0009)."""

import sys

from k0ntrol.tools.runner import run_command


def test_a_nested_pytest_starts_without_plugin_autoload(tmp_path):
    """Set per test, so the child ``test_cmd`` inherits it through ``run_command``."""
    probe = "import os; print(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'))"

    result = run_command([sys.executable, "-c", probe], root=tmp_path, timeout=30)

    assert result.output.strip() == "1"


def test_the_suite_itself_still_autoloads_plugins(request):
    """Never set at import time: xdist workers inherit ``os.environ`` and
    would start without pytest-cov."""
    assert request.config.pluginmanager.hasplugin("pytest_cov")
