"""The suite never reads the developer's real home (~/.k0ntrol config and credentials)."""

import os
import pwd
from pathlib import Path


def test_tests_never_see_the_real_home():
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)

    assert Path.home() != real_home
