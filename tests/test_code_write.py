from pathlib import Path

import pytest

from k0ntrol.modespec import ModeSpec
from k0ntrol.stages.code_write import CodeWriteError, code_write


MODE = ModeSpec(
    name="tester",
    instructions="stay on the mode",
    model="m",
    temperature=0.2,
    one_shot=True,
)


def test_code_write_strips_fences_and_writes_inside_root(tmp_path):
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode
        seen["prompt"] = prompt
        fence = "`" * 3
        return (
            "FILE tests/test_app.py\n"
            + fence
            + "python\n"
            "def test_one():\n"
            "    assert False\n"
            + fence
            + "\n"
        )

    written = code_write(
        "spec body",
        root=tmp_path,
        mode=MODE,
        invoke=fake_invoke,
        target=None,
        reference_text="src/app.py\nprint(1)\n",
    )
    assert written == tmp_path / "tests" / "test_app.py"
    assert written.read_text(encoding="utf-8") == "def test_one():\n    assert False\n"
    assert seen["mode"] is MODE
    assert seen["prompt"] == (
        "Write one file. The first line is FILE <relative-path>. Then the file body.\n"
        "\n"
        "spec body\n"
        "\n"
        "Target:\n"
        "<new file>\n"
        "\n"
        "Reference:\n"
        "src/app.py\n"
        "print(1)\n"
    )


def test_existing_target_is_inlined_whole(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("x\n" * 400, encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        seen["prompt"] = prompt
        return "FILE src/app.py\nprint(1)\n"

    written = code_write(
        "spec",
        root=tmp_path,
        mode=MODE,
        invoke=fake_invoke,
        target=target,
        reference_text="",
    )
    assert written == target
    assert written.read_text(encoding="utf-8") == "print(1)\n"
    assert "x\n" * 400 in seen["prompt"]
    assert "Reference:" not in seen["prompt"]


def test_file_line_outside_root_writes_nothing(tmp_path):
    def fake_invoke(mode, prompt):
        return "FILE ../secret.py\nnope\n"

    with pytest.raises(CodeWriteError, match="secret.py"):
        code_write(
            "spec",
            root=tmp_path,
            mode=MODE,
            invoke=fake_invoke,
            target=None,
            reference_text="",
        )
    assert not (tmp_path.parent / "secret.py").exists()
