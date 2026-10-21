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


def test_code_write_keeps_trailing_newline_when_unfenced_body_lacks_one(tmp_path):
    def fake_invoke(mode, prompt):
        return "FILE src/app.py\nprint(1)"

    written = code_write(
        "spec",
        root=tmp_path,
        mode=MODE,
        invoke=fake_invoke,
        target=None,
        reference_text="",
    )
    assert written.read_text(encoding="utf-8") == "print(1)\n"


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


def test_code_write_rejects_project_topology(tmp_path):
    dest = (
        tmp_path
        / ".k0-mem"
        / "Project_topology"
        / tmp_path.name
        / "architecture.md"
    )
    dest.parent.mkdir(parents=True)
    dest.write_text("human\n", encoding="utf-8")
    relative = f".k0-mem/Project_topology/{tmp_path.name}/architecture.md"

    def fake_invoke(mode, prompt):
        return f"FILE {relative}\noverwrite\n"

    with pytest.raises(CodeWriteError, match="Project_topology"):
        code_write(
            "spec",
            root=tmp_path,
            mode=MODE,
            invoke=fake_invoke,
            target=None,
            reference_text="",
        )
    assert dest.read_text(encoding="utf-8") == "human\n"


def _write(tmp_path, response, target=None):
    return code_write(
        "spec",
        root=tmp_path,
        mode=MODE,
        invoke=lambda mode, prompt: response,
        target=target,
        reference_text="",
    )


def test_code_write_drops_preamble_before_file_line(tmp_path):
    written = _write(
        tmp_path,
        "I'll draft the tests now.\nFILE tests/test_app.py\ndef test_one():\n    assert False\n",
    )
    assert written == tmp_path / "tests" / "test_app.py"
    assert written.read_text(encoding="utf-8") == "def test_one():\n    assert False\n"


def test_code_write_unwraps_outer_fence_around_file_line(tmp_path):
    fence = "`" * 3
    written = _write(
        tmp_path,
        f"{fence}python\nFILE tests/test_app.py\ndef test_one():\n    assert False\n{fence}\n",
    )
    assert written.read_text(encoding="utf-8") == "def test_one():\n    assert False\n"


def test_code_write_accepts_file_line_without_trailing_newline(tmp_path):
    written = _write(tmp_path, "FILE src/app.py")
    assert written == tmp_path / "src" / "app.py"
    assert written.read_text(encoding="utf-8") == "\n"


def test_code_write_strips_backticks_around_file_path(tmp_path):
    written = _write(tmp_path, "FILE `src/app.py`\nprint(1)\n")
    assert written == tmp_path / "src" / "app.py"
    assert written.read_text(encoding="utf-8") == "print(1)\n"


def test_code_write_accepts_indented_file_line(tmp_path):
    written = _write(tmp_path, "  FILE src/app.py\nprint(1)\n")
    assert written.read_text(encoding="utf-8") == "print(1)\n"


def test_code_write_skips_blank_lines_before_file_line(tmp_path):
    written = _write(tmp_path, "\n\nFILE src/app.py\nprint(1)\n")
    assert written.read_text(encoding="utf-8") == "print(1)\n"


def test_code_write_accepts_crlf_file_envelope(tmp_path):
    written = _write(
        tmp_path,
        "thinking\r\nFILE src/app.py\r\nprint(1)\r\n",
    )
    assert written.read_text(encoding="utf-8") == "print(1)\n"


def test_code_write_finds_file_line_inside_preamble_then_fence(tmp_path):
    fence = "`" * 3
    written = _write(
        tmp_path,
        (
            "I'll draft it.\n"
            f"{fence}python\n"
            "FILE tests/test_app.py\n"
            "def test_one():\n"
            "    assert False\n"
            f"{fence}\n"
        ),
    )
    assert written.read_text(encoding="utf-8") == "def test_one():\n    assert False\n"


def test_code_write_drops_preamble_inside_outer_fence(tmp_path):
    fence = "`" * 3
    written = _write(
        tmp_path,
        (
            f"{fence}\n"
            "I'll draft it.\n"
            "FILE src/app.py\n"
            "print(1)\n"
            f"{fence}\n"
        ),
    )
    assert written.read_text(encoding="utf-8") == "print(1)\n"


def test_code_write_does_not_invent_file_line_from_thinking(tmp_path):
    with pytest.raises(CodeWriteError, match="response must start with FILE"):
        _write(
            tmp_path,
            "The file should start with FILE tests/test_app.py but here is thinking.\n",
        )

