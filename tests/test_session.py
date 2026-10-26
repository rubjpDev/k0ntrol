from pathlib import Path

from k0ntrol.frontends.session import run_session
from k0ntrol.memory.layout import MEM, RULES
from k0ntrol.memory.session import add_mem_proposal, list_pending_mem_proposals


def config(**overrides):
    cfg = {
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "gates": [],
        "agents": {"chatty": {"provider": "anthropic", "model": "claude-haiku-5-5"}},
    }
    cfg.update(overrides)
    return cfg


def _seed_mem_files(root: Path) -> tuple[Path, Path, Path]:
    mem = root / ".k0-mem" / "General_mem" / "Mem.md"
    rules = root / ".k0-mem" / "General_mem" / "Rules.md"
    topology = root / ".k0-mem" / "Project_topology" / root.name / "strategy.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    topology.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    rules.write_text(RULES, encoding="utf-8")
    topology.write_text("human owned\n", encoding="utf-8")
    return mem, rules, topology


def test_help_stays_on_screen_and_quit_exits(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    text = "\n".join(shown)
    assert "› /help" in text
    assert "/ask" in text
    assert "/fast" in text
    assert "/full" in text
    assert "/quit" in text
    assert "\x1b[?1049h" not in text


def test_blank_and_unknown_stay_in_session(tmp_path):
    lines = iter(["", "/nope", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    assert "unknown command: /nope" in "\n".join(shown)


def test_full_fast_and_ask_are_not_available_yet(tmp_path):
    lines = iter(["/full add checkout", "/fast fix bug", "/ask what is x?", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    text = "\n".join(shown)
    assert "/full is not available yet" in text
    assert "/fast is not available yet" in text
    assert "/ask is not available yet" in text


def test_work_commands_do_not_touch_the_disk_beyond_the_transcript(tmp_path):
    lines = iter(["/full add checkout", "/fast fix bug", "/ask what is x?", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    assert not (tmp_path / "spec.md").exists()
    assert list(tmp_path.iterdir()) == [tmp_path / ".k0-mem"]


def test_ctrl_c_and_eof_exit_zero(tmp_path):
    def interrupt():
        raise KeyboardInterrupt

    assert (
        run_session(interrupt, lambda _line: None, root=tmp_path, config=config()) == 0
    )

    def end():
        raise EOFError

    assert run_session(end, lambda _line: None, root=tmp_path, config=config()) == 0


def test_transcript_rereads_what_was_shown(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    files = list((tmp_path / ".k0-mem" / "sessions").glob("*.txt"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    for chunk in shown:
        assert chunk in text


def test_session_prints_the_banner_once(tmp_path):
    from k0ntrol import __version__

    lines = iter(["/quit"])
    shown = []
    run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        color=False,
    )
    text = "\n".join(shown)
    assert text.count("▄█████▙") == 1
    assert f"k0ntrol v{__version__}" in text
    assert str(tmp_path) in text
    assert "\x1b" not in text


def test_banner_shows_the_configured_provider(tmp_path):
    lines = iter(["/quit"])
    shown = []
    run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(agents={"chatty": {"provider": "ollama", "model": "m"}}),
        color=False,
    )
    assert "ollama" in "\n".join(shown)


def test_terminal_echo_is_not_reprinted_but_is_recorded(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        echo_input=False,
    )
    assert code == 0
    output = "\n".join(shown)
    assert "› /help" not in output
    transcript = next((tmp_path / ".k0-mem" / "sessions").glob("*.txt"))
    assert "› /help" in transcript.read_text(encoding="utf-8")


def test_session_y_at_prompt_appends_mem_proposal(tmp_path):
    mem, rules, topology = _seed_mem_files(tmp_path)
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    lines = iter(["y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    assert "Mem.md += Prefer sqlite for checkout state? y/n" in shown
    assert mem.read_text(encoding="utf-8") == MEM + "- Prefer sqlite for checkout state\n"
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert list_pending_mem_proposals(tmp_path) == []


def test_session_n_at_prompt_drops_mem_proposal(tmp_path):
    mem, rules, topology = _seed_mem_files(tmp_path)
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    lines = iter(["n", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    assert "Mem.md += Prefer sqlite for checkout state? y/n" in shown
    assert mem.read_text(encoding="utf-8") == MEM
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert list_pending_mem_proposals(tmp_path) == []
