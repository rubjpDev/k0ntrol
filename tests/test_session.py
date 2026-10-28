import threading
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


def test_full_is_not_available_yet(tmp_path):
    lines = iter(["/full add checkout", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
    )
    assert code == 0
    assert "/full is not available yet" in "\n".join(shown)


def test_fast_drives_the_pending_gate_to_done(tmp_path):
    """The REPL renders Pending (question + answers + detail), reads the
    answer and calls resume until the run ends."""
    from k0ntrol.workflow import Outcome, Pending

    calls = []

    def fake_start(kind, task, **kwargs):
        calls.append(("start", kind, task))
        return Pending(
            run="r1",
            gate="diff",
            question="tests are green — keep the change?",
            answers=("y", "n"),
            detail=" M calc.py",
        )

    def fake_resume(run_id, decision, **kwargs):
        calls.append(("resume", run_id, decision))
        return Outcome(run=run_id, outcome="done")

    lines = iter(["/fast add a calc module", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        start=fake_start,
        resume=fake_resume,
    )
    assert code == 0
    text = "\n".join(shown)
    assert calls == [("start", "fast", "add a calc module"), ("resume", "r1", "y")]
    assert "tests are green — keep the change?" in text
    assert "y/n" in text
    assert " M calc.py" in text
    assert "done" in text


def test_fast_refusal_is_shown(tmp_path):
    from k0ntrol.workflow import RunRefused

    def refusing(kind, task, **kwargs):
        raise RunRefused("a run is already open; use another git worktree")

    lines = iter(["/fast fix bug", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        start=refusing,
    )
    assert code == 0
    assert "another git worktree" in "\n".join(shown)


def test_fast_bad_gate_answer_reasks(tmp_path):
    from k0ntrol.workflow import Outcome, Pending, RunRefused

    pending = Pending(run="r1", gate="diff", question="keep?", answers=("y", "n"))
    decisions = []

    def fake_resume(run_id, decision, **kwargs):
        decisions.append(decision)
        if decision != "y":
            raise RunRefused(f"answer {decision!r} is not one of: y, n")
        return Outcome(run=run_id, outcome="done")

    lines = iter(["/fast task", "maybe", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        start=lambda *a, **k: pending,
        resume=fake_resume,
    )
    assert code == 0
    assert decisions == ["maybe", "y"]
    assert "not one of: y, n" in "\n".join(shown)


def test_ask_runs_the_ask_path(tmp_path):
    asked = []

    def fake_ask(question, **kwargs):
        asked.append(question)
        kwargs["write"]("an answer")
        kwargs["on_text"]("an answer")

    lines = iter(["/ask what is x?", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        ask=fake_ask,
    )
    assert code == 0
    assert asked == ["what is x?"]
    assert "an answer" in "\n".join(shown)


def test_work_commands_do_not_touch_the_disk_beyond_the_transcript(tmp_path):
    from k0ntrol.workflow import Outcome

    lines = iter(["/full add checkout", "/fast fix bug", "/ask what is x?", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        ask=lambda question, **kwargs: None,
        start=lambda *a, **k: Outcome(run="r1", outcome="done"),
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


# --- reflect after each run ---------------------------------------------------


class _RecordingReflector:
    def __init__(self, shown: list[str] | None = None):
        self.calls: list = []
        self._shown = shown

    def ask_turn(self) -> None:
        self.calls.append("ask")

    def after_run(self, run: str) -> None:
        snapshot = None if self._shown is None else list(self._shown)
        self.calls.append(("run", run, snapshot))

    def finish(self) -> None:
        self.calls.append("finish")


def test_an_ended_run_is_reflected_after_the_outcome(tmp_path):
    from k0ntrol.workflow import Outcome

    shown = []
    reflector = _RecordingReflector(shown)
    lines = iter(["/fast fix bug", "/quit"])
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        start=lambda *a, **k: Outcome(run="r1", outcome="done"),
        reflector=reflector,
    )
    assert code == 0
    (call, *rest) = reflector.calls
    kind, run, snapshot = call
    assert kind == "run"
    assert run == "r1"
    assert "run done" in snapshot  # the outcome was already shown
    assert rest == ["finish"]  # /quit waited


def test_every_outcome_kind_is_reflected(tmp_path):
    from k0ntrol.workflow import Outcome

    reflector = _RecordingReflector()
    starts = iter(
        [
            Outcome(run="r1", outcome="done"),
            Outcome(run="r2", outcome="stuck", reason="no spec"),
            Outcome(run="r3", outcome="rejected", reason="gate n"),
        ]
    )
    lines = iter(["/fast a", "/fast b", "/fast c", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        start=lambda *a, **k: next(starts),
        reflector=reflector,
    )
    assert code == 0
    assert [call[1] for call in reflector.calls[:-1]] == ["r1", "r2", "r3"]
    assert reflector.calls[-1] == "finish"


def test_a_paused_run_is_not_reflected(tmp_path):
    from k0ntrol.workflow import Pending

    reflector = _RecordingReflector()
    inputs = iter(["/fast fix", KeyboardInterrupt(), "/quit"])

    def read_line():
        item = next(inputs)
        if isinstance(item, BaseException):
            raise item
        return item

    pending = Pending(run="r1", gate="diff", question="keep?", answers=("y", "n"))
    code = run_session(
        read_line,
        lambda _line: None,
        root=tmp_path,
        config=config(),
        start=lambda *a, **k: pending,
        reflector=reflector,
    )
    assert code == 0
    assert reflector.calls == ["finish"]


def test_a_refused_or_failed_start_is_not_reflected(tmp_path):
    reflector = _RecordingReflector()

    def boom(*a, **k):
        raise RuntimeError("kaput")

    lines = iter(["/fast fix", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        start=boom,
        reflector=reflector,
    )
    assert code == 0
    assert reflector.calls == ["finish"]


def test_quit_reflects_pending_ask_turns_once(tmp_path):
    reflector = _RecordingReflector()

    def fake_ask(question, **kwargs):
        kwargs["on_text"]("an answer")
        return object()  # a completed CHATTY turn

    lines = iter(["/ask one", "/ask two", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        ask=fake_ask,
        reflector=reflector,
    )
    assert code == 0
    assert reflector.calls == ["ask", "ask", "finish"]


def test_a_refused_ask_is_not_a_pending_turn(tmp_path):
    reflector = _RecordingReflector()

    def refusing_ask(question, **kwargs):
        kwargs["write"]("doctor said no")
        return None

    lines = iter(["/ask one", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        ask=refusing_ask,
        reflector=reflector,
    )
    assert code == 0
    assert reflector.calls == ["finish"]


def test_quit_waits_for_a_running_reflect(tmp_path):
    from k0ntrol.workflow import Outcome

    entered = threading.Event()
    release = threading.Event()
    inside_finish = threading.Event()

    class SlowReflector(_RecordingReflector):
        def after_run(self, run):
            super().after_run(run)
            entered.set()

        def finish(self):
            inside_finish.set()
            release.wait(5)
            super().finish()

    reflector = SlowReflector()
    lines = iter(["/fast fix", "/quit"])
    done = []

    def session():
        done.append(
            run_session(
                lambda: next(lines),
                lambda _line: None,
                root=tmp_path,
                config=config(),
                start=lambda *a, **k: Outcome(run="r1", outcome="done"),
                reflector=reflector,
            )
        )

    thread = threading.Thread(target=session)
    thread.start()
    assert entered.wait(5)
    assert inside_finish.wait(5)
    assert done == []  # /quit is blocked on the running reflect
    release.set()
    thread.join(5)
    assert done == [0]
