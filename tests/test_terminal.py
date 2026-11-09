"""The terminal front: gates as an option picker, answers behind a bullet,
and the session driving it instead of plain lines."""

import io

from rich.console import Console

from k0ntrol.frontends.prompt import Prompt
from k0ntrol.frontends.render import LiveRenderer
from k0ntrol.frontends.session import run_session
from k0ntrol.frontends.terminal import COMMANDS, Terminal, gate_options
from k0ntrol.workflow import Outcome, Pending


def terminal(answers):
    """A Terminal on a recorded console, answering prompts from ``answers``."""
    console = Console(file=io.StringIO(), record=True, width=100, color_system=None)
    replies = iter(answers)
    prompt = Prompt(console, commands=COMMANDS, tty=False, read_input=lambda _: next(replies))
    return Terminal(console, prompt, LiveRenderer(console, spinner=False)), console


def pending(gate="spec", answers=("ok", "redo")):
    return Pending(
        run="r1", gate=gate, question="approve this agent brief?",
        answers=answers, detail="## Agent Brief\n\n**Title:** Calc",
    )


def test_known_gates_get_plain_word_options():
    assert gate_options("review", ("ok", "redo", "spec"))[2] == (
        "spec", "Send back to SPEC with feedback"
    )
    assert gate_options("custom", ("a", "b")) == [("a", "a"), ("b", "b")]


def test_a_gate_shows_its_panel_and_returns_the_picked_key():
    ui, console = terminal(["1"])

    assert ui.gate(pending()) == "ok"
    text = console.export_text()
    assert "Spec gate" in text and "Calc" in text
    assert "approve this agent brief?" in text


def test_redo_asks_for_feedback_and_sends_it_with_the_answer():
    ui, _ = terminal(["2", "cover the empty string"])

    assert ui.gate(pending()) == "redo cover the empty string"


def test_a_streamed_answer_sits_behind_a_bullet_and_ends_its_line():
    ui, console = terminal([])

    ui.stream("  forty ")
    ui.stream("two")
    ui.line("done")

    assert "⏺ forty two\ndone" in console.export_text()


def test_the_session_asks_gates_and_reports_through_the_ui(tmp_path):
    ui, console = terminal(["/full add calc", "1", "/status", "/quit"])
    pending_gate = pending()

    def fake_resume(run, answer, **kwargs):
        assert answer == "ok"
        return Outcome(run="r1", outcome="done")

    code = run_session(
        lambda: "unused", print, root=tmp_path,
        config={"tracker": "github", "agents": {"spec": {"provider": "bedrock", "model": "m"}}},
        start=lambda *a, **k: pending_gate, resume=fake_resume, ui=ui,
    )

    assert code == 0
    text = console.export_text()
    assert "Spec gate" in text
    assert "bedrock/m" in text  # /status
    assert "run done" not in text  # the renderer reports the outcome, not the session
