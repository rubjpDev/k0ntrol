"""The terminal front: gates as an option picker, answers behind a bullet,
and the session driving it instead of plain lines."""

import io

from rich.console import Console

from k0ntrol.frontends.prompt import Prompt
from k0ntrol.frontends.render import LiveRenderer
from k0ntrol.frontends.session import run_session
from k0ntrol.frontends.terminal import COMMANDS, Terminal, gate_options, line_style
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


def test_failures_read_red_and_pauses_amber():
    assert line_style("run r1 failed: git checkout failed") == "#F87171"
    assert line_style("run paused · run the same command again to resume it") == "#FBBF24"
    assert line_style("saving session memory…") == ""


def test_the_resume_right_after_a_gate_answer_is_not_shown():
    ui, console = terminal(["1"])
    resumed = {"kind": "run_start", "run_kind": "full", "branch": "k0/1-x", "resumed": True}

    ui.gate(pending())
    ui.event(resumed)
    ui.event(resumed)  # a real resume later on is shown

    assert console.export_text().count("k0/1-x") == 1


def test_first_run_is_a_git_repo_without_config_and_a_machine_without_models(tmp_path):
    from k0ntrol.frontends.terminal import needs_project_setup, needs_provider_setup

    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / ".git").mkdir(parents=True)
    assert needs_project_setup(repo)
    (repo / ".k0-mem").mkdir()
    (repo / ".k0-mem" / "config.yaml").write_text("tracker: github\n")
    assert not needs_project_setup(repo)
    assert not needs_project_setup(tmp_path / "not-a-repo")

    assert needs_provider_setup(home)
    user = home / ".k0ntrol" / "config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("max_attempts: 3\n")
    assert needs_provider_setup(home)
    user.write_text("agents:\n  spec: {provider: ollama, model: m}\n")
    assert not needs_provider_setup(home)
    user.write_text("agents: [\n")
    assert not needs_provider_setup(home)  # the loader reports bad YAML


def test_config_command_reruns_the_project_setup_and_reloads(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "uv.lock").write_text("")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ui, console = terminal(["/config", "2", "1", "1", "1", "/quit"])
    config = {"agents": {}}

    run_session(lambda: "unused", print, root=tmp_path, config=config, ui=ui)

    saved = (tmp_path / ".k0-mem" / "config.yaml").read_text()
    assert "tracker: github" in saved and "uv run pytest -q" in saved
    assert config["tracker"] == "github"  # reloaded in place for the next run


def test_short_path_uses_tilde_and_cuts_from_the_left():
    from k0ntrol.frontends.terminal import short_path

    assert short_path("/Users/me/code/k0", 60, home="/Users/me") == "~/code/k0"
    assert short_path("/srv/a/very/long/path/to/repo", 12, home="/Users/me") == "…ath/to/repo"
