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
    assert "vcs       github" in text  # a missing vcs reads as github
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
    ui, console = terminal(["/config", "2", "1", "1", "1", "1", "/quit"])
    config = {"agents": {}}

    run_session(lambda: "unused", print, root=tmp_path, config=config, ui=ui)

    saved = (tmp_path / ".k0-mem" / "config.yaml").read_text()
    assert "tracker: github" in saved and "uv run pytest -q" in saved
    assert "vcs: github" in saved
    assert config["tracker"] == "github"  # reloaded in place for the next run


def test_short_path_uses_tilde_and_cuts_from_the_left():
    from k0ntrol.frontends.terminal import short_path

    assert short_path("/Users/me/code/k0", 60, home="/Users/me") == "~/code/k0"
    assert short_path("/srv/a/very/long/path/to/repo", 12, home="/Users/me") == "…ath/to/repo"


def test_session_stats_sum_tokens_and_estimate_cost():
    from k0ntrol.frontends.terminal import SessionStats

    now = [0.0]
    stats = SessionStats(clock=lambda: now[0])
    stats.add({"kind": "model", "provider": "amazon_bedrock", "model": "qwen.qwen3-next-80b-a3b",
               "tokens_in": 1_000_000, "tokens_out": 100_000})
    stats.add({"kind": "tool", "tool": "read_file"})
    now[0] = 125

    assert stats.summary() == ["2m 5s", "↑ 1M ↓ 100k", "≈ $0.27"]
    stats.add({"kind": "model", "provider": "anthropic", "model": "claude-opus-5-5",
               "tokens_in": 10, "tokens_out": 10})
    assert stats.summary()[-1] == "≈ $0.27+"  # one call had no known price


def test_local_models_cost_nothing():
    from k0ntrol.pricing import estimate_cost

    assert estimate_cost("ollama", "gpt-oss:20b", 5_000, 500) == 0.0
    assert estimate_cost("anthropic", "unknown", 5_000, 500) is None


def test_plain_text_chats_with_memory_and_a_bare_issue_number_works_it(tmp_path):
    asked, worked = [], []
    ui, _ = terminal(["what does slugify do?", "and where is it tested?", "#12", "/quit"])

    def fake_ask(question, **kwargs):
        asked.append((question, kwargs["history"]))
        kwargs["on_text"]("it makes slugs")
        return object()

    run_session(
        lambda: "unused", print, root=tmp_path, config={"agents": {}}, ui=ui,
        ask=fake_ask, start=lambda kind, task, **k: worked.append((kind, task)),
    )

    assert asked[0] == ("what does slugify do?", "")
    assert "Q: what does slugify do?\nA: it makes slugs" in asked[1][1]
    assert worked == [("full", "#12")]


def test_small_costs_keep_four_decimals_and_each_question_reprints_its_agent():
    from k0ntrol.frontends.terminal import format_cost

    assert format_cost(0.0012) == "$0.0012"
    assert format_cost(0.27) == "$0.27"
    ui, console = terminal(["first?", "second?"])
    chatty = {"kind": "tool", "agent": "chatty", "tool": "grep", "ok": True}
    for _ in range(2):
        ui.read()
        ui.event(chatty)
    assert console.export_text().count("CHATTY") == 2


def test_the_mode_turns_plain_text_into_its_command():
    from k0ntrol.frontends.terminal import as_command

    assert as_command("chat", "what is x?") == "what is x?"
    assert as_command("fast", "fix the slug") == "/fast fix the slug"
    assert as_command("full", "add is_even") == "/full add is_even"
    assert as_command("fast", "#12") == "#12"
    assert as_command("full", "/status") == "/status"


def test_with_an_editor_the_prompt_is_the_box(tmp_path):
    from k0ntrol.frontends.editor import LineEditor

    keys = iter(list("hola") + ["shift_tab", "enter"])
    console = Console(file=io.StringIO(), record=True, width=80, color_system=None)
    editor = LineEditor(console.file, read_key=lambda: next(keys), commands=COMMANDS,
                        width=lambda: 80, color=False)
    prompt = Prompt(console, commands=COMMANDS, tty=False, read_input=lambda _: "")
    ui = Terminal(console, prompt, LiveRenderer(console, spinner=False), editor=editor)

    assert ui.read() == "/fast hola"
    assert "shift+tab mode" in ui.status_line()


def test_a_gate_with_a_diff_draws_it_below_the_panel():
    ui, console = terminal(["1"])
    diff = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-x = 41\n+x = 42\n"
    gate = Pending(run="r1", gate="review", question="approve?", answers=("ok",),
                   detail="PR 7\n\n## Review\n\nfine", diff=diff)

    ui.gate(gate)

    text = console.export_text()
    assert text.index("Review gate") < text.index("calc.py +1 -1")
    assert "x = 41" in text and "x = 42" in text
