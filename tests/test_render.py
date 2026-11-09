import io

import pytest
from rich.console import Console

from k0ntrol.frontends.render import (
    LiveRenderer,
    format_elapsed,
    format_ms,
    format_tokens,
    render_gate,
    style_detail,
    tool_label,
    tool_line,
)
from k0ntrol.frontends.theme import ACCENT, DIM, ERROR, OK, SPINNER_FRAMES, WARN


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_console() -> Console:
    return Console(record=True, width=100, force_terminal=False, color_system=None)


def lines_of(console: Console) -> list[str]:
    return [line.rstrip() for line in console.export_text().splitlines()]


@pytest.fixture
def console() -> Console:
    return make_console()


@pytest.fixture
def renderer(console: Console) -> LiveRenderer:
    return LiveRenderer(console, spinner=False, clock=FakeClock())


def model(agent: str = "coder", **fields) -> dict:
    return {"kind": "model", "agent": agent, "ok": True, **fields}


def tool(name: str = "read_file", **fields) -> dict:
    return {"kind": "tool", "agent": "coder", "tool": name, "ok": True, **fields}


# --- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "text"),
    [
        (0, "0"),
        (950, "950"),
        (1000, "1k"),
        (34_500, "34.5k"),
        (999_999, "1M"),
        (1_200_000, "1.2M"),
    ],
)
def test_format_tokens(count: int, text: str) -> None:
    assert format_tokens(count) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (-3, "0s"),
        (45.9, "45s"),
        (192, "3m 12s"),
        (352, "5m 52s"),
        (3720, "1h 2m"),
    ],
)
def test_format_elapsed(seconds: float, text: str) -> None:
    assert format_elapsed(seconds) == text


def test_format_ms() -> None:
    assert format_ms(28) == "28ms"
    assert format_ms(1500) == "1.5s"
    assert format_ms(2000) == "2s"


def test_tool_label() -> None:
    assert tool_label("read_file", "src/x.py") == "Read(src/x.py)"
    assert tool_label("run_tests") == "RunTests"
    assert tool_label("mystery", "a") == "mystery(a)"
    assert tool_label(None) == "tool"


# --- run lifecycle ---------------------------------------------------------


def test_run_start_line(renderer: LiveRenderer, console: Console) -> None:
    renderer(
        {
            "kind": "run_start",
            "run_kind": "full",
            "work_item": "rubjpDev/k0-scratch#4",
            "branch": "k0/4-slug",
        }
    )
    assert lines_of(console) == ["⏺ /full rubjpDev/k0-scratch#4 · k0/4-slug"]


def test_run_start_resumed_and_minimal(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer({"kind": "run_start", "run_kind": "fast", "resumed": True})
    renderer({"kind": "run_start"})
    assert lines_of(console) == ["⏺ /fast (resumed)", "⏺ /?"]


def test_run_end_done(renderer: LiveRenderer, console: Console) -> None:
    renderer({"kind": "run_end", "outcome": "done", "ms": 352_000})
    assert lines_of(console) == ["✓ Run done · 5m 52s"]


def test_run_end_stuck(renderer: LiveRenderer, console: Console) -> None:
    renderer(
        {
            "kind": "run_end",
            "outcome": "stuck",
            "ms": 723_000,
            "reason": "model_call_limit",
            "node": "coder",
        }
    )
    assert lines_of(console) == ["✗ Run stuck · model_call_limit at coder · 12m 3s"]


def test_run_end_rejected_and_bare(renderer: LiveRenderer, console: Console) -> None:
    renderer({"kind": "run_end", "outcome": "rejected", "reason": "reviewer said no"})
    renderer({"kind": "run_end"})
    assert lines_of(console) == ["Run rejected · reviewer said no", "Run ?"]


def test_unknown_kind_and_garbage_are_ignored(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer({"kind": "weird"})
    renderer({})
    assert lines_of(console) == []


# --- agent headers and model events ----------------------------------------


def test_header_printed_only_when_agent_changes(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer(model("spec"))
    renderer(model("spec"))
    renderer(tool(agent="spec"))
    renderer(model("coder"))
    assert lines_of(console) == [
        "⏺ SPEC",
        "  ⎿  Read",
        "⏺ CODER",
    ]


def test_helper_header_is_lowercase(renderer: LiveRenderer, console: Console) -> None:
    renderer({"kind": "model", "helper": "bulk_read", "ok": True})
    assert lines_of(console) == ["⏺ bulk_read"]


def test_header_reprinted_after_a_new_run_starts(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer(model("spec"))
    renderer({"kind": "run_start", "run_kind": "full"})
    renderer(model("spec"))
    assert lines_of(console) == ["⏺ SPEC", "⏺ /full", "⏺ SPEC"]


def test_successful_model_call_prints_nothing_extra(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer(model("coder", tokens_in=10, tokens_out=5))
    assert lines_of(console) == ["⏺ CODER"]


def test_failed_model_call_shows_first_line_capped() -> None:
    console = Console(record=True, width=300, force_terminal=False, color_system=None)
    renderer = LiveRenderer(console, spinner=False)
    renderer(model("coder", ok=False, err="boom\nsecond line"))
    renderer(model("coder", ok=False, err="x" * 500))
    out = lines_of(console)
    assert out[1] == "  ⎿  ✗ model call failed: boom"
    failure = out[2].removeprefix("  ⎿  ✗ model call failed: ")
    assert len(failure) == 160
    assert failure.endswith("…")


def test_model_event_without_agent_never_crashes(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer({"kind": "model", "tokens_in": "lots"})
    assert lines_of(console) == []


# --- tool events -----------------------------------------------------------


def test_tool_line_with_path_and_timing(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer(tool("read_file", path="src/x.py", ms=28))
    assert lines_of(console)[-1] == "  ⎿  Read(src/x.py) · 28ms"


def test_tool_line_without_path_has_no_parentheses(
    renderer: LiveRenderer, console: Console
) -> None:
    renderer(tool("run_tests", ms=1500))
    assert lines_of(console)[-1] == "  ⎿  RunTests · 1.5s"


def test_tool_denied(renderer: LiveRenderer, console: Console) -> None:
    renderer(
        tool("edit_file", path="src/x.py", ok=False, denied="path_guard", ms=1)
    )
    assert lines_of(console)[-1] == "  ⎿  ✗ Edit(src/x.py) denied (path_guard)"


def test_tool_denied_without_reason(renderer: LiveRenderer, console: Console) -> None:
    renderer(tool("edit_file", ok=False, denied=True))
    assert lines_of(console)[-1] == "  ⎿  ✗ Edit denied"


def test_tool_failed_shows_error(renderer: LiveRenderer, console: Console) -> None:
    renderer(tool("edit_file", path="src/x.py", ok=False, err="no match\nmore"))
    assert lines_of(console)[-1] == "  ⎿  ✗ Edit(src/x.py): no match"


def test_tool_failed_without_error(renderer: LiveRenderer, console: Console) -> None:
    renderer(tool("grep", ok=False))
    assert lines_of(console)[-1] == "  ⎿  ✗ Search"


def test_failures_use_error_and_warn_colours() -> None:
    warn = tool_line(tool("edit_file", ok=False, err="x"))
    denied = tool_line(tool("edit_file", ok=False, denied="allowlist"))
    assert {span.style for span in warn.spans} >= {WARN}
    assert {span.style for span in denied.spans} >= {ERROR}


# --- gates -----------------------------------------------------------------


def test_gate_waiting_prints_nothing(renderer: LiveRenderer, console: Console) -> None:
    renderer({"kind": "gate", "gate": "spec", "result": "waiting"})
    assert lines_of(console) == []


@pytest.mark.parametrize(
    ("gate", "result", "line"),
    [
        ("red_check", "pass", "⏺ Red check passed"),
        ("validator", "fail", "⏺ Validator failed"),
        ("complexity", "pass", "⏺ Complexity passed"),
        ("coverage", "fail", "⏺ Coverage failed"),
        ("ci", "pass", "⏺ CI passed"),
        ("some_check", "pass", "⏺ Some check passed"),
    ],
)
def test_check_gates(
    renderer: LiveRenderer, console: Console, gate: str, result: str, line: str
) -> None:
    renderer({"kind": "gate", "gate": gate, "result": result})
    assert lines_of(console) == [line]


@pytest.mark.parametrize(
    ("gate", "result", "line"),
    [
        ("spec", "approve", "⏺ Spec gate · approved"),
        ("test", "reject", "⏺ Test gate · sent back"),
        ("review", "redo", "⏺ Review gate · sent back"),
        ("limit_hit", "retry", "⏺ Limit hit · retry"),
        ("diff", "stop", "⏺ Diff gate · stopped"),
        ("ci", "approve", "⏺ CI gate · approved"),
        ("custom", "approve", "⏺ Custom · approved"),
        ("spec", "whatever", "⏺ Spec gate · whatever"),
    ],
)
def test_human_decisions(
    renderer: LiveRenderer, console: Console, gate: str, result: str, line: str
) -> None:
    renderer({"kind": "gate", "gate": gate, "result": result})
    assert lines_of(console) == [line]


def test_decision_shows_redo_feedback(renderer: LiveRenderer, console: Console) -> None:
    renderer(
        {
            "kind": "gate",
            "gate": "review",
            "result": "reject",
            "redo": ["handle empty input", "add a test"],
        }
    )
    assert lines_of(console) == [
        "⏺ Review gate · sent back",
        "  ⎿  handle empty input",
        "  ⎿  add a test",
    ]


# --- spinner ---------------------------------------------------------------


def live_console() -> Console:
    return Console(file=io.StringIO(), width=100, force_terminal=True)


def test_no_live_when_spinner_disabled(renderer: LiveRenderer) -> None:
    renderer(model("coder"))
    assert renderer._live is None


def test_spinner_text_shows_agent_elapsed_calls_and_tokens() -> None:
    clock = FakeClock(100.0)
    renderer = LiveRenderer(make_console(), spinner=False, clock=clock)
    renderer(model("coder", tokens_in=30_000, tokens_out=200))
    clock.now = 145.0
    renderer(model("coder", tokens_in=4_500, tokens_out=1_000))
    text = renderer._spinner_text().plain
    assert text[2:] == "CODER working… (45s · 2 calls · ↑ 34.5k ↓ 1.2k)"
    assert text[0] in SPINNER_FRAMES


def test_spinner_text_singular_call_and_helper_name() -> None:
    clock = FakeClock()
    renderer = LiveRenderer(make_console(), spinner=False, clock=clock)
    renderer({"kind": "model", "helper": "reflect", "ok": True, "tokens_in": 950})
    assert renderer._spinner_text().plain[2:] == (
        "reflect working… (0s · 1 call · ↑ 950 ↓ 0)"
    )


def test_spinner_frame_cycles_with_the_clock() -> None:
    clock = FakeClock(0.0)
    renderer = LiveRenderer(make_console(), spinner=False, clock=clock)
    renderer(model("coder"))
    frames = set()
    for step in range(len(SPINNER_FRAMES)):
        clock.now = step / 8
        frames.add(renderer._spinner_text().plain[0])
    assert frames == set(SPINNER_FRAMES)


def test_spinner_stats_reset_when_agent_changes() -> None:
    clock = FakeClock(0.0)
    renderer = LiveRenderer(make_console(), spinner=False, clock=clock)
    renderer(model("coder", tokens_in=1000))
    clock.now = 60.0
    renderer(model("reviewer", tokens_in=7))
    assert renderer._spinner_text().plain[2:] == (
        "REVIEWER working… (0s · 1 call · ↑ 7 ↓ 0)"
    )


def test_spinner_text_is_empty_without_an_active_agent() -> None:
    renderer = LiveRenderer(make_console(), spinner=False)
    assert renderer._spinner_text().plain == ""


def test_live_starts_on_work_and_stops_at_gate_and_run_end() -> None:
    renderer = LiveRenderer(live_console(), clock=FakeClock())
    renderer(model("coder"))
    live = renderer._live
    assert live is not None and live.is_started
    renderer(tool())
    assert renderer._live is live  # one Live, not one per event
    renderer({"kind": "gate", "gate": "review", "result": "waiting"})
    assert renderer._live is None and not live.is_started
    renderer(model("reviewer"))
    assert renderer._live is not None
    renderer({"kind": "run_end", "outcome": "done", "ms": 1000})
    assert renderer._live is None


def test_pause_stops_and_next_event_restarts() -> None:
    renderer = LiveRenderer(live_console(), clock=FakeClock())
    renderer(model("coder"))
    renderer.pause()
    assert renderer._live is None
    renderer.pause()  # idempotent
    renderer(tool())
    assert renderer._live is not None
    renderer.close()
    assert renderer._live is None
    renderer.close()


def test_lines_print_above_the_live_spinner() -> None:
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=True, color_system=None)
    renderer = LiveRenderer(console, clock=FakeClock())
    renderer(model("coder"))
    renderer(tool("read_file", path="a.py", ms=3))
    renderer.close()
    assert "⏺ CODER" in out.getvalue()
    assert "Read(a.py) · 3ms" in out.getvalue()


# --- gate panel ------------------------------------------------------------


def gate_output(gate: str, detail: str, question: str = "Approve?") -> str:
    console = make_console()
    render_gate(console, gate=gate, question=question, detail=detail)
    return console.export_text()


def test_gate_panel_is_rounded_with_title_and_no_question() -> None:
    text = gate_output("spec", "hello", question="Approve this spec?")
    rows = text.splitlines()
    assert rows[0].startswith("╭─ Spec gate ")
    assert rows[0].endswith("╮")
    assert rows[-1].startswith("╰") and rows[-1].endswith("╯")
    assert "hello" in text
    assert "Approve this spec?" not in text


@pytest.mark.parametrize(
    ("gate", "title"),
    [
        ("spec", "Spec gate"),
        ("test", "Test gate"),
        ("review", "Review gate"),
        ("ci", "CI gate"),
        ("diff", "Diff gate"),
        ("limit_hit", "Limit hit"),
        ("custom_check", "Custom check"),
    ],
)
def test_gate_titles(gate: str, title: str) -> None:
    assert f" {title} " in gate_output(gate, "x").splitlines()[0]


def test_agent_brief_is_rendered_as_markdown() -> None:
    text = gate_output("spec", "## Agent Brief\n\n- **bold** item\n")
    assert "**" not in text
    assert "Agent Brief" in text
    assert "• bold item" in text


def test_review_with_pr_prefix_is_rendered_as_markdown() -> None:
    text = gate_output("review", "PR 12\n\n## Review\n\n- `code` ok\n")
    assert "`" not in text
    assert "PR 12" in text


def test_plain_detail_is_not_treated_as_markdown() -> None:
    text = gate_output("diff", "- **keep** stars")
    assert "**keep**" in text


@pytest.mark.parametrize(
    ("line", "style"),
    [
        ("diff --git a/x b/x", f"bold {DIM}"),
        ("--- a/x", f"bold {DIM}"),
        ("+++ b/x", f"bold {DIM}"),
        ("@@ -1 +1 @@", f"bold {DIM}"),
        ("+added", OK),
        ("-removed", ERROR),
        ("## Heading", f"bold {ACCENT}"),
        ("FAILED tests/test_x.py", ERROR),
        ("RuntimeError: boom", ERROR),
        ("?? new.py", WARN),
        (" M changed.py", WARN),
        ("plain", ""),
    ],
)
def test_detail_lines_are_styled_by_kind(line: str, style: str) -> None:
    text = style_detail(["first", line])
    assert [span.style for span in text.spans if span.start > 5] == ([style] if style else [])


def test_style_detail_joins_lines_with_newlines() -> None:
    assert style_detail(["a", "b"]).plain == "a\nb"
    assert style_detail([]).plain == ""


def test_body_is_capped_at_300_lines() -> None:
    detail = "\n".join(f"line {i}" for i in range(350))
    text = gate_output("diff", detail)
    assert "line 299" in text
    assert "line 300" not in text
    assert "… 50 more lines" in text


def test_body_at_the_cap_has_no_more_lines_note() -> None:
    text = gate_output("diff", "\n".join(f"line {i}" for i in range(300)))
    assert "more lines" not in text


def test_long_markdown_is_capped_too() -> None:
    detail = "## Review\n\n" + "\n\n".join(f"para {i}" for i in range(400))
    assert "more lines" in gate_output("review", detail)


def test_empty_detail_renders_an_empty_panel() -> None:
    assert gate_output("spec", "").splitlines()[0].startswith("╭─ Spec gate")
