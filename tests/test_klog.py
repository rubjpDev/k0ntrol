import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from k0ntrol.frontends import klog
from k0ntrol.frontends.klog import Klog, build_data, read_events, render_page, write_page
from k0ntrol.frontends.session import run_session
from k0ntrol.frontends.terminal import COMMANDS

RUN = "a" * 32
CHAT = "b" * 32
DATA_OPEN = '<script type="application/json" id="k0-data">'


def line(**event) -> str:
    return json.dumps(event)


def seed(root: Path, *lines: str, name: str = "events.jsonl") -> None:
    mem = root / ".k0-mem"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def model(ts: str, **fields) -> dict:
    base = {"ts": ts, "kind": "model", "run": RUN, "agent": "coder", "provider": "amazon_bedrock",
            "model": "qwen.qwen3-next-80b-a3b", "ms": 1000, "tokens_in": 1_000_000, "tokens_out": 0, "ok": True}
    return {**base, **fields}


def fetch(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, "", ""


# -- reading ---------------------------------------------------------------


def test_read_events_merges_rotated_files_in_time_order_and_skips_garbage(tmp_path):
    seed(tmp_path, line(ts="2026-11-02T00:00:00.000Z", kind="tool"), "not json", "[1, 2]", "",
         line(ts="2026-11-03T00:00:00.000Z", kind="tool"))
    seed(tmp_path, line(ts="2026-11-01T00:00:00.000Z", kind="model"), name="events.1.jsonl")
    seed(tmp_path, line(ts="2026-10-31T00:00:00.000Z", kind="gate"), name="events.2026-10-31.jsonl")
    (tmp_path / ".k0-mem" / "events.bak").write_text(line(ts="2026-01-01", kind="tool"))
    assert [e["kind"] for e in read_events(tmp_path)] == ["gate", "model", "tool", "tool"]


def test_read_events_without_a_log_is_empty(tmp_path):
    assert read_events(tmp_path) == []


# -- runs ------------------------------------------------------------------


def test_runs_have_one_row_per_run_or_chat_with_totals():
    events = [
        {"ts": "2026-11-09T10:00:00.000Z", "kind": "run_start", "run": RUN, "run_kind": "fast",
         "work_item": "#12", "branch": "k0/12", "session": "s1"},
        model("2026-11-09T10:00:01.000Z", tokens_in=100, tokens_out=50, cost_usd=0.5),
        model("2026-11-09T10:00:02.000Z", tokens_in=10, tokens_out=5, ok=False, err="boom"),
        {"ts": "2026-11-09T10:00:03.000Z", "kind": "run_end", "run": RUN, "outcome": "done", "ms": 3000},
        {"ts": "2026-11-09T11:00:00.000Z", "kind": "model", "chat": CHAT, "agent": "chatty",
         "provider": "ollama", "model": "m", "ms": 5, "tokens_in": 1, "tokens_out": 2, "ok": True},
        {"ts": "2026-11-09T11:00:02.500Z", "kind": "tool", "chat": CHAT, "tool": "read_file", "ok": True},
    ]
    runs = {r["id"]: r for r in build_data(events)["runs"]}
    run, chat = runs[RUN], runs[CHAT]
    assert (run["kind"], run["work_item"], run["branch"], run["session"]) == ("fast", "#12", "k0/12", "s1")
    assert (run["outcome"], run["duration_ms"], run["model_calls"]) == ("done", 3000, 2)
    assert (run["tokens_in"], run["tokens_out"], run["failures"]) == (110, 55, 1)
    assert run["cost"] == pytest.approx(0.5 + 10 * 0.15 / 1e6 + 5 * 1.20 / 1e6)
    assert run["cost_estimated"] is True
    assert (chat["kind"], chat["outcome"], chat["duration_ms"], chat["cost"]) == ("chat", None, 2500, 0.0)


def test_a_run_without_run_end_is_running_and_without_start_is_unknown():
    runs = build_data([model("2026-11-09T10:00:00.000Z")])["runs"]
    assert (runs[0]["kind"], runs[0]["outcome"]) == ("unknown", "running")


# -- analytics -------------------------------------------------------------


def analytics(events: list[dict]) -> dict:
    return build_data(events)["analytics"]


def test_cost_prefers_the_logged_value_and_falls_back_to_pricing():
    data = build_data([
        model("2026-11-09T10:00:00.000Z", cost_usd=2.0),
        model("2026-11-09T10:00:01.000Z"),
        model("2026-11-09T10:00:02.000Z", model="unknown-model"),
    ])
    totals = data["analytics"]["totals"]
    assert totals["cost"] == pytest.approx(2.15)
    assert (totals["cost_estimated"], totals["unpriced_calls"], totals["model_calls"]) == (True, 1, 3)
    assert [e.get("cost") for e in data["events"]] == [2.0, pytest.approx(0.15), None]


def test_totals_count_runs_not_chats_and_sum_tokens():
    totals = analytics([
        {"ts": "2026-11-09T10:00:00.000Z", "kind": "run_start", "run": RUN, "run_kind": "full"},
        model("2026-11-09T10:00:01.000Z", tokens_in=3, tokens_out=4, cost_usd=0),
        model("2026-11-09T10:00:02.000Z", chat=CHAT, run=None, tokens_in=1, tokens_out=1, cost_usd=0),
    ])["totals"]
    assert (totals["runs"], totals["tokens_in"], totals["tokens_out"]) == (1, 4, 5)
    assert totals["cost_estimated"] is False


def test_by_agent_and_by_day_group_cost_and_tokens():
    result = analytics([
        model("2026-11-08T23:00:00.000Z", cost_usd=1.0, tokens_in=10, tokens_out=1),
        model("2026-11-09T01:00:00.000Z", cost_usd=2.0, tokens_in=20, tokens_out=2),
        model("2026-11-09T02:00:00.000Z", cost_usd=4.0, agent=None, helper="router", tokens_in=1, tokens_out=1),
    ])
    agents = {a["name"]: a for a in result["by_agent"]}
    assert agents["coder"]["cost"] == 3.0 and agents["coder"]["tokens_in"] == 30
    assert agents["router"]["helper"] is True and agents["coder"]["helper"] is False
    assert [(d["day"], d["cost"], d["calls"]) for d in result["by_day"]] == [
        ("2026-11-08", 1.0, 1), ("2026-11-09", 6.0, 2)
    ]


def test_each_day_splits_its_cost_by_agent_and_by_model():
    result = analytics([
        model("2026-11-09T01:00:00.000Z", cost_usd=2.0),
        model("2026-11-09T02:00:00.000Z", cost_usd=0.5, model="claude-opus-5-5"),
        model("2026-11-09T03:00:00.000Z", cost_usd=4.0, agent=None, helper="router"),
    ])
    (day,) = result["by_day"]
    assert day["agents"] == {"coder": 2.5, "router": 4.0}
    assert day["models"] == {"qwen.qwen3-next-80b-a3b": 6.0, "claude-opus-5-5": 0.5}


def test_the_page_opens_dark_with_a_theme_switch_and_a_spend_board():
    page = render_page(build_data([]))
    assert '<html lang="en" data-theme="dark">' in page
    assert 'id="theme-toggle"' in page
    assert 'data-tab="spend"' in page and 'id="spend"' in page


def test_latency_percentiles_per_model_use_successful_calls_only():
    events = [model(f"2026-11-09T10:00:{i:02d}.000Z", ms=ms) for i, ms in enumerate(range(100, 1100, 100))]
    events.append(model("2026-11-09T10:01:00.000Z", ms=99999, ok=False))
    (row,) = analytics(events)["latency"]
    assert row["model"] == "qwen.qwen3-next-80b-a3b"
    assert (row["p50"], row["p95"], row["calls"]) == (500, 1000, 10)


def test_tools_count_calls_failures_and_denials_separately():
    tool = {"kind": "tool", "run": RUN}
    (row,) = analytics([
        {**tool, "ts": "2026-11-09T10:00:00.000Z", "tool": "write_file", "ok": True},
        {**tool, "ts": "2026-11-09T10:00:01.000Z", "tool": "write_file", "ok": False, "err": "x"},
        {**tool, "ts": "2026-11-09T10:00:02.000Z", "tool": "write_file", "ok": False, "denied": "path_guard"},
    ])["tools"]
    assert row == {"tool": "write_file", "calls": 3, "failed": 1, "denied": 1}


def test_gates_split_approved_from_sent_back_and_ignore_waiting():
    gate = {"kind": "gate", "run": RUN, "ts": "2026-11-09T10:00:00.000Z"}
    rows = analytics([
        {**gate, "gate": "spec", "result": "waiting"},
        {**gate, "gate": "spec", "result": "approve"},
        {**gate, "gate": "spec", "result": "reject"},
        {**gate, "gate": "validator", "result": "pass"},
        {**gate, "gate": "validator", "result": "fail"},
        {**gate, "gate": "validator", "result": "fail"},
    ])["gates"]
    assert {r["gate"]: (r["approved"], r["sent_back"]) for r in rows} == {"spec": (1, 1), "validator": (1, 2)}


def test_run_outcomes_count_done_stuck_and_running():
    end = {"kind": "run_end", "ts": "2026-11-09T10:00:00.000Z", "ms": 1}
    start = {"kind": "run_start", "ts": "2026-11-09T10:00:00.000Z", "run_kind": "fast"}
    outcomes = analytics([
        {**start, "run": "r1"}, {**end, "run": "r1", "outcome": "done"},
        {**start, "run": "r2"}, {**end, "run": "r2", "outcome": "stuck"},
        {**start, "run": "r3"}, {**end, "run": "r3", "outcome": "stuck"},
        {**start, "run": "r4"},
    ])["outcomes"]
    assert outcomes == {"done": 1, "stuck": 2, "running": 1}


def test_failures_and_denials_are_flagged_on_the_events():
    denied = {"ts": "2026-11-09T10:00:02.000Z", "kind": "tool", "run": RUN, "tool": "t", "ok": False,
              "denied": "allowlist"}
    flags = [e.get("flag") for e in build_data([
        model("2026-11-09T10:00:00.000Z"),
        model("2026-11-09T10:00:01.000Z", ok=False),
        denied,
        {"ts": "2026-11-09T10:00:03.000Z", "kind": "gate", "run": RUN, "gate": "g", "result": "fail"},
        {"ts": "2026-11-09T10:00:04.000Z", "kind": "run_end", "run": RUN, "outcome": "stuck", "ms": 1},
    ])["events"]]
    assert flags == [None, "fail", "denied", "fail", "fail"]


def test_bad_timestamps_and_odd_values_do_not_break_the_data():
    data = build_data([
        {"ts": "garbage", "kind": "model", "run": RUN, "ms": "x", "tokens_in": "y", "ok": True},
        {"kind": "tool", "run": RUN, "tool": "t"},
        {"ts": "2026-11-09T10:00:00.000Z", "kind": "mystery"},
    ])
    assert data["runs"][0]["duration_ms"] is None


# -- the page --------------------------------------------------------------


def page_json(page: str) -> dict:
    start = page.index(DATA_OPEN) + len(DATA_OPEN)
    return json.loads(page[start:page.index("</script>", start)])


def test_page_embeds_the_data_and_content_cannot_break_out_of_the_script_block():
    evil = "</script><script>alert(1)</script><!--"
    data = build_data([{"ts": "2026-11-09T10:00:00.000Z", "kind": "tool", "run": RUN, "tool": "t",
                        "ok": False, "err": evil}])
    page = render_page(data)
    assert evil not in page
    assert page_json(page)["events"][0]["err"] == evil


def test_page_is_self_contained():
    page = render_page(build_data([]))
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in page
    assert '[data-theme="light"]' in page  # light is one switch away
    for tab in ("Runs", "Spend", "Log", "Analytics"):
        assert tab in page


def test_write_page_creates_then_overwrites_and_touches_no_gitignore(tmp_path):
    seed(tmp_path, line(ts="2026-11-09T10:00:00.000Z", kind="tool", run=RUN, tool="tool_one", ok=True))
    page = write_page(tmp_path)
    assert page == tmp_path / ".k0-mem" / "klog" / "index.html"
    assert "tool_one" in page.read_text(encoding="utf-8")
    seed(tmp_path, line(ts="2026-11-09T10:00:00.000Z", kind="tool", run=RUN, tool="tool_two", ok=True))
    write_page(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "tool_two" in text and "tool_one" not in text
    assert not list(tmp_path.rglob(".gitignore"))


# -- the server ------------------------------------------------------------


@pytest.fixture
def opened():
    return []


@pytest.fixture
def klog_server(opened):
    instance = Klog(opener=opened.append, port=0)
    yield instance
    instance.close()


def test_open_writes_the_page_serves_it_on_loopback_and_opens_the_browser(tmp_path, opened, klog_server):
    seed(tmp_path, line(ts="2026-11-09T10:00:00.000Z", kind="tool", run=RUN, tool="read_file", ok=True))
    url = klog_server.open(tmp_path)
    assert opened == [url]
    assert url.startswith("http://127.0.0.1:")
    status, content_type, body = fetch(url)
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert "read_file" in body
    assert klog_server.address[0] == "127.0.0.1"


def test_only_the_one_page_is_served(tmp_path, klog_server):
    url = klog_server.open(tmp_path)
    for path in ("/.k0-mem/events.jsonl", "/klog/", "/../../etc/passwd", "/index.html/x", "/other"):
        assert fetch(url.rstrip("/") + path)[0] == 404
    assert fetch(url.rstrip("/") + "/?x=1")[0] == 200


def test_a_foreign_host_header_is_refused(tmp_path, klog_server):
    url = klog_server.open(tmp_path)
    request = urllib.request.Request(url, headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    assert refused.value.code == 403


def test_a_deleted_page_is_a_404_until_the_next_klog(tmp_path, klog_server):
    url = klog_server.open(tmp_path)
    write_page(tmp_path).unlink()
    assert fetch(url)[0] == 404
    assert fetch(klog_server.open(tmp_path))[0] == 200


def test_a_second_open_regenerates_the_page_and_reuses_the_server(tmp_path, opened, klog_server):
    seed(tmp_path, line(ts="2026-11-09T10:00:00.000Z", kind="tool", run=RUN, tool="tool_one", ok=True))
    first = klog_server.open(tmp_path)
    seed(tmp_path, line(ts="2026-11-09T10:00:00.000Z", kind="tool", run=RUN, tool="tool_two", ok=True))
    second = klog_server.open(tmp_path)
    assert first == second and opened == [first, second]
    assert "tool_two" in fetch(second)[2]


def test_a_busy_port_falls_back_to_a_free_one(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        instance = Klog(opener=lambda url: None, port=port)
        try:
            url = instance.open(tmp_path)
            assert f":{port}/" not in url
            assert fetch(url)[0] == 200
        finally:
            instance.close()


def test_the_default_port_is_6767_and_the_default_opener_is_the_browser():
    assert klog.PORT == 6767
    assert Klog()._opener is klog.webbrowser.open


def test_close_without_a_server_is_a_no_op():
    Klog().close()


# -- the command -----------------------------------------------------------


def test_klog_is_listed_in_the_commands():
    assert "/klog" in COMMANDS


def test_klog_command_shows_the_url_and_stays_in_the_session(tmp_path):
    asked = []
    lines = iter(["/klog", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines), shown.append, root=tmp_path,
        config={"agents": {"chatty": {"provider": "anthropic", "model": "m"}}},
        klog=lambda root: asked.append(root) or "http://127.0.0.1:6767/",
    )
    assert code == 0 and asked == [tmp_path]
    assert "http://127.0.0.1:6767/" in "\n".join(shown)


def test_klog_command_reports_a_write_failure_and_stays_in_the_session(tmp_path):
    def broken(root):
        raise OSError("read-only file system")

    lines = iter(["/klog", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines), shown.append, root=tmp_path,
        config={"agents": {"chatty": {"provider": "anthropic", "model": "m"}}}, klog=broken,
    )
    assert code == 0
    assert "klog failed: read-only file system" in "\n".join(shown)
