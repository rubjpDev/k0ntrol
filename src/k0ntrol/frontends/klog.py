"""`/klog`: a local web page over k0's event log.

`read_events` loads `.k0-mem/events.jsonl` and any rotated
`events.*.jsonl` siblings. `build_data` adds the aggregation (runs,
totals, charts) so the page's JS only renders and filters. `write_page`
embeds the data in one self-contained HTML file at
`.k0-mem/klog/index.html`, and `Klog` serves that single file on
127.0.0.1 and opens it in the browser.

Cost is the logged `cost_usd`; when a call has none it is estimated from
`k0ntrol.pricing` and flagged `estimated`.
"""

import json
import math
import threading
import webbrowser
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from k0ntrol.pricing import estimate_cost

HOST = "127.0.0.1"
PORT = 6767
TEMPLATE = Path(__file__).with_name("klog.html")
PLACEHOLDER = "__K0_DATA__"
LOCAL_HOSTS = frozenset({HOST, "localhost"})
PAGE_PATHS = frozenset({"/", "/index.html"})

APPROVED = frozenset({"approve", "pass"})
SENT_BACK = frozenset({"reject", "retry", "stop", "fail"})

# What the browser may do with the page: run its own inline code, nothing else.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'"
)


# -- reading ---------------------------------------------------------------


def read_events(root: Path) -> list[dict]:
    """Every event of the project, oldest first. Malformed lines are skipped."""
    mem = root / ".k0-mem"
    files = [mem / "events.jsonl", *sorted(mem.glob("events.*.jsonl"))]
    events = [event for path in files for event in _read_file(path)]
    return sorted(events, key=lambda event: str(event.get("ts", "")))


def _read_file(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [event for event in map(_parse_line, lines) if event is not None]


def _parse_line(line: str) -> dict | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


# -- aggregation -----------------------------------------------------------


def build_data(events: list[dict]) -> dict:
    """The page's data: annotated events, one row per run or chat, analytics."""
    annotated = [_annotate(event) for event in events]
    runs = _runs(annotated)
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": annotated,
        "runs": runs,
        "analytics": _analytics(annotated, runs),
    }


def _number(value: Any) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if ok else 0.0


def _annotate(event: dict) -> dict:
    """The event plus `cost`/`estimated` (model calls) and `flag` (failures)."""
    out = dict(event)
    if event.get("kind") == "model":
        cost, estimated = _model_cost(event)
        if cost is not None:
            out["cost"], out["estimated"] = cost, estimated
    flag = _flag(event)
    if flag:
        out["flag"] = flag
    return out


def _model_cost(event: dict) -> tuple[float | None, bool]:
    logged = event.get("cost_usd")
    if isinstance(logged, (int, float)) and not isinstance(logged, bool):
        return float(logged), False
    tokens_in, tokens_out = int(_number(event.get("tokens_in"))), int(_number(event.get("tokens_out")))
    estimate = estimate_cost(event.get("provider"), event.get("model"), tokens_in, tokens_out)
    return estimate, estimate is not None


_ALSO_BAD: dict[str, Callable[[dict], bool]] = {
    "gate": lambda event: event.get("result") == "fail",
    "run_end": lambda event: event.get("outcome") != "done",
}


def _flag(event: dict) -> str | None:
    if event.get("denied"):
        return "denied"
    also_bad = _ALSO_BAD.get(str(event.get("kind")), lambda _event: False)
    return "fail" if event.get("ok") is False or also_bad(event) else None


def _parse_ts(ts: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None


def _runs(events: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        key = event.get("run") or event.get("chat")
        if key:
            groups[str(key)].append(event)
    rows = [_run_row(key, group) for key, group in groups.items()]
    return sorted(rows, key=lambda row: row["start"] or "", reverse=True)


def _first(events: Iterable[dict], kind: str) -> dict:
    return next((event for event in events if event.get("kind") == kind), {})


def _usage(models: list[dict]) -> dict:
    """Model calls, tokens and cost over `models` (`cost` is None when no call is priced)."""
    costs = [model["cost"] for model in models if "cost" in model]
    return {
        "model_calls": len(models),
        "tokens_in": int(sum(_number(model.get("tokens_in")) for model in models)),
        "tokens_out": int(sum(_number(model.get("tokens_out")) for model in models)),
        "cost": sum(costs) if costs else None,
        "cost_estimated": any(model.get("estimated") for model in models),
        "unpriced_calls": len(models) - len(costs),
    }


def _failures(events: list[dict]) -> int:
    return sum(1 for event in events if event.get("flag"))


def _run_row(key: str, events: list[dict]) -> dict:
    start, end = _first(events, "run_start"), _first(events, "run_end")
    is_chat = not events[0].get("run")
    models = [event for event in events if event.get("kind") == "model"]
    row = {
        "id": key,
        "kind": start.get("run_kind") or ("chat" if is_chat else "unknown"),
        "work_item": start.get("work_item"),
        "branch": start.get("branch"),
        "session": start.get("session"),
        "start": events[0].get("ts"),
        "duration_ms": _duration_ms(events, end),
        "outcome": end.get("outcome") or (None if is_chat else "running"),
        "failures": _failures(events),
        "events": len(events),
    }
    return row | _usage(models)


def _duration_ms(events: list[dict], end: dict) -> int | None:
    if isinstance(end.get("ms"), (int, float)):
        return int(end["ms"])
    first, last = _parse_ts(events[0].get("ts")), _parse_ts(events[-1].get("ts"))
    if first is None or last is None:
        return None
    return int((last - first).total_seconds() * 1000)


def _analytics(events: list[dict], runs: list[dict]) -> dict:
    models = [event for event in events if event.get("kind") == "model"]
    return {
        "totals": _totals(events, models, runs),
        "by_agent": _by_agent(models),
        "by_day": _by_day(models),
        "latency": _latency(models),
        "tools": _tools(events),
        "gates": _gates(events),
        "outcomes": dict(Counter(row["outcome"] for row in runs if row["kind"] != "chat")),
    }


def _totals(events: list[dict], models: list[dict], runs: list[dict]) -> dict:
    usage = _usage(models)
    usage["cost"] = usage["cost"] or 0.0
    return usage | {
        "runs": sum(1 for row in runs if row["kind"] != "chat"),
        "chats": sum(1 for row in runs if row["kind"] == "chat"),
        "failures": _failures(events),
    }


def _by_agent(models: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for model in models:
        name = str(model.get("agent") or model.get("helper") or "unknown")
        row = rows.setdefault(
            name,
            {"name": name, "helper": not model.get("agent"), "calls": 0, "cost": 0.0, "tokens_in": 0, "tokens_out": 0},
        )
        row["calls"] += 1
        row["cost"] += model.get("cost", 0.0)
        row["tokens_in"] += int(_number(model.get("tokens_in")))
        row["tokens_out"] += int(_number(model.get("tokens_out")))
    return sorted(rows.values(), key=lambda row: (-row["cost"], row["name"]))


def _by_day(models: list[dict]) -> list[dict]:
    """Spend per UTC day, split by agent and by model for the spend board."""
    days: dict[str, dict] = {}
    for model in models:
        day = str(model.get("ts", ""))[:10]
        if len(day) == 10:
            _add_to_day(days.setdefault(day, _empty_day(day)), model)
    return [days[day] for day in sorted(days)]


def _empty_day(day: str) -> dict:
    return {"day": day, "cost": 0.0, "calls": 0, "agents": defaultdict(float), "models": defaultdict(float)}


def _add_to_day(row: dict, model: dict) -> None:
    cost = model.get("cost", 0.0)
    row["cost"] += cost
    row["calls"] += 1
    row["agents"][str(model.get("agent") or model.get("helper") or "unknown")] += cost
    row["models"][str(model.get("model") or "unknown")] += cost


def _percentile(sorted_values: list[float], percent: int) -> float:
    """Nearest-rank percentile of an ascending, non-empty list."""
    return sorted_values[max(math.ceil(percent / 100 * len(sorted_values)), 1) - 1]


def _latency(models: list[dict]) -> list[dict]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for model in models:
        if model.get("ok") is not False and isinstance(model.get("ms"), (int, float)):
            by_model[str(model.get("model") or "unknown")].append(model["ms"])
    return [
        {"model": name, "calls": len(ms), "p50": _percentile(sorted(ms), 50), "p95": _percentile(sorted(ms), 95)}
        for name, ms in sorted(by_model.items())
    ]


def _tools(events: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for event in events:
        if event.get("kind") != "tool":
            continue
        name = str(event.get("tool") or "unknown")
        row = rows.setdefault(name, {"tool": name, "calls": 0, "failed": 0, "denied": 0})
        row["calls"] += 1
        flag = event.get("flag")
        if flag:
            row["denied" if flag == "denied" else "failed"] += 1
    return sorted(rows.values(), key=lambda row: (-row["calls"], row["tool"]))


def _gates(events: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for event in events:
        result = event.get("result")
        if event.get("kind") != "gate" or result not in APPROVED | SENT_BACK:
            continue
        row = rows.setdefault(str(event.get("gate")), {"gate": str(event.get("gate")), "approved": 0, "sent_back": 0})
        row["approved" if result in APPROVED else "sent_back"] += 1
    return sorted(rows.values(), key=lambda row: row["gate"])


# -- the page --------------------------------------------------------------


def render_page(data: dict) -> str:
    """The HTML with `data` embedded; `<` is escaped so no content can end the script block."""
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload, 1)


def write_page(root: Path) -> Path:
    """Create or overwrite `.k0-mem/klog/index.html` from the current log."""
    page = root / ".k0-mem" / "klog" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render_page(build_data(read_events(root))), encoding="utf-8")
    return page


# -- the server ------------------------------------------------------------


def _is_local_host(host_header: str) -> bool:
    name = host_header.rpartition(":")[0] or host_header
    return name in LOCAL_HOSTS


class _PageHandler(BaseHTTPRequestHandler):
    """Serves one file, at `/` and `/index.html`. Nothing else."""

    page: Path

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        if not _is_local_host(self.headers.get("Host", "")):
            return self._reply(403, b"")
        if self.path.partition("?")[0] not in PAGE_PATHS:
            return self._reply(404, b"")
        try:
            body = self.page.read_bytes()
        except OSError:
            return self._reply(404, b"")
        self._reply(200, body, "text/html; charset=utf-8")

    def _reply(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(status)
        for name, value in (
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
            ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
        ):
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet: the REPL owns the terminal."""


def _bind(page: Path, port: int) -> ThreadingHTTPServer:
    handler = type("PageHandler", (_PageHandler,), {"page": page})
    try:
        return ThreadingHTTPServer((HOST, port), handler)
    except OSError:
        return ThreadingHTTPServer((HOST, 0), handler)


class Klog:
    """Writes the page and serves it: one server per instance, started on
    the first `open` and reused after. `opener` and `port` are injectable
    so tests never open a browser or hold a fixed port."""

    def __init__(self, *, opener: Callable[[str], Any] = webbrowser.open, port: int = PORT) -> None:
        self._opener = opener
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Host and port the running server listens on."""
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return str(host), port

    def open(self, root: Path) -> str:
        """Regenerate the page, make sure it is served, open it; return its URL."""
        page = write_page(root)
        if self._server is None:
            self._server = _bind(page, self._port)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
        url = f"http://{HOST}:{self.address[1]}/"
        self._opener(url)
        return url

    def close(self) -> None:
        """Stop the server (tests; the daemon thread dies with k0 otherwise)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
