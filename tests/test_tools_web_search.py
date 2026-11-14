"""``web_search`` queries the configured search API (ADR 0013).

The tool is tested at the agent-tool seam: called directly, HTTP injected,
credentials saved through ``credentials.py`` like provider keys.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from k0ntrol.config import load_config
from k0ntrol.credentials import load_credentials, save_credential
from k0ntrol.events import EventWriter, K0EventHandler, render_event
from k0ntrol.search import SearchRequest
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.web_search import http_fetch, make_web_search

BRAVE_BODY = {
    "web": {
        "results": [
            {
                "title": "Example Title",
                "url": "https://example.com/page",
                "description": "A snippet about the example.",
            },
            {
                "title": "Second Result",
                "url": "https://other.test/docs",
                "description": "Another snippet.",
            },
        ]
    }
}

RUN = "a" * 32


def brave_fetch(body=BRAVE_BODY):
    calls = []

    def fetch(request):
        calls.append(request)
        return json.dumps(body).encode()

    fetch.calls = calls
    return fetch


def brave_cfg(**search):
    return {"search": {"provider": "brave", **search}}


def environ(**extra):
    return {"BRAVE_SEARCH_API_KEY": "fake-key", **extra}


def make_tool(cfg=None, env=None, fetch=None):
    return make_web_search(
        brave_cfg() if cfg is None else cfg,
        environ=environ() if env is None else env,
        fetch=brave_fetch() if fetch is None else fetch,
    )


def test_missing_key_names_the_variable_to_set():
    tool = make_web_search(brave_cfg(), environ={}, fetch=brave_fetch())
    with pytest.raises(ToolError, match="BRAVE_SEARCH_API_KEY"):
        tool.invoke({"query": "python testing"})


def test_query_returns_titles_urls_and_snippets_marked_untrusted():
    tool = make_tool()
    out = tool.invoke({"query": "python testing"})
    first = out.splitlines()[0]
    assert first.startswith("UNTRUSTED")
    for text in (
        "Example Title",
        "https://example.com/page",
        "A snippet about the example.",
        "Second Result",
        "https://other.test/docs",
        "Another snippet.",
    ):
        assert text in out


def test_the_agent_sees_only_the_query_argument():
    tool = make_tool()
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]


def test_the_request_goes_to_the_brave_endpoint_with_the_key():
    fetch = brave_fetch()
    tool = make_web_search(brave_cfg(), environ=environ(), fetch=fetch)
    tool.invoke({"query": "python testing"})
    (request,) = fetch.calls
    assert isinstance(request, SearchRequest)
    assert request.method == "GET"
    parts = urlsplit(request.url)
    assert parts.scheme == "https"
    assert parts.netloc == "api.search.brave.com"
    assert parts.path == "/res/v1/web/search"
    assert parse_qs(parts.query)["q"] == ["python testing"]
    assert request.headers["X-Subscription-Token"] == "fake-key"


def test_a_missing_search_provider_is_a_tool_error():
    tool = make_tool(cfg={})
    with pytest.raises(ToolError, match="search.provider"):
        tool.invoke({"query": "python testing"})


def test_an_unknown_search_provider_is_a_tool_error():
    tool = make_tool(cfg={"search": {"provider": "google"}})
    with pytest.raises(ToolError, match="google"):
        tool.invoke({"query": "python testing"})


def test_a_non_json_response_is_a_tool_error():
    tool = make_tool(fetch=lambda request: b"not json at all")
    with pytest.raises(ToolError):
        tool.invoke({"query": "python testing"})


def test_an_unexpected_response_shape_is_a_tool_error():
    tool = make_tool(fetch=brave_fetch(body={"unexpected": True}))
    with pytest.raises(ToolError):
        tool.invoke({"query": "python testing"})


def test_a_fetch_failure_is_a_tool_error_that_hides_the_url():
    def boom(request):
        raise OSError("connection refused https://api.search.brave.com/?q=sekret")

    tool = make_tool(fetch=boom)
    with pytest.raises(ToolError) as error:
        tool.invoke({"query": "sekret query"})
    assert "sekret" not in str(error.value)


def test_empty_results_say_so():
    tool = make_tool(fetch=brave_fetch(body={"web": {"results": []}}))
    out = tool.invoke({"query": "nothing matches this"})
    assert out.splitlines()[0].startswith("UNTRUSTED")
    assert "no results" in out.lower()


def test_results_missing_fields_still_render():
    body = {"web": {"results": [{"title": "Only a title"}, "junk", {"url": "https://x.test"}]}}
    tool = make_tool(fetch=brave_fetch(body=body))
    out = tool.invoke({"query": "q"})
    assert "Only a title" in out
    assert "https://x.test" in out


def test_a_saved_key_loads_like_a_provider_key(tmp_path):
    save_credential(tmp_path, "BRAVE_SEARCH_API_KEY", "saved-brave-key")
    env: dict[str, str] = {}
    assert load_credentials(tmp_path, env) == ["BRAVE_SEARCH_API_KEY"]
    tool = make_tool(env=env)
    assert "UNTRUSTED" in tool.invoke({"query": "python testing"})


def test_the_search_provider_comes_from_the_config_file(tmp_path):
    project = tmp_path / "proj"
    (project / ".k0-mem").mkdir(parents=True)
    (project / ".k0-mem" / "config.yaml").write_text("search:\n  provider: brave\n")
    home = tmp_path / "home"
    home.mkdir()
    cfg = load_config(project, home)
    assert cfg["search"]["provider"] == "brave"


def test_the_tool_event_records_the_search_host_not_the_query(tmp_path):
    handler, path = _handler(tmp_path)
    tool = make_tool()
    tool.invoke(
        {"query": "python testing"},
        config={"callbacks": [handler], "metadata": {"k0_agent": "researcher"}},
    )
    (event,) = _events(path)
    assert event["kind"] == "tool"
    assert event["agent"] == "researcher"
    assert event["tool"] == "web_search"
    assert event["host"] == "api.search.brave.com"
    assert event["ok"] is True
    blob = json.dumps(event)
    assert "python testing" not in blob
    assert "q=" not in blob
    assert "api.search.brave.com" in render_event(event)


def test_the_error_event_still_records_the_host_and_hides_the_query(tmp_path):
    handler, path = _handler(tmp_path)
    tool = make_web_search(brave_cfg(), environ={}, fetch=brave_fetch())
    with pytest.raises(ToolError):
        tool.invoke(
            {"query": "sekret query"},
            config={"callbacks": [handler], "metadata": {"k0_agent": "planner"}},
        )
    (event,) = _events(path)
    assert event["host"] == "api.search.brave.com"
    assert event["ok"] is False
    assert "BRAVE_SEARCH_API_KEY" in event["err"]
    blob = json.dumps(event)
    assert "sekret" not in blob


def test_a_string_call_still_records_the_host(tmp_path):
    handler, path = _handler(tmp_path)
    tool = make_tool()
    tool.invoke(
        "python testing",
        config={"callbacks": [handler], "metadata": {"k0_agent": "researcher"}},
    )
    (event,) = _events(path)
    assert event["host"] == "api.search.brave.com"
    assert "python testing" not in json.dumps(event)


def test_ainvoke_records_the_host_too(tmp_path):
    handler, path = _handler(tmp_path)
    tool = make_tool()
    asyncio.run(
        tool.ainvoke(
            {"query": "python testing"},
            config={"callbacks": [handler], "metadata": {"k0_agent": "researcher"}},
        )
    )
    (event,) = _events(path)
    assert event["host"] == "api.search.brave.com"
    assert "python testing" not in json.dumps(event)


def test_a_tool_error_from_fetch_passes_through_unchanged():
    def fetch(request):
        raise ToolError("already a tool error")

    tool = make_tool(fetch=fetch)
    with pytest.raises(ToolError, match="already a tool error"):
        tool.invoke({"query": "python testing"})


def _handler(tmp_path: Path) -> tuple[K0EventHandler, Path]:
    path = tmp_path / ".k0-mem" / "events.jsonl"
    path.parent.mkdir(parents=True)
    return K0EventHandler(EventWriter(path, run=RUN)), path


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- the stdlib transport ---------------------------------------------------


class _SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        if self.path == "/huge":
            body = b"x" * 300_000
        else:
            body = json.dumps(BRAVE_BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _local_fetch(path: str):
    server = HTTPServer(("127.0.0.1", 0), _SearchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def fetch():
        try:
            return http_fetch(
                SearchRequest(
                    method="GET",
                    url=f"http://127.0.0.1:{server.server_port}{path}",
                    headers={"X-Subscription-Token": "fake-key"},
                )
            )
        finally:
            server.shutdown()
            server.server_close()

    return fetch


def test_http_fetch_reads_a_response_over_stdlib_http():
    body = _local_fetch("/res/v1/web/search?q=x")()
    assert json.loads(body)["web"]["results"][0]["title"] == "Example Title"


def test_http_fetch_refuses_a_redirect():
    with pytest.raises(ToolError, match="redirected"):
        _local_fetch("/redirect")()


def test_http_fetch_refuses_an_oversized_response():
    with pytest.raises(ToolError, match="too large"):
        _local_fetch("/huge")()
