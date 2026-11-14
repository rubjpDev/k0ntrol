"""``fetch_url`` reads one public page as untrusted text, behind SSRF guards.

The seam is the agent tool itself: ``make_fetch_url`` with DNS and HTTP
injected, invoked the way an agent's tool call would be. No live network.
"""

import json
import socket
import threading
import time

import pytest

from k0ntrol.events import EventWriter, K0EventHandler
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.fetch_url import make_fetch_url
from k0ntrol.tools.net import HttpResponse

PUBLIC_IP = "93.184.216.34"


def public_dns(host: str) -> list[str]:
    return [PUBLIC_IP]


def no_transport(*args, **kwargs):
    raise AssertionError("transport must not be called")


def test_fetch_url_returns_a_public_page_as_untrusted_text():
    seen = {}

    def transport(target, addresses, *, timeout, max_bytes):
        seen["target"] = target
        seen["addresses"] = addresses
        return HttpResponse(
            status=200, headers={}, body=b"<h1>Hello page</h1>", truncated=False
        )

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/docs/page"})

    assert "UNTRUSTED" in out
    assert "Hello page" in out
    assert seen["addresses"] == (PUBLIC_IP,)
    assert seen["target"].host == "example.com"
    assert seen["target"].scheme == "https"
    assert seen["target"].port == 443


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "javascript:alert(1)",
        "example.com/no-scheme",
        "//example.com/protocol-relative",
    ],
)
def test_fetch_url_refuses_unsafe_schemes(url):
    tool = make_fetch_url(resolve=public_dns, transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": url})
    assert error.value.denied == "network_guard"


@pytest.mark.parametrize(
    "answer",
    [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_fetch_url_refuses_a_host_resolving_to_a_non_public_address(answer):
    tool = make_fetch_url(resolve=lambda host: [answer], transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://internal.example/"})
    assert error.value.denied == "network_guard"


def test_fetch_url_refuses_when_any_dns_answer_is_non_public():
    tool = make_fetch_url(
        resolve=lambda host: [PUBLIC_IP, "192.168.1.1"], transport=no_transport
    )
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert error.value.denied == "network_guard"


def test_fetch_url_refuses_a_loopback_ip_literal():
    tool = make_fetch_url(resolve=lambda host: [host], transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "http://127.0.0.1:8080/admin"})
    assert error.value.denied == "network_guard"


def _redirecting_transport(pages, targets, resolved_hosts):
    def transport(target, addresses, *, timeout, max_bytes):
        targets.append(target)
        return next(pages)

    def resolve(host):
        resolved_hosts.append(host)
        return public_dns(host)

    return transport, resolve


def test_fetch_url_follows_a_redirect():
    pages = iter(
        [
            HttpResponse(301, {"location": "/next"}, b"", False),
            HttpResponse(200, {}, b"final page", False),
        ]
    )
    targets, resolved_hosts = [], []
    transport, resolve = _redirecting_transport(pages, targets, resolved_hosts)
    tool = make_fetch_url(resolve=resolve, transport=transport, max_redirects=3)

    out = tool.invoke({"url": "https://example.com/start"})

    assert "UNTRUSTED" in out
    assert "final page" in out
    assert [t.selector for t in targets] == ["/start", "/next"]
    assert resolved_hosts == ["example.com", "example.com"]


def test_fetch_url_follows_a_redirect_to_another_public_host():
    pages = iter(
        [
            HttpResponse(302, {"location": "https://cdn.example.net/f?q=2"}, b"", False),
            HttpResponse(200, {}, b"cdn page", False),
        ]
    )
    targets, resolved_hosts = [], []
    transport, resolve = _redirecting_transport(pages, targets, resolved_hosts)
    tool = make_fetch_url(resolve=resolve, transport=transport)

    out = tool.invoke({"url": "https://example.com/start"})

    assert "cdn page" in out
    assert targets[1].host == "cdn.example.net"
    assert targets[1].query == "q=2"
    assert resolved_hosts == ["example.com", "cdn.example.net"]


def test_fetch_url_revalidates_the_target_after_a_redirect():
    pages = iter(
        [HttpResponse(302, {"location": "https://internal.example/x"}, b"", False)]
    )
    targets = []

    def transport(target, addresses, *, timeout, max_bytes):
        targets.append(target)
        return next(pages)

    def resolve(host):
        return {"example.com": [PUBLIC_IP], "internal.example": ["10.0.0.1"]}[host]

    tool = make_fetch_url(resolve=resolve, transport=transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert error.value.denied == "network_guard"
    assert len(targets) == 1


def test_fetch_url_gives_up_after_too_many_redirects():
    calls = []

    def transport(target, addresses, *, timeout, max_bytes):
        calls.append(target)
        return HttpResponse(302, {"location": "/loop"}, b"", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport, max_redirects=2)
    with pytest.raises(ToolError, match="too many redirects"):
        tool.invoke({"url": "https://example.com/"})
    assert len(calls) == 3


def test_fetch_url_cuts_a_body_over_the_cap():
    body = b"a" * 150 + b"ZZZTAIL"

    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(200, {}, body, False)

    tool = make_fetch_url(
        resolve=public_dns, transport=transport, max_bytes=100
    )
    out = tool.invoke({"url": "https://example.com/"})
    assert "a" * 101 not in out
    assert "ZZZTAIL" not in out
    assert "truncated" in out


def test_fetch_url_marks_a_body_the_transport_cut():
    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(200, {}, b"partial", True)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/"})
    assert "partial" in out
    assert "truncated" in out


def test_fetch_url_turns_a_timeout_into_a_safe_tool_error():
    def transport(target, addresses, *, timeout, max_bytes):
        raise TimeoutError("timed out")

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/private?token=sekret"})
    assert "example.com" in str(error.value)
    assert "sekret" not in str(error.value)
    assert "private" not in str(error.value)


def test_fetch_url_turns_a_value_error_into_a_tool_error():
    def transport(target, addresses, *, timeout, max_bytes):
        raise ValueError("malformed response")

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert "example.com" in str(error.value)


def test_fetch_url_turns_a_hanging_dns_lookup_into_a_tool_error(monkeypatch):
    release = threading.Event()

    def sleeper(*args, **kwargs):
        release.wait(10)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", sleeper)
    tool = make_fetch_url(transport=no_transport, timeout=0.05)
    start = time.monotonic()
    try:
        with pytest.raises(ToolError) as error:
            tool.invoke({"url": "https://example.com/private?token=sekret"})
    finally:
        release.set()
    assert time.monotonic() - start < 5
    assert "example.com" in str(error.value)
    assert "sekret" not in str(error.value)
    assert "private" not in str(error.value)


def test_fetch_url_passes_its_bounds_to_the_transport():
    seen = {}

    def transport(target, addresses, *, timeout, max_bytes):
        seen.update(timeout=timeout, max_bytes=max_bytes)
        return HttpResponse(200, {}, b"ok", False)

    tool = make_fetch_url(
        resolve=public_dns, transport=transport, timeout=3.5, max_bytes=64
    )
    tool.invoke({"url": "https://example.com/"})
    assert 0 < seen["timeout"] <= 3.5
    assert seen["max_bytes"] == 64


def test_fetch_url_shares_one_deadline_across_redirects():
    budgets = []

    def transport(target, addresses, *, timeout, max_bytes):
        budgets.append(timeout)
        time.sleep(0.03)
        return HttpResponse(302, {"location": "/again"}, b"", False)

    tool = make_fetch_url(
        resolve=public_dns, transport=transport, timeout=0.1, max_redirects=50
    )
    start = time.monotonic()
    with pytest.raises(ToolError, match="timed out for example.com"):
        tool.invoke({"url": "https://example.com/"})
    assert time.monotonic() - start < 1
    assert len(budgets) < 10
    assert budgets == sorted(budgets, reverse=True)


def test_fetch_url_tool_event_records_only_the_host(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run="r1", mask=[])
    handler = K0EventHandler(writer)

    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(200, {}, b"page", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    tool.invoke(
        {"url": "https://example.com/docs?token=sekret"},
        config={"callbacks": [handler], "metadata": {"k0_agent": "researcher"}},
    )

    (event,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert event["kind"] == "tool"
    assert event["agent"] == "researcher"
    assert event["tool"] == "fetch_url"
    assert event["host"] == "example.com"
    assert event["ok"] is True
    blob = json.dumps(event)
    assert "sekret" not in blob
    assert "docs" not in blob
    assert "https://" not in blob


def test_fetch_url_refusal_event_records_host_denied_and_no_url(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run="r1", mask=[])
    handler = K0EventHandler(writer)
    tool = make_fetch_url(resolve=public_dns, transport=no_transport)

    with pytest.raises(ToolError):
        tool.invoke(
            {"url": "file:///etc/passwd?token=sekret"},
            config={"callbacks": [handler], "metadata": {"k0_agent": "planner"}},
        )

    (event,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert event["ok"] is False
    assert event["denied"] == "network_guard"
    blob = json.dumps(event)
    assert "sekret" not in blob
    assert "passwd" not in blob


def test_fetch_url_reports_a_non_2xx_status_as_a_page():
    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(404, {}, b"nothing here", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/missing"})
    assert "UNTRUSTED" in out
    assert "HTTP 404" in out
    assert "nothing here" in out


def test_fetch_url_passes_the_explicit_port_to_the_transport():
    seen = {}

    def transport(target, addresses, *, timeout, max_bytes):
        seen["target"] = target
        return HttpResponse(200, {}, b"ok", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    tool.invoke({"url": "http://example.com:8080/x"})
    assert seen["target"].port == 8080
    assert seen["target"].scheme == "http"


def test_fetch_url_pins_every_validated_address():
    seen = {}

    def transport(target, addresses, *, timeout, max_bytes):
        seen["addresses"] = addresses
        return HttpResponse(200, {}, b"ok", False)

    ips = [PUBLIC_IP, "1.1.1.1", "2606:4700:4700::1111"]
    tool = make_fetch_url(resolve=lambda host: ips, transport=transport)
    tool.invoke({"url": "https://example.com/"})
    assert seen["addresses"] == tuple(ips)


def test_fetch_url_refuses_a_non_ip_dns_answer():
    tool = make_fetch_url(
        resolve=lambda host: ["not-an-ip"], transport=no_transport
    )
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert error.value.denied == "network_guard"


def test_fetch_url_reports_an_unresolvable_host():
    tool = make_fetch_url(resolve=lambda host: [], transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert error.value.denied is None
    assert "example.com" in str(error.value)


def test_fetch_url_reports_a_dns_failure():
    def resolve(host):
        raise OSError("no such host")

    tool = make_fetch_url(resolve=resolve, transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert "example.com" in str(error.value)


def test_fetch_url_decodes_the_pages_charset():
    body = "café".encode("iso-8859-1")

    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(
            200, {"content-type": "text/html; charset=iso-8859-1"}, body, False
        )

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/"})
    assert "café" in out


def test_fetch_url_reads_a_charset_with_a_spaced_parameter_name():
    body = "café".encode("iso-8859-1")

    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(
            200, {"content-type": "text/html; charset = iso-8859-1"}, body, False
        )

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/"})
    assert "café" in out


def test_fetch_url_falls_back_to_utf8_on_an_unknown_charset():
    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(
            200, {"content-type": "text/html; charset=x-not-a-codec"}, b"plain", False
        )

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    out = tool.invoke({"url": "https://example.com/"})
    assert "plain" in out


def test_fetch_url_encodes_the_request_target():
    seen = {}

    def transport(target, addresses, *, timeout, max_bytes):
        seen["selector"] = target.selector
        return HttpResponse(200, {}, b"ok", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    tool.invoke({"url": "https://example.com/a page?q=two words#frag"})
    assert seen["selector"] == "/a%20page?q=two%20words"


def test_fetch_url_reports_a_malformed_url():
    tool = make_fetch_url(resolve=public_dns, transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "http://example.com:not-a-port/"})
    assert error.value.denied is None


def test_fetch_url_reports_a_url_without_a_host():
    tool = make_fetch_url(resolve=public_dns, transport=no_transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https:///path-only"})
    assert error.value.denied is None


def test_fetch_url_refuses_a_redirect_to_an_unsafe_scheme():
    calls = []

    def transport(target, addresses, *, timeout, max_bytes):
        calls.append(target)
        return HttpResponse(302, {"location": "file:///etc/passwd"}, b"", False)

    tool = make_fetch_url(resolve=public_dns, transport=transport)
    with pytest.raises(ToolError) as error:
        tool.invoke({"url": "https://example.com/"})
    assert error.value.denied == "network_guard"
    assert len(calls) == 1
