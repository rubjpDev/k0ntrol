"""``fetch_url``: read one public web page as untrusted text (ADR 0013).

Only PLANNER and RESEARCHER get this tool. It speaks HTTP(S) through
``k0ntrol.tools.net`` and refuses anything that is not a public address:
unsafe schemes, hosts whose DNS answers include a loopback, private or
link-local address (checked again after every redirect), oversized bodies
and slow servers are all bounded. Everything it returns is marked
untrusted, and its event records the host only — never the URL, query or
page.
"""

import codecs
import http.client
import ipaddress
from functools import partial
from urllib.parse import quote, urljoin, urlsplit

from langchain_core.tools import BaseTool, tool

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.net import (
    FetchTarget,
    HttpResponse,
    Resolver,
    Transport,
    resolve_addresses,
    socket_transport,
)

MAX_BYTES = 200_000
TIMEOUT = 15.0
MAX_REDIRECTS = 5
_SCHEMES = ("http", "https")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_PATH_SAFE = "/%!$&'()*+,;=:@"
_QUERY_SAFE = "%/?:@!$&'()*+,;="


def make_fetch_url(
    *,
    resolve: Resolver | None = None,
    transport: Transport | None = None,
    max_bytes: int = MAX_BYTES,
    timeout: float = TIMEOUT,
    max_redirects: int = MAX_REDIRECTS,
) -> BaseTool:
    """Build the ``fetch_url`` tool with injectable DNS and HTTP."""
    resolver = resolve or partial(resolve_addresses, timeout=timeout)
    transport = transport or socket_transport

    @tool
    def fetch_url(url: str) -> str:
        """Fetch one web page over HTTP(S); returns its text, marked untrusted."""
        return _fetch_page(
            url,
            resolver,
            transport,
            timeout=timeout,
            max_bytes=max_bytes,
            max_redirects=max_redirects,
        )

    return fetch_url


def _fetch_page(
    url: str,
    resolver: Resolver,
    transport: Transport,
    *,
    timeout: float,
    max_bytes: int,
    max_redirects: int,
) -> str:
    current = url
    redirects = 0
    while True:
        target = _target(current)
        addresses = _addresses(target.host, resolver)
        page = _fetch(transport, target, addresses, timeout, max_bytes)
        location = _redirect_location(page)
        if location is None:
            return _render(target, page, max_bytes)
        redirects += 1
        if redirects > max_redirects:
            raise ToolError(f"too many redirects (limit {max_redirects})")
        current = urljoin(current, location)


def _target(url: str) -> FetchTarget:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise ToolError("invalid URL") from error
    scheme = parts.scheme.lower()
    if scheme not in _SCHEMES:
        raise ToolError(
            f"unsupported URL scheme: {scheme or 'none'}", denied="network_guard"
        )
    host = parts.hostname
    if not host:
        raise ToolError("URL has no host")
    return FetchTarget(
        scheme=scheme,
        host=host,
        port=port if port is not None else (443 if scheme == "https" else 80),
        path=quote(parts.path or "/", safe=_PATH_SAFE),
        query=quote(parts.query, safe=_QUERY_SAFE),
    )


def _fetch(
    transport: Transport,
    target: FetchTarget,
    addresses: tuple[str, ...],
    timeout: float,
    max_bytes: int,
) -> HttpResponse:
    try:
        return transport(target, addresses, timeout=timeout, max_bytes=max_bytes)
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise ToolError(
            f"fetch failed for {target.host}: {type(error).__name__}"
        ) from error


def _redirect_location(page: HttpResponse) -> str | None:
    if page.status not in _REDIRECT_STATUSES:
        return None
    return page.headers.get("location") or None


def _addresses(host: str, resolver: Resolver) -> tuple[str, ...]:
    """The host's addresses, refused when any answer is not public."""
    try:
        answers = resolver(host)
    except OSError as error:
        raise ToolError(f"cannot resolve {host}") from error
    addresses = []
    for answer in answers:
        if not _public_ip(answer):
            raise ToolError(
                f"{host} resolves to a non-public address", denied="network_guard"
            )
        addresses.append(answer)
    if not addresses:
        raise ToolError(f"cannot resolve {host}")
    return tuple(addresses)


def _public_ip(answer: str) -> bool:
    try:
        ip = ipaddress.ip_address(answer)
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


def _render(target: FetchTarget, page: HttpResponse, max_bytes: int) -> str:
    body = page.body[:max_bytes]
    truncated = page.truncated or len(page.body) > max_bytes
    shown = body.decode(_charset(page), errors="replace")
    lines = [
        f"UNTRUSTED WEB PAGE {target.scheme}://{target.host}{target.path}",
        f"HTTP {page.status}",
        "",
        shown,
    ]
    if truncated:
        lines.append(f"... truncated at {max_bytes} bytes")
    return "\n".join(lines)


def _charset(page: HttpResponse) -> str:
    content_type = page.headers.get("content-type", "")
    for param in content_type.split(";")[1:]:
        name, _, value = param.partition("=")
        name = name.strip().lower()
        value = value.strip().strip('"')
        if name == "charset" and value:
            return _known_codec(value)
    return "utf-8"


def _known_codec(name: str) -> str:
    try:
        codecs.lookup(name)
    except LookupError:
        return "utf-8"
    return name
