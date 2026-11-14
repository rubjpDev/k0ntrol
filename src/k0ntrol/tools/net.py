"""Stdlib HTTP transport for the network tools (ADR 0013).

``socket_transport`` performs one GET against addresses the caller already
resolved and validated: it connects to a pinned address — the name is never
re-resolved between check and connect, so a DNS answer cannot change under
us — keeps the hostname for the ``Host`` header and TLS SNI, verifies the
certificate as usual, and never consults proxy environment variables, so an
ambient ``HTTP_PROXY`` cannot smuggle a request past the guards.
``resolve_addresses`` is the default DNS lookup callers inject around.
"""

import http.client
import socket
import ssl
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FetchTarget:
    """One HTTP(S) request target: where to connect and what to ask for."""

    scheme: str
    host: str
    port: int
    path: str
    query: str = ""

    @property
    def selector(self) -> str:
        """The origin-form request target, ``path?query``."""
        return self.path + (f"?{self.query}" if self.query else "")


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response: status, lowercased headers and a bounded body."""

    status: int
    headers: dict[str, str]
    body: bytes
    truncated: bool = False


class Transport(Protocol):
    """One validated GET over pinned ``addresses`` with bounded waits."""

    def __call__(
        self,
        target: FetchTarget,
        addresses: tuple[str, ...],
        *,
        timeout: float,
        max_bytes: int,
    ) -> HttpResponse: ...


Resolver = Callable[[str], Sequence[str]]
Connect = Callable[..., socket.socket]


def resolve_addresses(host: str, *, timeout: float) -> list[str]:
    """The host's addresses via ``getaddrinfo``, deduplicated and sorted.

    ``getaddrinfo`` has no timeout of its own, so the lookup runs on a
    single-use executor and ``timeout`` bounds the wait; on expiry a
    ``TimeoutError`` surfaces for the caller to turn into a tool error.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        infos = pool.submit(
            socket.getaddrinfo, host, None, type=socket.SOCK_STREAM
        ).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return sorted(
        {info[4][0] for info in infos if isinstance(info[4][0], str)}
    )


def socket_transport(
    target: FetchTarget,
    addresses: tuple[str, ...],
    *,
    timeout: float,
    max_bytes: int,
    connect: Connect = socket.create_connection,
) -> HttpResponse:
    """GET ``target`` through the first reachable pinned address.

    ``connect`` is the socket seam (``socket.create_connection`` in
    production). Raises the last ``OSError``/``HTTPException`` when no
    address answers; the caller turns it into a tool error.
    """
    if not addresses:
        raise ValueError("socket_transport needs at least one address")
    failure: OSError | http.client.HTTPException | None = None
    for address in addresses:
        try:
            return _get_once(
                target, address, timeout=timeout, max_bytes=max_bytes, connect=connect
            )
        except (OSError, http.client.HTTPException) as error:
            failure = error
    assert failure is not None
    raise failure


def _get_once(
    target: FetchTarget,
    address: str,
    *,
    timeout: float,
    max_bytes: int,
    connect: Connect,
) -> HttpResponse:
    conn = _connection(target, address, timeout, connect)
    try:
        conn.request("GET", target.selector, headers=_request_headers())
        response = conn.getresponse()
        body = response.read(max_bytes + 1)
        return HttpResponse(
            status=response.status,
            headers={name.lower(): value for name, value in response.getheaders()},
            body=body[:max_bytes],
            truncated=len(body) > max_bytes,
        )
    finally:
        conn.close()


def _connection(
    target: FetchTarget, address: str, timeout: float, connect: Connect
) -> http.client.HTTPConnection:
    cls = _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
    return cls(target.host, target.port, address, timeout, connect)


def _request_headers() -> dict[str, str]:
    return {
        "Accept": "text/html, text/plain, */*",
        "Accept-Encoding": "identity",
        "User-Agent": "k0ntrol",
        "Connection": "close",
    }


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to a validated address while naming ``host`` in requests."""

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        timeout: float,
        connect: Connect,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address
        self._connect = connect

    def connect(self) -> None:
        self.sock = self._connect((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Like ``_PinnedHTTPConnection``; TLS still verifies ``host``."""

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        timeout: float,
        connect: Connect,
    ) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address
        self._connect = connect
        self._tls_context = context

    def connect(self) -> None:
        self.sock = self._connect((self._address, self.port), self.timeout)
        self.sock = self._tls_context.wrap_socket(self.sock, server_hostname=self.host)
