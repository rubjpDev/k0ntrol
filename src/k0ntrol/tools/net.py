"""Stdlib HTTP transport for the network tools (ADR 0013).

``socket_transport`` performs one GET against addresses the caller already
resolved and validated: it connects to a pinned address — the name is never
re-resolved between check and connect, so a DNS answer cannot change under
us — keeps the hostname for the ``Host`` header and TLS SNI, verifies the
certificate as usual, and never consults proxy environment variables, so an
ambient ``HTTP_PROXY`` cannot smuggle a request past the guards.
``resolve_addresses`` is the default DNS lookup callers inject around.
"""

import contextlib
import http.client
import socket
import ssl
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

DNS_LOOKUP_SLOTS = 4
_dns_slots = threading.BoundedSemaphore(DNS_LOOKUP_SLOTS)


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

    ``getaddrinfo`` has no timeout and cannot be cancelled, so the lookup
    runs on a daemon thread — a hung resolver never blocks process exit —
    and at most ``DNS_LOOKUP_SLOTS`` lookups run at once, so hung lookups
    cannot pile up threads. ``timeout`` bounds the whole wait, slot
    included; on expiry a ``TimeoutError`` surfaces for the caller to turn
    into a tool error.
    """
    start = time.monotonic()
    if not _dns_slots.acquire(timeout=timeout):
        raise TimeoutError("DNS resolver busy")
    lookup = _Lookup(host)
    threading.Thread(target=lookup.run, name="k0-dns", daemon=True).start()
    if not lookup.done.wait(max(0.0, timeout - (time.monotonic() - start))):
        raise TimeoutError("DNS lookup timed out")
    if lookup.error is not None:
        raise lookup.error
    return sorted(
        {info[4][0] for info in lookup.infos if isinstance(info[4][0], str)}
    )


class _Lookup:
    """One ``getaddrinfo`` call run off-thread; frees its DNS slot when done."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.done = threading.Event()
        self.infos: list[Any] = []
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.infos = socket.getaddrinfo(self.host, None, type=socket.SOCK_STREAM)
        except Exception as error:
            self.error = error
        finally:
            _dns_slots.release()
            self.done.set()


class _Deadline:
    """A monotonic deadline; ``remaining`` raises once it has passed."""

    def __init__(self, timeout: float) -> None:
        self._end = time.monotonic() + timeout

    def remaining(self) -> float:
        left = self._end - time.monotonic()
        if left <= 0:
            raise TimeoutError("fetch deadline passed")
        return left


class _Watchdog:
    """Shuts the live socket down at the deadline, unblocking any read.

    Socket timeouts bound each blocking call, not the request: a server
    dripping one byte per read would never trip them. The watchdog bounds
    the whole exchange — TLS handshake, headers and body. ``watch`` starts
    the timer on the first socket and retargets it when TLS wraps the
    socket (the wrap detaches the original), so the shutdown always hits
    the descriptor in use.
    """

    def __init__(self, deadline: _Deadline) -> None:
        self._deadline = deadline
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._timer: threading.Timer | None = None
        self.fired = False

    def watch(self, sock: socket.socket) -> None:
        with self._lock:
            self._sock = sock
            if self.fired:
                raise TimeoutError("fetch deadline passed")
        if self._timer is None:
            self._timer = threading.Timer(self._deadline.remaining(), self._fire)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _fire(self) -> None:
        with self._lock:
            self.fired = True
            if self._sock is not None:
                _shutdown(self._sock)


def _shutdown(sock: socket.socket) -> None:
    """Shut the descriptor down; for TLS, skips the ``SSLSocket`` unwrap."""
    with contextlib.suppress(OSError):
        socket.socket.shutdown(sock, socket.SHUT_RDWR)


def socket_transport(
    target: FetchTarget,
    addresses: tuple[str, ...],
    *,
    timeout: float,
    max_bytes: int,
    connect: Connect = socket.create_connection,
) -> HttpResponse:
    """GET ``target`` through the first reachable pinned address.

    ``timeout`` is one deadline for the whole call — every address tried,
    connect, TLS, headers and body — not a per-read socket timeout.
    ``connect`` is the socket seam (``socket.create_connection`` in
    production). Raises ``TimeoutError`` past the deadline, else the last
    ``OSError``/``HTTPException`` when no address answers; the caller turns
    it into a tool error.
    """
    if not addresses:
        raise ValueError("socket_transport needs at least one address")
    deadline = _Deadline(timeout)
    failure: OSError | http.client.HTTPException | None = None
    for address in addresses:
        try:
            return _get_once(
                target, address, deadline=deadline, max_bytes=max_bytes, connect=connect
            )
        except (OSError, http.client.HTTPException) as error:
            failure = error
    assert failure is not None
    raise failure


def _get_once(
    target: FetchTarget,
    address: str,
    *,
    deadline: _Deadline,
    max_bytes: int,
    connect: Connect,
) -> HttpResponse:
    watchdog = _Watchdog(deadline)
    conn = _connection(target, address, deadline.remaining(), connect, watchdog)
    try:
        response = _exchange(conn, target, max_bytes)
    except (OSError, http.client.HTTPException) as error:
        if watchdog.fired:
            raise TimeoutError("fetch deadline passed") from error
        raise
    finally:
        watchdog.cancel()
        conn.close()
    if watchdog.fired:
        raise TimeoutError("fetch deadline passed")
    return response


def _exchange(
    conn: http.client.HTTPConnection, target: FetchTarget, max_bytes: int
) -> HttpResponse:
    conn.request("GET", target.selector, headers=_request_headers())
    response = conn.getresponse()
    body = response.read(max_bytes + 1)
    return HttpResponse(
        status=response.status,
        headers={name.lower(): value for name, value in response.getheaders()},
        body=body[:max_bytes],
        truncated=len(body) > max_bytes,
    )


def _connection(
    target: FetchTarget,
    address: str,
    timeout: float,
    connect: Connect,
    watchdog: _Watchdog,
) -> http.client.HTTPConnection:
    cls = _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
    return cls(target.host, target.port, address, timeout, connect, watchdog)


def _request_headers() -> dict[str, str]:
    return {
        "Accept": "text/html, text/plain, */*",
        "Accept-Encoding": "identity",
        "User-Agent": "k0ntrol",
        "Connection": "close",
    }


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to a validated address while naming ``host`` in requests.

    The socket is stored before the watchdog sees it, so ``close`` still
    reaches it when watching fails past the deadline.
    """

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        timeout: float,
        connect: Connect,
        watchdog: _Watchdog,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address
        self._connect = connect
        self._watchdog = watchdog

    def connect(self) -> None:
        self.sock = self._connect((self._address, self.port), self.timeout)
        self._watchdog.watch(self.sock)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Like ``_PinnedHTTPConnection``; TLS still verifies ``host``.

    The handshake runs only after the watchdog follows the socket into
    its TLS wrapper, so the deadline bounds the handshake too.
    """

    def __init__(
        self,
        host: str,
        port: int,
        address: str,
        timeout: float,
        connect: Connect,
        watchdog: _Watchdog,
    ) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address
        self._connect = connect
        self._watchdog = watchdog
        self._tls_context = context

    def connect(self) -> None:
        self.sock = self._connect((self._address, self.port), self.timeout)
        self._watchdog.watch(self.sock)
        tls = self._tls_context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.sock = tls
        self._watchdog.watch(tls)
        tls.do_handshake()
