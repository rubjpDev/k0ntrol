"""``socket_transport``: pinned, bounded GET — checked over socket pairs.

No live network: ``connect`` is the transport's socket seam, fed by
``socket.socketpair`` ends holding canned responses.
"""

import socket
import threading
import time

import pytest

from k0ntrol.tools.net import FetchTarget, resolve_addresses, socket_transport


def make_target(**overrides) -> FetchTarget:
    fields = {
        "scheme": "http",
        "host": "example.com",
        "port": 80,
        "path": "/x",
        "query": "q=1",
    }
    fields.update(overrides)
    return FetchTarget(**fields)


def canned_response(body: bytes = b"hi", *, length: int | None = None) -> bytes:
    size = len(body) if length is None else length
    return (
        b"HTTP/1.1 200 OK\r\nContent-Length: " + str(size).encode() + b"\r\n\r\n" + body
    )


def test_socket_transport_connects_to_the_pinned_address():
    seen = {}
    client, peer = socket.socketpair()
    peer.sendall(canned_response())
    peer.settimeout(5)

    def connect(address, timeout):
        seen["address"] = address
        return client

    response = socket_transport(
        make_target(), ("93.184.216.34",), timeout=5, max_bytes=100, connect=connect
    )

    assert seen["address"] == ("93.184.216.34", 80)
    assert response.status == 200
    assert response.body == b"hi"
    assert response.truncated is False
    request = peer.recv(65536)
    assert request.startswith(b"GET /x?q=1 HTTP/1.1")
    assert b"Host: example.com" in request
    client.close()
    peer.close()


def test_socket_transport_reads_at_most_max_bytes_plus_one():
    client, peer = socket.socketpair()
    peer.sendall(canned_response(b"a" * 200))
    peer.settimeout(5)

    response = socket_transport(
        make_target(),
        ("93.184.216.34",),
        timeout=5,
        max_bytes=50,
        connect=lambda address, timeout: client,
    )

    assert response.body == b"a" * 50
    assert response.truncated is True
    client.close()
    peer.close()


def test_socket_transport_tries_the_next_address_after_a_failure():
    first, first_peer = socket.socketpair()
    second, second_peer = socket.socketpair()
    second_peer.sendall(canned_response(b"second"))
    second_peer.settimeout(5)
    tried = []

    def connect(address, timeout):
        tried.append(address)
        if len(tried) == 1:
            first.close()
            first_peer.close()
            raise OSError("connection refused")
        return second

    response = socket_transport(
        make_target(),
        ("192.0.2.1", "93.184.216.34"),
        timeout=5,
        max_bytes=100,
        connect=connect,
    )

    assert tried == [("192.0.2.1", 80), ("93.184.216.34", 80)]
    assert response.body == b"second"
    second.close()
    second_peer.close()


def test_socket_transport_raises_the_last_failure():
    def connect(address, timeout):
        raise OSError("connection refused")

    with pytest.raises(OSError, match="connection refused"):
        socket_transport(
            make_target(),
            ("192.0.2.1", "198.51.100.1"),
            timeout=5,
            max_bytes=100,
            connect=connect,
        )


def test_socket_transport_needs_an_address():
    with pytest.raises(ValueError):
        socket_transport(
            make_target(), (), timeout=5, max_bytes=100, connect=lambda a, t: None
        )


def test_resolve_addresses_reads_a_literal_without_dns():
    assert resolve_addresses("93.184.216.34", timeout=5) == ["93.184.216.34"]


def test_resolve_addresses_times_out_a_hanging_lookup(monkeypatch):
    release = threading.Event()

    def sleeper(*args, **kwargs):
        release.wait(30)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", sleeper)
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            resolve_addresses("example.com", timeout=0.05)
    finally:
        release.set()
    assert time.monotonic() - start < 5
