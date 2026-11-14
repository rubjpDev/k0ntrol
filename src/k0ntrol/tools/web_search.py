"""``web_search``: query the configured search API as untrusted text (ADR 0013).

Only PLANNER and RESEARCHER get this tool. ``search.provider`` in config
names the API (``k0ntrol.search.SEARCH_PROVIDERS``); its key is an env var
saved like the provider keys. The agent sees only ``query``; the tool adds
the provider endpoint as a ``url`` argument to the recorded inputs, so the
tool event keeps the search host — never the query. Everything it returns
is marked untrusted, the way ``fetch_url`` marks pages.
"""

import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool

from k0ntrol.search import (
    SEARCH_PROVIDERS,
    SearchError,
    SearchProvider,
    SearchRequest,
    SearchResult,
)
from k0ntrol.tools.errors import ToolError

TIMEOUT = 15.0
_MAX_BODY = 256 * 1024

Fetch = Callable[[SearchRequest], bytes]


class _EndpointTool(StructuredTool):
    """A tool whose recorded inputs carry the endpoint ``url``.

    LangChain records the raw call arguments — schema defaults never reach
    the event handler — so the endpoint is injected here, letting the tool
    event keep the search host while the agent sees only ``query``.
    """

    endpoint: str = ""

    def run(
        self, tool_input: str | dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return super().run(_with_url(tool_input, self.endpoint), *args, **kwargs)

    async def arun(
        self, tool_input: str | dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await super().arun(
            _with_url(tool_input, self.endpoint), *args, **kwargs
        )


def make_web_search(
    cfg: dict,
    *,
    environ: Mapping[str, str] | None = None,
    fetch: Fetch | None = None,
) -> BaseTool:
    """Build ``web_search`` for the ``search.provider`` named in ``cfg``."""
    name = _provider_name(cfg)
    provider = SEARCH_PROVIDERS.get(name) if name is not None else None
    env = os.environ if environ is None else environ
    http = fetch or http_fetch

    def web_search(query: str) -> str:
        """Search the web; returns titles, URLs and snippets marked untrusted."""
        return _run_search(query, name, provider, env, http)

    return _EndpointTool.from_function(
        web_search, endpoint=provider.endpoint if provider is not None else ""
    )


def http_fetch(request: SearchRequest) -> bytes:
    """The default ``fetch``: one stdlib HTTP call, proxies and redirects off.

    Redirects are refused outright: urllib would re-send the auth header to
    the redirect target, so a 3xx must surface as an error, not a hop.
    """
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    req = urllib.request.Request(
        request.url, headers=request.headers, method=request.method
    )
    try:
        response = opener.open(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as error:
        try:
            if 300 <= error.code < 400:
                raise ToolError("search endpoint redirected") from error
            raise
        finally:
            error.close()
    try:
        body = response.read(_MAX_BODY + 1)
    finally:
        response.close()
    if len(body) > _MAX_BODY:
        raise ToolError("search response too large")
    return body


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Makes ``opener.open`` raise ``HTTPError`` on any 3xx."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _provider_name(cfg: dict) -> str | None:
    search = cfg.get("search")
    name = search.get("provider") if isinstance(search, dict) else None
    return name if isinstance(name, str) else None


def _run_search(
    query: str,
    name: str | None,
    provider: SearchProvider | None,
    environ: Mapping[str, str],
    fetch: Fetch,
) -> str:
    if provider is None:
        raise ToolError(_provider_error(name))
    api_key = environ.get(provider.key_env)
    if not api_key:
        raise ToolError(
            f"web_search needs {provider.key_env}; save it like a provider "
            "key or export it before running k0"
        )
    request = provider.build_request(provider.endpoint, query, api_key)
    body = _fetch(fetch, request, provider)
    return _render(provider, _results(provider, body))


def _provider_error(name: str | None) -> str:
    known = ", ".join(SEARCH_PROVIDERS)
    if name is None:
        return f"config has no search.provider; set it to one of: {known}"
    return f"unknown search provider {name!r}; expected one of: {known}"


def _fetch(fetch: Fetch, request: SearchRequest, provider: SearchProvider) -> bytes:
    try:
        return fetch(request)
    except ToolError:
        raise
    except Exception as error:
        raise ToolError(_fetch_error(provider, error)) from error


def _fetch_error(provider: SearchProvider, error: Exception) -> str:
    host = urlsplit(provider.endpoint).hostname or provider.endpoint
    code = getattr(error, "code", None)
    detail = (
        f"{type(error).__name__} {code}"
        if isinstance(code, int)
        else type(error).__name__
    )
    return f"search failed for {host}: {detail}"


def _results(provider: SearchProvider, body: bytes) -> list[SearchResult]:
    try:
        return provider.parse_results(body)
    except SearchError as error:
        raise ToolError(str(error)) from error


def _render(provider: SearchProvider, results: list[SearchResult]) -> str:
    host = urlsplit(provider.endpoint).hostname or provider.endpoint
    lines = [f"UNTRUSTED WEB SEARCH RESULTS {host}", ""]
    if not results:
        lines.append("no results")
    for index, result in enumerate(results, 1):
        lines += [
            f"{index}. {_flat(result.title)}",
            f"   {_flat(result.url)}",
            f"   {_flat(result.snippet)}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _with_url(
    tool_input: str | dict[str, Any], url: str
) -> str | dict[str, Any]:
    """The call input plus the endpoint ``url`` the event records."""
    if isinstance(tool_input, dict):
        return {**tool_input, "url": url}
    if isinstance(tool_input, str):
        return {"query": tool_input, "url": url}
    return tool_input
