"""Web-search providers and their request/response codec (ADR 0013).

``search.provider`` in config names one entry of ``SEARCH_PROVIDERS``; the
``web_search`` tool calls it over the standard library with a key saved
like the provider keys. Each provider carries its own codec — where the
endpoint lives, which env var holds the key, how a query becomes a request
and how the JSON body becomes results — so the tool stays
provider-agnostic. Everything a provider returns is untrusted text.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


class SearchError(Exception):
    """A search response the tool cannot use; the tool maps it to ToolError."""


@dataclass(frozen=True)
class SearchRequest:
    """One search API call for the injected ``fetch`` seam."""

    method: str
    url: str
    headers: dict[str, str]


@dataclass(frozen=True)
class SearchResult:
    """One web result: title, URL and snippet — untrusted text."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchProvider:
    """One configured search API and the codec that speaks to it."""

    name: str
    endpoint: str
    key_env: str
    build_request: Callable[[str, str, str], SearchRequest]
    parse_results: Callable[[bytes], list[SearchResult]]


def search_provider_name(cfg: dict) -> str | None:
    """Read the provider named by ``search.provider``, if configured."""
    search = cfg.get("search")
    name = search.get("provider") if isinstance(search, dict) else None
    return name if isinstance(name, str) else None


# The agents a provider's own web search can replace k0's ``web_search``
# for (ADR 0017): the read-only network agents, never a writing one.
NATIVE_SEARCH_AGENTS = ("planner", "researcher")

# The k0 providers whose own web search is wired (ADR 0017): OpenAI
# (#182) and Anthropic (#209). Every other provider keeps search.provider.
NATIVE_SEARCH_PROVIDERS = ("openai", "anthropic")


def uses_native_search(cfg: dict, agent: str) -> bool:
    """True when ``agent`` searches with its provider's own web search.

    Only with ``search.native: true``, only PLANNER and RESEARCHER, and
    only on a k0 provider in ``NATIVE_SEARCH_PROVIDERS`` — never on
    LangChain's ``model_provider`` name, which ``lmstudio`` and
    ``openai_compatible`` share with ``openai`` (ADR 0017).
    """
    search = cfg.get("search")
    if not isinstance(search, dict) or search.get("native") is not True:
        return False
    if agent not in NATIVE_SEARCH_AGENTS:
        return False
    return native_search_provider(cfg, agent) in NATIVE_SEARCH_PROVIDERS


def native_search_provider(cfg: dict, agent: str) -> str:
    """The k0 provider ``agent`` runs on, as its ``agents:`` entry names
    it, or ``""`` when the entry names none."""
    entry = (cfg.get("agents") or {}).get(agent) or {}
    provider = entry.get("provider")
    return provider if isinstance(provider, str) else ""


def _brave_request(endpoint: str, query: str, api_key: str) -> SearchRequest:
    url = f"{endpoint}?q={quote(query, safe='')}"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    return SearchRequest(method="GET", url=url, headers=headers)


def _brave_results(body: bytes) -> list[SearchResult]:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SearchError("search response was not valid JSON") from error
    return [_brave_result(entry) for entry in _web_results(data)]


def _web_results(data: Any) -> list[dict]:
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        raise SearchError("search response had no results list")
    return [entry for entry in results if isinstance(entry, dict)]


def _brave_result(entry: dict) -> SearchResult:
    return SearchResult(
        title=_text(entry.get("title")),
        url=_text(entry.get("url")),
        snippet=_text(entry.get("description")),
    )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


SEARCH_PROVIDERS: dict[str, SearchProvider] = {
    "brave": SearchProvider(
        name="brave",
        endpoint="https://api.search.brave.com/res/v1/web/search",
        key_env="BRAVE_SEARCH_API_KEY",
        build_request=_brave_request,
        parse_results=_brave_results,
    ),
}
