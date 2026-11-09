"""8-bit badges for the services a project uses, drawn in the welcome banner.

Each icon is 8 pixels tall; half blocks put two pixel rows in one text row,
so an icon is 4 text rows, with the service name and its role (issues,
code, models) under it. The art is k0's own pixel take on each service in
its brand colour, not a copy of anyone's logo.
"""

from dataclasses import dataclass

from rich.text import Text

from k0ntrol.frontends.theme import DIM

_GAP = "  "  # between two badges
_ROLE_COLOR = "#6B7280"


@dataclass(frozen=True)
class Badge:
    name: str
    role: str
    pixels: tuple[str, ...]  # one string per pixel row; "." is empty
    colors: dict[str, str]  # pixel key -> colour


def _badge(name: str, role: str, colors: dict[str, str], *pixels: str) -> Badge:
    return Badge(name, role, pixels, colors)


def _issues(name: str, color: str) -> Badge:
    """An open-issue ring, in the tracker's colour."""
    return _badge(name, "issues", {"i": color},
                  "..iiiii..",
                  ".i.....i.",
                  "i.......i",
                  "i...i...i",
                  "i..iii..i",
                  "i...i...i",
                  ".i.....i.",
                  "..iiiii..")


TRACKER_BADGES: dict[str, Badge] = {
    "github": _issues("GitHub", "#3FB950"),
    "gitlab": _issues("GitLab", "#FC6D26"),
    "linear": _badge("Linear", "issues", {"p": "#5E6AD2"},
                     "...ppp...",
                     ".p.ppppp.",
                     "p.p.ppppp",
                     "pp.p.pppp",
                     "ppp.p.ppp",
                     "pppp.p.pp",
                     ".pppp.p..",
                     "...ppp..."),
    "jira": _badge("Jira", "issues", {"j": "#2684FF", "d": "#0052CC"},
                   "....j....",
                   "...jjj...",
                   "..jj.jj..",
                   ".jj.d.jj.",
                   "jj.ddd.jj",
                   ".jj.d.jj.",
                   "..jj.jj..",
                   "...jjj..."),
}

VCS_BADGES: dict[str, Badge] = {
    "github": _badge("GitHub", "code", {"w": "#E5E7EB"},
                     ".w.....w.",
                     ".ww...ww.",
                     ".wwwwwww.",
                     "ww.www.ww",
                     "wwwwwwwww",
                     ".wwwwwww.",
                     "..w.w.w..",
                     ".w..w..w."),
    "gitlab": _badge("GitLab", "code", {"r": "#E24329", "o": "#FC6D26", "y": "#FCA326"},
                     "r.......r",
                     "rr.....rr",
                     "rrr...rrr",
                     "yrrooorry",
                     "yyoooooyy",
                     ".yyoooyy.",
                     "...yoy...",
                     "....o...."),
    "bitbucket": _badge("Bitbucket", "code", {"b": "#2684FF", "d": "#0052CC"},
                        "bbbbbbbbb",
                        "bbbbbbbbb",
                        ".bbdddbb.",
                        ".bbd.dbb.",
                        "..bdddb..",
                        "..bbbbb..",
                        "...bbb...",
                        "........."),
}

LLM_BADGES: dict[str, Badge] = {
    "anthropic": _badge("Anthropic", "models", {"c": "#D97757"},
                        "....c....",
                        ".c..c..c.",
                        "..c.c.c..",
                        "...ccc...",
                        "ccccccccc",
                        "...ccc...",
                        "..c.c.c..",
                        ".c..c..c."),
    "bedrock": _badge("Bedrock", "models", {"o": "#FF9900", "s": "#4B5563"},
                      "sssssssss",
                      ".........",
                      "sssssssss",
                      ".........",
                      "o.......o",
                      ".oo...oo.",
                      "...ooo..o",
                      ".......oo"),
    "ollama": _badge("Ollama", "models", {"w": "#E5E7EB"},
                     "..w...w..",
                     "..w...w..",
                     "..wwwww..",
                     ".ww.w.ww.",
                     ".wwwwwww.",
                     "..wwwww..",
                     "...www...",
                     "...www..."),
    "lmstudio": _badge("LM Studio", "models", {"v": "#8B5CF6", "l": "#C4B5FD"},
                       ".vvvvv...",
                       ".........",
                       "...lllll.",
                       ".........",
                       ".vvvvvv..",
                       ".........",
                       "..lllll..",
                       "........."),
    "openai": _badge("OpenAI", "models", {"w": "#E5E7EB"},
                     "...www...",
                     ".ww...ww.",
                     "w..www..w",
                     "w.w...w.w",
                     "w.w...w.w",
                     "w..www..w",
                     ".ww...ww.",
                     "...www..."),
    "openrouter": _badge("OpenRouter", "models", {"s": "#94A3B8"},
                         ".......s.",
                         ".....ssss",
                         "....s..s.",
                         "ssss.....",
                         "....s..s.",
                         ".....ssss",
                         ".......s.",
                         "........."),
    "vertex": _badge("Vertex", "models", {"b": "#4285F4", "r": "#EA4335", "y": "#FBBC04", "g": "#34A853"},
                     "....b....",
                     "...bbr...",
                     "..bbbrr..",
                     ".bbbbrrr.",
                     ".ggggyyy.",
                     "..gggyy..",
                     "...ggy...",
                     "....g...."),
}


def _chip(name: str) -> Badge:
    """Any other model provider: a plain chip with its name."""
    return _badge(name, "models", {"g": "#9CA3AF", "d": "#4B5563"},
                  ".g.g.g.g.",
                  "ggggggggg",
                  ".gdddddg.",
                  "ggdddddgg",
                  ".gdddddg.",
                  "ggdddddgg",
                  "ggggggggg",
                  ".g.g.g.g.")


def service_badges(*, tracker: str | None, vcs: str | None, provider: str | None) -> list[Badge]:
    """The project's badges, ticket then code then models; unknown or
    missing trackers and hosts are left out, unknown models get a chip."""
    badges = [TRACKER_BADGES.get(tracker or ""), VCS_BADGES.get(vcs or "")]
    if provider:
        badges.append(LLM_BADGES.get(provider) or _chip(provider))
    return [badge for badge in badges if badge is not None]


def half_block(top: str, bottom: str, colors: dict[str, str]) -> tuple[str, str]:
    """One text cell holding two pixels: the character and its style."""
    if top == bottom:
        return (" ", "") if top == "." else ("█", colors[top])
    if bottom == ".":
        return "▀", colors[top]
    if top == ".":
        return "▄", colors[bottom]
    return "▀", f"{colors[top]} on {colors[bottom]}"


def _width(badge: Badge) -> int:
    return max(len(badge.pixels[0]), len(badge.name), len(badge.role))


def block_width(badges: list[Badge]) -> int:
    """Columns the badge block takes, gaps included."""
    return sum(_width(badge) for badge in badges) + len(_GAP) * max(len(badges) - 1, 0)


def _icon_rows(badge: Badge) -> list[Text]:
    rows = []
    for top, bottom in zip(badge.pixels[::2], badge.pixels[1::2], strict=True):
        row = Text()
        for pair in zip(top, bottom, strict=True):
            row.append(*half_block(*pair, badge.colors))
        rows.append(row)
    return rows


def _column(badge: Badge) -> list[Text]:
    """A badge's rows, each centred to the badge width."""
    width = _width(badge)
    rows = [*_icon_rows(badge), Text(badge.name, style=DIM), Text(badge.role, style=_ROLE_COLOR)]
    for row in rows:
        row.align("center", width)
    return rows


def badge_block(badges: list[Badge]) -> list[Text]:
    """The badges side by side: four icon rows, then names, then roles."""
    columns = [_column(badge) for badge in badges]
    return [Text(_GAP).join(cells) for cells in zip(*columns, strict=True)]
