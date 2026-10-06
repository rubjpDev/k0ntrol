from k0ntrol.frontends.banner import color_enabled, render_banner
from k0ntrol.frontends.logo import LOGO


def test_logo_is_the_decided_art():
    assert LOGO.splitlines() == [
        "   ███▘         ▄█████▙▖ ",
        "  ▗███        ▄███▀▀▜███▌",
        "  ▟██▌  ▄███▘▐██▛  ▟█▀███",
        " ▗███ ▄██▛▘ ▗███▘ ▟█▘▐███",
        " ▐██████▖   ▐██▛ ▟█▘ ▟██▌",
        " ███▛▀███   ▐██▙▟▛▘ ▗███ ",
        "▐██▛  ▜██▙   ████▄▄▟██▛  ",
        "███▘   ▜██▌   ▀█████▀▘   ",
    ]


def test_plain_banner_has_logo_fields_and_no_ansi():
    text = render_banner(
        version="0.0.1.dev0",
        backend="cursor",
        cwd="/work/repo",
        color=False,
    )
    assert "╭" in text
    assert "▄█████▙" in text
    assert "/work/repo" in text
    assert "\x1b" not in text
    version_rows = [line for line in text.splitlines() if "k0ntrol v0.0.1.dev0" in line]
    assert len(version_rows) == 1
    assert "▟" in version_rows[0]
    assert "▄█████▙" not in version_rows[0]
    body = "\n".join(line for line in text.splitlines() if line.startswith("│"))
    assert "k0ntrol v0.0.1.dev0" in body
    assert "cursor" in body
    start = body.index("k0ntrol v0.0.1.dev0")
    middle = body.index("cursor", start)
    end = body.index("/work/repo", middle)
    assert "\n\n" not in body[start:end]


def test_color_banner_uses_the_decided_palette():
    text = render_banner(
        version="0.0.1.dev0",
        backend="cursor",
        cwd="/work",
        color=True,
    )
    assert "\x1b[38;2;55;65;81m" in text
    assert "\x1b[38;2;156;163;175m" in text
    assert "\x1b[38;2;94;234;212m" in text
    assert "\x1b[38;2;167;139;250m" in text


def test_color_enabled_follows_no_color_and_tty(monkeypatch):
    class TTY:
        def isatty(self):
            return True

    class NotTTY:
        def isatty(self):
            return False

    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(TTY()) is True
    monkeypatch.setenv("NO_COLOR", "")
    assert color_enabled(TTY()) is False
    monkeypatch.delenv("NO_COLOR")
    assert color_enabled(NotTTY()) is False
