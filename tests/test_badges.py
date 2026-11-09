from k0ntrol.frontends.badges import (
    LLM_BADGES,
    TRACKER_BADGES,
    VCS_BADGES,
    badge_block,
    block_width,
    half_block,
    service_badges,
)
from k0ntrol.frontends.banner import banner_layout, render_banner


def test_half_blocks_carry_two_pixel_rows_per_cell():
    colors = {"a": "#111111", "b": "#222222"}
    assert half_block(".", ".", colors) == (" ", "")
    assert half_block("a", "a", colors) == ("█", "#111111")
    assert half_block("a", ".", colors) == ("▀", "#111111")
    assert half_block(".", "b", colors) == ("▄", "#222222")
    assert half_block("a", "b", colors) == ("▀", "#111111 on #222222")


def test_every_icon_is_eight_pixels_tall_with_even_rows_and_known_colours():
    icons = [*TRACKER_BADGES.values(), *VCS_BADGES.values(), *LLM_BADGES.values()]
    for badge in icons:
        assert len(badge.pixels) == 8, badge.name
        assert len({len(row) for row in badge.pixels}) == 1, badge.name
        used = {key for row in badge.pixels for key in row} - {"."}
        assert used <= set(badge.colors), badge.name


def test_badges_go_ticket_then_code_then_models():
    badges = service_badges(tracker="linear", vcs="github", provider="anthropic")
    assert [(b.name, b.role) for b in badges] == [
        ("Linear", "issues"), ("GitHub", "code"), ("Anthropic", "models"),
    ]


def test_missing_services_are_left_out_and_unknown_models_get_a_generic_chip():
    badges = service_badges(tracker=None, vcs="github", provider="mystery")
    assert [(b.name, b.role) for b in badges] == [("GitHub", "code"), ("mystery", "models")]
    assert service_badges(tracker=None, vcs=None, provider=None) == []


def test_the_block_is_icon_rows_then_name_then_role_under_each_badge():
    rows = badge_block(service_badges(tracker="github", vcs="github", provider="ollama"))
    plain = [row.plain for row in rows]
    assert len(plain) == 6
    assert ["GitHub", "GitHub", "Ollama"] == plain[4].split()
    assert ["issues", "code", "models"] == plain[5].split()
    assert len({len(row) for row in plain}) == 1
    assert block_width(service_badges(tracker="github", vcs="github", provider="ollama")) == len(plain[0])


def test_the_banner_shows_the_badges_flush_right_at_the_terminal_width():
    badges = service_badges(tracker="github", vcs="github", provider="anthropic")
    text = render_banner(version="0.1.0", provider="anthropic", cwd="/w", color=False,
                         badges=badges, width=110)
    lines = text.splitlines()
    assert {len(line) for line in lines} == {110}
    names = next(line for line in lines if "Anthropic" in line)
    assert names.rstrip(" │").endswith("Anthropic")
    assert "issues" in text and "code" in text and "models" in text
    assert "k0ntrol v0.1.0" in text


def test_a_narrow_terminal_drops_the_badges_and_keeps_the_path():
    badges = service_badges(tracker="github", vcs="github", provider="anthropic")
    kept, room = banner_layout(70, badges)
    assert kept == [] and room > 20
    shown, room_with = banner_layout(140, badges)
    assert shown == badges and room_with < 140 - 40


def test_names_and_roles_stay_centred_under_their_icons_in_the_banner():
    badges = service_badges(tracker="linear", vcs="github", provider="anthropic")
    text = render_banner(version="0.1.0", provider="anthropic", cwd="/w", color=False,
                         badges=badges, width=120)
    lines = text.splitlines()
    names = next(line for line in lines if "Linear" in line)
    roles = next(line for line in lines if "issues" in line)
    assert roles.index("issues") == names.index("Linear")
    assert roles.index("models") == names.index("Anthropic") + 1
