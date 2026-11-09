"""The terminal look: one palette and glyph set every frontend shares.

Modelled on Claude Code's terminal UI: a bullet per action, an elbow for
its result, a spinner while an agent works, a pointer for the selected
option. Colours are rich style strings.
"""

ACCENT = "#A78BFA"  # violet: bullets, pointer, active option
BRAND = "#5EEAD4"  # teal: the k0 logo's start colour, prompts
DIM = "#9CA3AF"  # grey: secondary text, hints, timings
BORDER = "#374151"  # dark grey: panels, rules
OK = "#4ADE80"  # green: success, approvals, added lines
ERROR = "#F87171"  # red: failures, denials, removed lines
WARN = "#FBBF24"  # amber: limits, warnings, pauses

BULLET = "⏺"  # one action or message
ELBOW = "⎿"  # the result under an action
POINTER = "❯"  # the selected option, the input prompt
CHECK = "✓"
CROSS = "✗"
SPINNER_FRAMES = ("·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢")

# Workflow agents show in capitals, helpers in lower case (GLOSSARY).
AGENTS = ("spec", "tester", "coder", "reviewer", "chatty")
