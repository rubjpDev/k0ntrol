import re

from rich.console import Console

from k0ntrol.frontends.diffview import parse_diff, render_diff, side_by_side

NEW_FILE = """\
diff --git a/calc.py b/calc.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/calc.py
@@ -0,0 +1,2 @@
+def answer():
+    return 42
"""

APPEND = """\
diff --git a/notes.py b/notes.py
index 1111111..2222222 100644
--- a/notes.py
+++ b/notes.py
@@ -1,2 +1,3 @@
 keep_one = 1
 keep_two = 2
+added_three = 3
"""

CHANGE = """\
diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,3 @@
 def answer():
-    return 41
+    return 42
 # end
@@ -20,2 +20,2 @@
-old_tail = 1
+new_tail = 1
\\ No newline at end of file
"""

DELETED = """\
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-bye = 1
"""


def rendered(diff: str, width: int = 120) -> list[str]:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(render_diff(diff))
    return [line.rstrip() for line in console.export_text().splitlines()]


def row_with(rows: list[str], text: str) -> str:
    return next(row for row in rows if text in row)


def test_parse_reads_path_kinds_and_line_numbers() -> None:
    (file,) = parse_diff(CHANGE)
    assert file.path == "calc.py"
    changes = [(line.kind, line.old, line.new) for line in file.lines if line.kind != " "]
    assert changes[:2] == [("-", 2, None), ("+", None, 2)]
    assert ("-", 20, None) in changes and ("+", None, 20) in changes
    assert not any("No newline" in line.text for line in file.lines)


def test_paths_of_new_and_deleted_files() -> None:
    assert [f.path for f in parse_diff(NEW_FILE + DELETED)] == ["calc.py", "gone.py"]


def test_side_by_side_pairs_removed_with_added_lines() -> None:
    (file,) = parse_diff(CHANGE)
    pairs = [
        (left.text if left else None, right.text if right else None)
        for left, right in side_by_side(file.lines)
    ]
    assert ("    return 41", "    return 42") in pairs
    assert ("# end", "# end") in pairs  # context sits on both sides


def test_a_new_file_shows_one_column_of_additions() -> None:
    rows = rendered(NEW_FILE)
    assert "calc.py +2 -0" in rows[0]
    assert re.search(r"1 \+ def answer\(\):", row_with(rows, "def answer"))
    assert "│" not in row_with(rows, "def answer").strip("│ ")


def test_pure_additions_to_an_existing_file_stay_one_column() -> None:
    rows = rendered(APPEND)
    assert re.search(r"3 \+ added_three = 3", row_with(rows, "added_three"))
    assert row_with(rows, "keep_one").count("keep_one") == 1


def test_changed_code_is_shown_side_by_side() -> None:
    rows = rendered(CHANGE)
    assert "calc.py +2 -2" in rows[0]
    row = row_with(rows, "return 41")
    assert re.search(r"2 - +return 41 .*│ +2 \+ +return 42", row)
    assert row_with(rows, "# end").count("# end") == 2


def test_hunks_are_separated_in_the_split_view() -> None:
    rows = rendered(CHANGE)
    gap = rows.index(row_with(rows, "⋯"))
    assert gap < rows.index(row_with(rows, "old_tail"))


def test_a_deleted_file_shows_one_column_of_removals() -> None:
    rows = rendered(DELETED)
    assert re.search(r"1 - bye = 1", row_with(rows, "bye"))
    assert row_with(rows, "bye").count("bye") == 1


def test_each_file_gets_its_own_view() -> None:
    rows = rendered(NEW_FILE + CHANGE.replace("calc.py", "lib.py"))
    assert row_with(rows, "def answer").count("def answer") == 1  # one column
    assert row_with(rows, "# end").count("# end") == 2  # split


def test_a_long_diff_is_cut_with_a_count() -> None:
    body = "".join(f"+line {n}\n" for n in range(1, 501))
    diff = f"diff --git a/big.py b/big.py\n--- /dev/null\n+++ b/big.py\n@@ -0,0 +1,500 @@\n{body}"
    rows = rendered(diff)
    assert any("more diff lines" in row for row in rows)
    assert not any("line 500" in row for row in rows)


def test_a_file_without_hunks_says_so() -> None:
    rows = rendered("diff --git a/logo.png b/logo.png\nBinary files a/logo.png and b/logo.png differ\n")
    assert "logo.png" in rows[0]
    assert any("no text changes" in row for row in rows)
