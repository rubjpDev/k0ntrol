from k0ntrol.memory.layout import MEM, RULES
from k0ntrol.memory.reflect import apply_mem_decision, offer_mem_proposals, run_reflect
from k0ntrol.memory.session import (
    add_mem_proposal,
    list_facts,
    list_pending_mem_proposals,
)


def test_run_reflect_stores_facts_from_the_helper_reply(tmp_path):
    transcript = tmp_path / ".k0-mem" / "sessions" / "run.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("/ask how does checkout work\n", encoding="utf-8")
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return "fact: checkout uses sqlite\nepisode: asked how checkout works\n"

    run_reflect(root=tmp_path, invoke=fake_invoke, transcript=transcript)
    assert "how does checkout work" in seen["prompt"]
    assert [item["body"] for item in list_facts(tmp_path)] == ["checkout uses sqlite"]


def test_run_reflect_queues_mem_proposals_without_writing_files(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    topology = tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name / "strategy.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    topology.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    rules.write_text(RULES, encoding="utf-8")
    topology.write_text("human owned\n", encoding="utf-8")

    def fake_invoke(prompt):
        return (
            "fact: checkout uses sqlite\n"
            "mem: Prefer sqlite for checkout state\n"
        )

    run_reflect(root=tmp_path, invoke=fake_invoke)
    assert list_pending_mem_proposals(tmp_path) == [
        {"id": 1, "body": "Prefer sqlite for checkout state"}
    ]
    assert mem.read_text(encoding="utf-8") == MEM
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert not (tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name / "code_rules.md").exists()


def test_run_reflect_does_not_requeue_decided_mem_proposals(tmp_path):
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    apply_mem_decision(tmp_path, 1, False)
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return "mem: Prefer sqlite for checkout state\n"

    run_reflect(root=tmp_path, invoke=fake_invoke)
    assert list_pending_mem_proposals(tmp_path) == []
    assert "rejected: Prefer sqlite for checkout state" in seen["prompt"]


def test_offer_mem_proposals_answers_between_commands(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    shown = []

    offer_mem_proposals(tmp_path, iter(["n", "y"]).__next__, shown.append)
    assert shown == ["Mem.md += Prefer sqlite for checkout state? y/n"]
    assert mem.read_text(encoding="utf-8") == MEM
    assert list_pending_mem_proposals(tmp_path) == []


def test_offer_mem_proposals_reprompts_on_other_answers(tmp_path):
    add_mem_proposal(tmp_path, "Prefer sqlite")
    shown = []
    answers = iter(["maybe", "n"])
    offer_mem_proposals(tmp_path, lambda: next(answers), shown.append)
    assert shown == [
        "Mem.md += Prefer sqlite? y/n",
        "y or n",
    ]


def test_apply_mem_decision_appends_approved_body(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    apply_mem_decision(tmp_path, 1, True)
    assert mem.read_text(encoding="utf-8").endswith(
        "- Prefer sqlite for checkout state\n"
    )


def test_apply_mem_decision_appends_after_unterminated_text(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("# Mem\nno newline", encoding="utf-8")
    add_mem_proposal(tmp_path, "fact two")
    apply_mem_decision(tmp_path, 1, True)
    assert mem.read_text(encoding="utf-8") == "# Mem\nno newline\n- fact two\n"
