from k0ntrol.harness.delegate import should_delegate


def test_threshold_is_exclusive_at_350(tmp_path):
    path = tmp_path / "big.py"
    path.write_text("x\n" * 350)
    assert should_delegate(path, 350) is False
    path.write_text("x\n" * 351)
    assert should_delegate(path, 350) is True
