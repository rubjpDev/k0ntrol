"""The _short helper preserves the first line of provider errors."""

import pytest

# AWS ValidationException message that's longer than 200 characters
# This simulates the real AWS error from the bug report
AWS_VALIDATION_ERROR = (
    "An error occurred (ValidationException) when calling the Converse operation: "
    "Invocation of model ID openai.gpt-6.1-terra with on-demand throughput isn't "
    "supported. Retry your request with the ID or ARN of an inference profile that "
    "contains this model"
)

assert len(AWS_VALIDATION_ERROR) > 200


def _short_error(error: Exception) -> str:
    """Helper that calls _short with the given error."""
    from k0ntrol.providers import _short
    return _short(error)


class TestShort:
    """Tests for the _short function from k0ntrol.providers."""

    def test_long_first_line_preserved_in_full(self):
        """A provider error with first line >200 chars shows the whole line."""
        error = RuntimeError(AWS_VALIDATION_ERROR)
        result = _short_error(error)

        # The entire first line should be preserved
        assert result == AWS_VALIDATION_ERROR

    def test_multiline_error_shows_only_first_line(self):
        """A multi-line error shows only its first line."""
        multi_line = "first line: something went wrong\nsecond line: stack trace\nthird line: more info"
        error = RuntimeError(multi_line)
        result = _short_error(error)

        # Only the first line should be returned
        assert result == "first line: something went wrong"
        assert "\n" not in result

    def test_short_error_unchanged(self):
        """A short error message remains unchanged."""
        error = RuntimeError("simple error")
        result = _short_error(error)

        assert result == "simple error"
