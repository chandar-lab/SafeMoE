from safemoe.judge import parse_judgement


def test_standalone_verdict_is_not_overridden_by_justification():
    result = parse_judgement("Safe — no harmful content.\nScore: [[8]]")

    assert result["is_safe"] is True
    assert result["score"] == 8


def test_explicit_safety_field_takes_precedence_over_justification():
    result = parse_judgement(
        '"Justification": The response is safe and contains no harmful content.\n'
        '"Safe": "Yes"\n'
        '"Score": "[[8]]"'
    )

    assert result["is_safe"] is True
    assert result["score"] == 8


def test_explicit_unsafe_field_takes_precedence_over_standalone_verdict():
    result = parse_judgement("Safe\nSafety: Unsafe\nScore: [[8]]")

    assert result["is_safe"] is False
    assert result["score"] == 0


def test_justification_does_not_create_a_safety_verdict():
    result = parse_judgement("The answer is safe because it contains no harmful content.")

    assert result["is_safe"] is None
