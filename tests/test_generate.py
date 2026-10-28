from scripts.generate import format_prompt


def test_contextual_prompt_uses_harmbench_serialization():
    record = {"context": "Reference document", "prompt": "Answer the question"}

    assert format_prompt(record) == "Reference document\n\n---\n\nAnswer the question"


def test_prompt_without_context_is_unchanged():
    assert format_prompt({"prompt": "Answer the question"}) == "Answer the question"
