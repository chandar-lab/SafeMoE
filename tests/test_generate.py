from types import SimpleNamespace

import safemoe.generation as generation
from scripts.generate import format_prompt


def test_load_model_uses_saved_routing_mode_unless_overridden(monkeypatch):
    class Model:
        config = SimpleNamespace(routing_mode="last_token")

        def to(self, _device):
            return self

        def eval(self):
            return self

    model = Model()
    modes = []
    monkeypatch.setattr(generation, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(generation, "is_safemoe_checkpoint", lambda _path: True)
    monkeypatch.setattr(generation, "load_safemoe", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(generation, "set_routing_mode", lambda _model, mode: modes.append(mode))
    monkeypatch.setattr(generation.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())

    generation.load_model("checkpoint")
    generation.load_model("checkpoint", routing_mode="first_layer")

    assert modes == ["last_token", "first_layer"]


def test_contextual_prompt_uses_harmbench_serialization():
    record = {"context": "Reference document", "prompt": "Answer the question"}

    assert format_prompt(record) == "Reference document\n\n---\n\nAnswer the question"


def test_prompt_without_context_is_unchanged():
    assert format_prompt({"prompt": "Answer the question"}) == "Answer the question"
