import torch

from safemoe.training import CausalCollator, build_dataset


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation, max_length):
        return {"input_ids": list(range(1, min(len(text), max_length) + 1))}


def test_prompt_lengths_are_preserved_by_tokenization_and_collation():
    tokenizer = CharacterTokenizer()
    dataset = build_dataset(
        [{"prompt": "a", "response": "x"}, {"prompt": "long", "response": "y"}],
        tokenizer,
        "Q{prompt} A:{response}",
        max_length=32,
    )

    batch = CausalCollator(tokenizer)([dataset[0], dataset[1]])

    assert batch["prompt_lengths"].tolist() == [5, 8]
    assert batch["input_ids"].shape == (2, 9)
    assert torch.equal(batch["attention_mask"].sum(dim=1), torch.tensor([6, 9]))
