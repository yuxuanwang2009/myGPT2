from itertools import islice

import config
import regex_tokenizer as rt
from datasets import load_dataset


def _iter_docs(max_docs: int, dataset_name: str):
    ds = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
    for row in islice(ds, max_docs):
        text = row.get("text", "") if isinstance(row, dict) else ""
        if text:
            yield text


def _token_per_word(text: str, tokenizer: rt.RegexTokenizer) -> float:
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    token_count = len(tokenizer.encode(text))
    return token_count / word_count


if __name__ == "__main__":
    dataset_name = "HuggingFaceFW/fineweb-edu"
    max_docs = 1000
    print(f"Streaming {max_docs} docs from {dataset_name} to train tokenizer...", flush=True)

    text_corpus = "\n".join(_iter_docs(max_docs, dataset_name))
    print(f"Collected {len(text_corpus)} characters of text.", flush=True)

    tok = rt.RegexTokenizer.train(
        text_corpus,
        config.vocab_size,
        path="tokenizer.json",
        verbose=True,
        special_tokens=["<|endoftext|>"],
    )

    train_tpw = _token_per_word(text_corpus, tok)
    print(f"Tokens/word on training slice: {train_tpw:.3f}")
