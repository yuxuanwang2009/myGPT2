"""Regex-based tokenizer that splits text with a GPT-4-style pattern, trains BPE merges,
supports encode/decode, and can save/load its learned state.

Quick usage:
    tok = RegexTokenizer.train(corpus_text, vocab_size=10_000, path="tokenizer.json")
    ids = tok.encode("hello world")
    text = tok.decode(ids)

    # later/restart
    tok = RegexTokenizer.load("tokenizer.json")
"""

import regex as re
import json


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self, merges=None, vocab=None, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {} if merges is None else merges
        self.vocab = {i: bytes([i]) for i in range(256)} if vocab is None else vocab

    def merge(self, ids, pair, new_token, byte_shuffle=None):
        """Replace all non-overlapping occurrences of pair with new_token."""
        i = 0
        out_ids = []
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out_ids.append(new_token)
                i += 2
            else:
                out_ids.append(ids[i])
                i += 1
        return out_ids

    def _encode_chunk(self, chunk_bytes, byte_shuffle=None):
        ids = list(chunk_bytes)
        if byte_shuffle is not None:
            for i, id in enumerate(ids):
                if id<256:
                    ids[i] = byte_shuffle[id]
        for pair, new_token in self.merges.items():
            ids = self.merge(ids, pair, new_token, byte_shuffle)                
        return ids

    @classmethod
    def train(cls, text, vocab_size, path, special_characters: list[str] = None, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokenizer = cls()
        chunks = re.findall(tokenizer.compiled_pattern, text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks] # list converts bytes to a list of ints

        for step in range(num_merges):
            counts = {}
            for ids in chunk_ids:
                for a, b in zip(ids, ids[1:]):
                    pair = (a, b)
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            next_pair = max(counts, key=counts.get)
            new_token = 256 + step
            tokenizer.merges[next_pair] = new_token
            tokenizer.vocab[new_token] = tokenizer.vocab[next_pair[0]] + tokenizer.vocab[next_pair[1]]
            if verbose:
                render_token = None
                translation_table = str.maketrans({" ": "<sp>", "\n": "<nl>"})
                def render_token(token_ids):
                    return tokenizer.decode(token_ids).translate(translation_table)

                print(
                        f"merge {step+1}/{num_merges}: {render_token(next_pair[:1])}, "
                        f"{render_token(next_pair[1:])} -> {render_token([new_token])} "
                        f"had {counts[next_pair]} occurrences"
                    )
            chunk_ids = [tokenizer.merge(ids, next_pair, new_token) for ids in chunk_ids]
        tokenizer.save(path)
        return tokenizer

    def encode(self, text, byte_shuffle=None):
        chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8"), byte_shuffle))
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        merges_serialized = [(a, b, tok) for (a, b), tok in self.merges.items()]
        state = {"pattern": self.pattern, "merges": merges_serialized}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for a, b, tok in state["merges"]:
            merges[(a, b)] = tok
            vocab[tok] = vocab[a] + vocab[b]
        return cls(merges=merges, vocab=vocab, pattern=state.get("pattern", GPT4_SPLIT_PATTERN))
