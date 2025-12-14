import os
import csv
from config import vocab_size
import matplotlib.pyplot as plt
import regex_tokenizer as rt

text = open("Dataset/tinyshakespeare.txt", "r", encoding="utf-8").read()
text_tok_training = text

tok = rt.RegexTokenizer.train(text_tok_training, vocab_size, path="tokenizer.json", verbose=True)

# tok = rt.RegexTokenizer.load("tokenizer.json")

def token_per_word(text: str, tokenizer: rt.RegexTokenizer) -> float:
    """Compute average tokens per whitespace-delimited word for a text sample."""
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    token_count = len(tokenizer.encode(text))
    return token_count / word_count


# train_tpw = token_per_word(text_tok_training, tok)
# full_tpw = token_per_word(text, tok)
# print(f"Tokens/word on training slice: {train_tpw:.3f}")
# print(f"Tokens/word on full corpus:   {full_tpw:.3f}")
