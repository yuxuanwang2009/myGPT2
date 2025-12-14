import os

from config import vocab_size
import csv
import matplotlib.pyplot as plt
import regex_tokenizer as rt

CSV_SOURCE = os.environ.get("CSV_SOURCE")
if not CSV_SOURCE:
    raise ValueError("Set CSV_SOURCE to the path/URL of the CSV file.")


def csv_to_string(csv_path: str, text_col: str = "headline_text"):
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return "\n".join(row[text_col] for row in reader if row.get(text_col))
    
# text = csv_to_string(CSV_SOURCE)
# text_tok_training = text[:int(len(text)/20)] 

# tok = rt.RegexTokenizer.train(text_tok_training, vocab_size, path="tokenizer.json", verbose=True)

tok = rt.RegexTokenizer.load("tokenizer.json")
print(tok.encode("\n")[0])
