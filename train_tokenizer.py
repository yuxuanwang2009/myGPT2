import io
import os
import csv
from urllib.parse import urlparse
from urllib.request import urlopen

from config import vocab_size
import matplotlib.pyplot as plt
import regex_tokenizer as rt

DEFAULT_CSV_URL = "https://drive.google.com/drive/folders/1qiZULwjOETORmWU_eG_L6h5cDHqv38KL?usp=share_link"
CSV_SOURCE = os.environ.get("CSV_SOURCE") or (
    os.path.join(os.environ["CSV_DIR"], "examiner-date-text-shuffled.csv") if "CSV_DIR" in os.environ else None
)
if not CSV_SOURCE:
    CSV_SOURCE = DEFAULT_CSV_URL


def _open_csv_source(csv_source: str):
    parsed = urlparse(csv_source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(csv_source) as resp:
            content = resp.read().decode("utf-8")
        return io.StringIO(content)
    return open(csv_source, "r", encoding="utf-8", newline="")


def csv_to_string(csv_path: str, text_col: str = "headline_text"):
    with _open_csv_source(csv_path) as f:
        reader = csv.DictReader(f)
        return "\n".join(row[text_col] for row in reader if row.get(text_col))
    
# text = csv_to_string(CSV_SOURCE)
# text_tok_training = text[:int(len(text)/20)] 

# tok = rt.RegexTokenizer.train(text_tok_training, vocab_size, path="tokenizer.json", verbose=True)

tok = rt.RegexTokenizer.load("tokenizer.json")
print(tok.encode("\n")[0])
