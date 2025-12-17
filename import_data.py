import csv
import regex_tokenizer as rt
import torch
from data_utils import stot

def _csv_to_string(csv_path: str, text_col: str = "headline_text"):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return "\n".join(row[text_col] for row in reader if row.get(text_col))
    
# text = _csv_to_string("Dataset/examiner-date-text-shuffled.csv")
text = open("Dataset/tinyshakespeare.txt", "r", encoding="utf-8").read()
data = stot(text)
