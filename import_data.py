import csv
import regex_tokenizer as rt
import torch


def csv_to_string(csv_path: str, text_col: str = "headline_text"):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return "\n".join(row[text_col] for row in reader if row.get(text_col))
    
text = csv_to_string("Dataset/examiner-date-text-shuffled.csv")[:100000]

tok = rt.RegexTokenizer.load("tokenizer.json")


# tokenizer encoding: from a string to a tensor of tokens, wrapped with my compactifaction scheme
def stot(s: str) -> torch.Tensor:
    ids = tok.encode(s)
    return torch.tensor(ids, dtype=torch.long)

# tokenizer encoding: from a tensor of tokens to a strong, wrapped with my compactification scheme
def ttos(t: torch.Tensor, for_output: bool = False) -> str:
    out = tok.decode(t.tolist())
    return out.replace("<|endoftext|>", "\n") if for_output else out

data = stot(text)