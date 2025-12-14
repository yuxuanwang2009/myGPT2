# run_pretrained.py
import argparse
import torch
from config import device, n_emb, n_heads, n_layers, T, dropout, n_ffd_hidden
from data_utils import vocab_size, ttos, stot
from model import GPTLanguageModel

# Utility functions
def load_pretrained(checkpoint_path: str = "checkpoint.pt", training = False) -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_emb=n_emb,
        n_heads=n_heads,
        n_ffd_hidden = n_ffd_hidden,
        n_layers=n_layers,
        T=T,
        dropout=dropout,
        device = device
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if training == True:
        optimizer = torch.optim.AdamW(model.parameters(), 0)
        optimizer.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optimizer
    else:
        model.eval()
        return model

def generate_words(prompt: str, model: GPTLanguageModel, max_new_tokens: int = 3000, beta: float = 1.0):
    with torch.no_grad():
        generated = model.generate(
            idx=prompt,
            max_new_tokens=max_new_tokens,
            beta=beta,
        )
    words_gen_string = ttos(generated.view(-1),for_output=True)
    return words_gen_string

def main():
    # CLI options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", "-p",
        action="store_true",
        help="Enter a prompt interactively."
    )
    args = parser.parse_args()

    model = load_pretrained("checkpoint.pt")

    while True:
        if args.prompt:
            user_prompt = input("\nEnter your prompt (or '<q>' to quit): ")
            if user_prompt == "<q>":
                break
            prompt = stot("<|endoftext|>" + user_prompt).view(1, -1)
        else:
            prompt = stot("<|endoftext|>").view(1, -1)

        prompt = prompt.to(device)
        words_gen_string = generate_words(prompt, model, max_new_tokens=3000, beta=1.5)

        with open("generated.txt", "w") as f:
            f.write(words_gen_string)
        print("\nHere's your headline:", ''.join(words_gen_string.splitlines()[:2]))
        print("\nMore generated headlines written to generated.txt.")
        
        if not args.prompt:
            break

if __name__ == "__main__":
        main()
        
