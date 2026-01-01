# run_pretrained.py
import argparse
import torch
import config
from data_utils import ttos, stot
from model import GPTLanguageModel

# Utility functions
def Load_pretrained(checkpoint_path: str = "checkpoint.pt", training = False, device = config.device) -> GPTLanguageModel:
    model = GPTLanguageModel(cfg=config.cfg).to(device)
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

    model = Load_pretrained("checkpoint.pt")
    # model = GPTLanguageModel(cfg=config.cfg).to(config.device)
    # model = GPTLanguageModel.load_gpt2_from_hf().to(config.device) # for testing imported GPT-2, temporary
    

    while True:

        user_prompt = input("Enter your prompt (or '<q>' to quit): ")
        if user_prompt == "<q>":
            break
        prompt = stot(user_prompt).view(1, -1)

        prompt = prompt.to(config.device)
        # torch.manual_seed(42)
        # torch.mps.manual_seed(42)
        words_gen_string = generate_words(prompt, model, max_new_tokens=200, beta=1)

        print(words_gen_string)
        with open("generated.txt", "w") as f:
            f.write(words_gen_string)
        print("\nSaved to generated.txt.")

if __name__ == "__main__":
        main()
        
