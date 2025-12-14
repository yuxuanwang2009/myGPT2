# Misc utilities
import torch
from torchview import draw_graph
from IPython.display import display, Image
from model import GPTLanguageModel
from data_utils import stot, vocab_size
from config import T, n_emb
import io, matplotlib.pyplot as plt, matplotlib.image as mpimg
import time


# Visualize the model
def Visualize_model(model):
    dummy = torch.randint(0, vocab_size, (1, T))

    # full model
    graph = draw_graph(model, input_data=dummy, expand_nested=True, depth=4)
    png_full = graph.visual_graph.pipe(format="png")

    # first MultiHeadAttention
    mha = model.blocks[0].atten[1]
    mha_dummy = torch.zeros(1, T, n_emb)
    mha_graph = draw_graph(mha, input_data=mha_dummy, expand_nested=True, depth=2)
    png_mha = mha_graph.visual_graph.pipe(format="png")

    # show both
    plt.figure(); plt.imshow(mpimg.imread(io.BytesIO(png_full), format="png")); plt.axis("off"); plt.title("Full model")
    plt.figure(); plt.imshow(mpimg.imread(io.BytesIO(png_mha), format="png")); plt.axis("off"); plt.title("First MHA")
    plt.show(block=False)
    plt.pause(0.1)
    # time.sleep(10**6)  # effectively keeps the process alive until you close the windows
    
    # keep windows open until you press Enter
    input("Press Enter to exit...") 


# Fun with embedding table
def vec(s: str, m: GPTLanguageModel) -> torch.tensor:
    ix = stot(s)
    vector = m.token_embedding_table(ix)
    vector /= vector.norm()
    return vector

def Overlap(letter_a: str, letter_b: str, m: GPTLanguageModel) -> float:
    print('The overlap between', repr(letter_a), "and", repr(letter_b), "is",  f"{(vec(letter_a, m) @ vec(letter_b, m).T).item():.4f}")

# for i in range(27):
    # overlap('w', ttos(torch.tensor([i], dtype=torch.long)))