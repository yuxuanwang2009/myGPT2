from utils import Overlap, Visualize_model, vec
from run_pretrained import load_pretrained
from data_utils import ttos
import torch

model = load_pretrained("checkpoint.pt")
# Visualize_model(model)
print(vec("walk", model))