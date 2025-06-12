import argparse
import torch
import torch.nn.functional as F
from model import MLP
from vocab import build_vocab

# === CLI Arguments ===
parser = argparse.ArgumentParser(description="Generate names from a trained model.")
parser.add_argument('--file', type=str, default='names.txt', help='Path to dataset file')
parser.add_argument('--n', type=int, default=10, help='Number of names to generate')
parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature (0.8 = stable, 1.2 = creative)')
args = parser.parse_args()

# === Load dataset to rebuild vocab ===
with open(args.file, 'r') as f:
    words = f.read().splitlines()

stoi, itos = build_vocab(words)
vocab_size = len(stoi)
context_size = 3

# === Load trained model ===
model = MLP(vocab_size, context_size)
model.load_state_dict(torch.load('namegen_model.pt'))
model.eval()
print(f"✅ Model loaded. Generating from: {args.file}")

# === Generate names ===
print(f"\n🔁 Generating {args.n} names (temperature = {args.temp}):\n")

for _ in range(args.n):
    context = [0] * context_size
    name = ''
    while True:
        x = torch.tensor([context])
        logits = model(x) / args.temp
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        name += itos[ix]
        context = context[1:] + [ix]
    print(name)
