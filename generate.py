import torch
import torch.nn.functional as F
import argparse
from model import MLP
from vocab import build_vocab

context_size = 3

# === Argument parser ===
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="names.txt", help="Dataset file (e.g. names.txt)")
parser.add_argument("--n", type=int, default=10, help="Number of names to generate")
parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--topk", type=int, default=0, help="Top-k sampling cutoff (0 = no top-k)")
args = parser.parse_args()

# === Load data ===
with open(args.file, 'r') as f:
    words = f.read().splitlines()

stoi, itos = build_vocab(words)
vocab_size = len(stoi)

# === Load model ===
model_file = f"model_{args.file.replace('.txt', '')}.pt"
model = MLP(vocab_size, context_size)
model.load_state_dict(torch.load(model_file))
model.eval()
print(f"✅ Model loaded. Generating from: {args.file}\n")
print(f"🔁 Generating {args.n} names (temperature = {args.temp}, top-k = {args.topk}):\n")

# === Generate names ===
for _ in range(args.n):
    context = [0] * context_size
    name = ''
    while True:
        x = torch.tensor([context])
        logits = model(x) / args.temp
        probs = F.softmax(logits, dim=1)

        if args.topk > 0:
            topk = min(args.topk, probs.shape[1])
            topk_vals, topk_idx = torch.topk(probs, topk)
            probs = torch.zeros_like(probs).scatter_(1, topk_idx, topk_vals)
            probs = probs / probs.sum(dim=1, keepdim=True)

        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        name += itos[ix]
        context = context[1:] + [ix]
    print(name)
