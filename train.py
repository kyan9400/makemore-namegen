import torch
import torch.nn.functional as F
import argparse
from model import MLP
from vocab import build_vocab

# === CLI args ===
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="names.txt", help="Path to dataset file")
args = parser.parse_args()

# === Load dataset ===
with open(args.file, 'r') as f:
    words = f.read().splitlines()
print("Dataset size:", len(words))

# === Build vocab ===
stoi, itos = build_vocab(words)
vocab_size = len(stoi)
print("Vocab size:", vocab_size)

# === Create training data ===
context_size = 3
X, Y = [], []

for w in words:
    context = [0] * context_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context[:])
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

# === Model setup ===
model = MLP(vocab_size, context_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === Train ===
losses = []
weight_decay = 0.01  # L2 regularization coefficient

for epoch in range(1000):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)

    # Apply L2 regularization
    l2_reg = sum((param**2).sum() for param in model.parameters())
    loss += weight_decay * l2_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Save ===
model_file = f"model_{args.file.replace('.txt', '')}.pt"
torch.save(model.state_dict(), model_file)
print(f"✅ Model saved to {model_file}")

# === Save loss curve ===
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title(f"Loss Curve - {args.file}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
print("📈 Loss curve saved to loss_curve.png")
