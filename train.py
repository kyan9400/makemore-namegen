import torch
import torch.nn.functional as F
import json
from model import MLP
from vocab import build_vocab
import sys

context_size = 3
dataset_file = "names.txt"
if len(sys.argv) > 1:
    dataset_file = sys.argv[1]

# Load data
with open(dataset_file, 'r') as f:
    words = f.read().splitlines()
print("Dataset size:", len(words))

# Vocab
stoi, itos = build_vocab(words)
vocab_size = len(stoi)
print("Vocab size:", vocab_size)

# Prepare data
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

# Model
model = MLP(vocab_size, context_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Log for loss plotting
loss_log = []

for epoch in range(1000):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        loss_log.append({"epoch": epoch, "loss": loss.item()})

# Save model
torch.save(model.state_dict(), "namegen_model.pt")
print("✅ Model saved to namegen_model.pt")

# Save loss log
with open("loss_log.json", "w") as f:
    json.dump(loss_log, f, indent=2)
print("✅ Loss log saved to loss_log.json")
