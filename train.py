import torch
import torch.nn.functional as F
from model import MLP
from vocab import build_vocab

# === Load dataset ===
with open('names.txt', 'r') as f:
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
for epoch in range(1000):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Generate names ===
print("\nGenerated names:")
temperature = 1.0

for _ in range(10):
    context = [0] * context_size
    name = ''
    while True:
        x = torch.tensor([context])
        logits = model(x) / temperature
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        name += itos[ix]
        context = context[1:] + [ix]
    print(name)

# === Save ===
torch.save(model.state_dict(), 'namegen_model.pt')
print("✅ Model saved to namegen_model.pt")
