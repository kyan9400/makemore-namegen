import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, vocab_size, context_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 10)
        self.fc1 = nn.Linear(context_size * 10, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
