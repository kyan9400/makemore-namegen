import torch
import torch.nn.functional as F
from model import MLP
from vocab import build_vocab
import gradio as gr

# Load dataset and build vocab
with open('names.txt', 'r') as f:
    words = f.read().splitlines()

context_size = 3
stoi, itos = build_vocab(words)
vocab_size = len(stoi)

# Load trained model
model = MLP(vocab_size, context_size)
model.load_state_dict(torch.load("namegen_model.pt"))
model.eval()

def generate_name(n=5, temp=1.0):
    names = []
    for _ in range(n):
        context = [0] * context_size
        name = ''
        while True:
            x = torch.tensor([context])
            logits = model(x)
            logits = logits / temp
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            name += itos[ix]
            context = context[1:] + [ix]
        names.append(name)
    return "\n".join(names)

# Create Gradio UI
demo = gr.Interface(
    fn=generate_name,
    inputs=[
        gr.Slider(1, 50, step=1, label="Number of Names"),
        gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="Temperature"),
    ],
    outputs="text",
    title="🧠 Name Generator",
    description="Generate character-level names using a trained neural network model."
)

if __name__ == "__main__":
    demo.launch(share=True)

