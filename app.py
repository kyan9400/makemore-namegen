import torch
import torch.nn.functional as F
import gradio as gr
from model import MLP
from vocab import build_vocab

context_size = 3

# === File mapping: dataset → correct trained model ===
model_file_map = {
    "names.txt": "model_names.pt",
    "pokemon.txt": "model_pokemon.pt",
    "fantasy.txt": "model_fantasy.pt",
}

# === Core generation logic ===
def generate_name(n, temp, dataset):
    # Load dataset
    with open(dataset, 'r') as f:
        words = f.read().splitlines()

    # Rebuild vocab + model
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    model = MLP(vocab_size, context_size)
    model.load_state_dict(torch.load(model_file_map[dataset]))
    model.eval()

    # Generate names
    names = []
    for _ in range(n):
        context = [0] * context_size
        name = ''
        while True:
            x = torch.tensor([context])
            logits = model(x) / temp
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            name += itos[ix]
            context = context[1:] + [ix]
        names.append(name)
    return "\n".join(names)

# === Gradio UI ===
demo = gr.Interface(
    fn=generate_name,
    inputs=[
        gr.Slider(1, 50, value=10, step=1, label="Number of Names"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Dropdown(choices=["names.txt", "pokemon.txt", "fantasy.txt"], value="names.txt", label="Dataset")
    ],
    outputs="text",
    title="🧠 Name Generator",
    description="Generate character-level names using a trained neural network model. Choose your dataset and temperature."
)

if __name__ == "__main__":
    demo.launch()
