import torch
import torch.nn.functional as F
import gradio as gr
from model import MLP
from vocab import build_vocab

context_size = 3

# === File mapping: dataset → trained model checkpoint ===
model_file_map = {
    "names.txt": "model_names.pt",
    "pokemon.txt": "model_pokemon.pt",
    "fantasy.txt": "model_fantasy.pt",
}

def generate_name(n, temp, dataset):
    # Load dataset and build vocab
    with open(dataset, 'r') as f:
        words = f.read().splitlines()

    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    # Load correct model
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
with gr.Blocks(title="🧠 Name Generator", theme="default") as demo:
    gr.Markdown("# 🧠 Neural Name Generator")
    gr.Markdown("Generate creative names from trained models based on your selected dataset.")

    with gr.Row():
        dataset = gr.Dropdown(choices=["names.txt", "pokemon.txt", "fantasy.txt"], value="names.txt", label="Dataset")
        count = gr.Slider(1, 50, value=10, step=1, label="Number of Names")
        temp = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Creativity (Temperature)")

    output = gr.Textbox(label="Generated Names", lines=10)

    generate_btn = gr.Button("✨ Generate Names")
    generate_btn.click(fn=generate_name, inputs=[count, temp, dataset], outputs=output)

if __name__ == "__main__":
    demo.launch()
