import torch
import torch.nn.functional as F
import gradio as gr
import os
import sys
import threading
import time
from model import MLP
from vocab import build_vocab

context_size = 3

# === Map datasets to trained models
model_file_map = {
    "names.txt": "model_names.pt",
    "pokemon.txt": "model_pokemon.pt",
    "fantasy.txt": "model_fantasy.pt",
}

# === Generate names from model
def generate_name(n, temp, dataset):
    print(f"→ Generating {n} names from {dataset} at temperature {temp}")
    try:
        with open(dataset, 'r') as f:
            words = f.read().splitlines()
    except FileNotFoundError:
        return f"[Error] Dataset '{dataset}' not found."

    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    model = MLP(vocab_size, context_size)
    model_path = model_file_map.get(dataset)
    if not os.path.exists(model_path):
        return f"[Error] Model '{model_path}' not found."

    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        return f"[Model Load Error] {e}"

    model.eval()
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

# === Restart app with new theme
def delayed_restart(new_theme):
    time.sleep(1.5)
    os.environ["APP_THEME"] = new_theme
    print(f"🌈 Restarting with theme: {new_theme}")
    os.execv(sys.executable, [sys.executable] + sys.argv)

def handle_theme_change(choice):
    new_theme = "default" if choice == "Light" else "gradio/soft"
    threading.Thread(target=delayed_restart, args=(new_theme,)).start()
    return "🔄 Switching theme... Please wait..."

# === Determine current theme
selected_theme = os.environ.get("APP_THEME", "default")
current_theme_name = "Light" if selected_theme == "default" else "Dark"

# === Gradio UI
with gr.Blocks(theme=selected_theme) as demo:
    gr.Markdown("## 🧠 Name Generator")

    with gr.Row():
        n_slider = gr.Slider(1, 50, value=10, step=1, label="Number of Names")
        temp_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Temperature")
        dataset_dropdown = gr.Dropdown(
            choices=["names.txt", "pokemon.txt", "fantasy.txt"],
            value="names.txt",
            label="Dataset"
        )

    output = gr.Textbox(label="Generated Names", lines=10)
    generate_btn = gr.Button("🔁 Generate")
    generate_btn.click(fn=generate_name, inputs=[n_slider, temp_slider, dataset_dropdown], outputs=output)

    gr.Markdown("---")
    gr.Markdown("🎨 **Toggle Theme**")

    theme_radio = gr.Radio(choices=["Light", "Dark"], value=current_theme_name, label="Choose Theme")

    loading_notice = gr.Textbox(
        value="",
        visible=True,
        interactive=False,
        show_label=False
    )

    theme_radio.change(fn=handle_theme_change, inputs=theme_radio, outputs=loading_notice)

# === Launch
if __name__ == "__main__":
    demo.launch()
