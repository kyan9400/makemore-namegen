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

# === Map datasets to trained models ===
model_file_map = {
    "names.txt": "model_names.pt",
    "pokemon.txt": "model_pokemon.pt",
    "fantasy.txt": "model_fantasy.pt",
}

# === Extract tags from dataset ===
def get_tags_for_dataset(dataset):
    try:
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return ["All"]
    tags = set()
    for line in lines:
        if '|' in line:
            _, tag = line.split('|')
            tags.add(tag.strip())
    return ["All"] + sorted(tags)

# === Core generation logic ===
def generate_name(n, temp, topk, dataset, prefix, tag_filter):
    print(f"→ Generating {n} names from {dataset} at temperature {temp}, top-k={topk}, prefix={prefix}, tag={tag_filter}")
    try:
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return f"[Error] Dataset '{dataset}' not found."

    # Split into names and tags
    names_raw, tags = [], []
    for line in lines:
        if '|' in line:
            name, tag = line.split('|')
            names_raw.append(name.strip())
            tags.append(tag.strip())
        else:
            names_raw.append(line.strip())
            tags.append("default")

    # Build vocab
    stoi, itos = build_vocab(names_raw)
    vocab_size = len(stoi)

    model_path = model_file_map.get(dataset)
    if not model_path or not os.path.exists(model_path):
        return f"[Error] Model '{model_path}' not found."

    model = MLP(vocab_size, context_size)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        return f"[Model Load Error] {e}"

    model.eval()
    results = []

    for _ in range(n * 2):
        context = [0] * context_size
        name = ''

        for ch in prefix:
            ix = stoi.get(ch.lower(), 0)
            name += ch
            context = context[1:] + [ix]

        while True:
            x = torch.tensor([context])
            logits = model(x) / temp
            probs = F.softmax(logits, dim=1)

            if topk > 0:
                k = min(topk, probs.shape[1])
                topk_vals, topk_idx = torch.topk(probs, k)
                probs = torch.zeros_like(probs).scatter_(1, topk_idx, topk_vals)
                probs = probs / probs.sum(dim=1, keepdim=True)

            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            name += itos[ix]
            context = context[1:] + [ix]

        results.append(name)
        if len(results) >= n * 2:
            break

    # Filter
    if tag_filter and tag_filter != "All":
        filtered = [name for name, tag in zip(names_raw, tags) if tag == tag_filter]
        results = [r for r in results if r in filtered]
        results = results[:n]
    else:
        results = results[:n]

    return "\n".join(results)

# === Theme toggling ===
def delayed_restart(new_theme):
    time.sleep(1.5)
    os.environ["APP_THEME"] = new_theme
    os.execv(sys.executable, [sys.executable] + sys.argv)

def handle_theme_change(choice):
    new_theme = "default" if choice == "Light" else "gradio/soft"
    threading.Thread(target=delayed_restart, args=(new_theme,)).start()
    return "🔄 Switching theme... Please wait..."

# === Launch UI ===
selected_theme = os.environ.get("APP_THEME", "default")
current_theme_name = "Light" if selected_theme == "default" else "Dark"

with gr.Blocks(theme=selected_theme) as demo:
    gr.Markdown("## 🧠 Name Generator")

    with gr.Row():
        n_slider = gr.Slider(1, 50, value=10, step=1, label="Number of Names")
        temp_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Temperature")
        topk_slider = gr.Slider(0, 50, value=0, step=1, label="Top-k Sampling")
        dataset_dropdown = gr.Dropdown(
            choices=["names.txt", "pokemon.txt", "fantasy.txt"],
            value="names.txt",
            label="Dataset"
        )

    with gr.Row():
        prefix_input = gr.Textbox(label="Start With (Prefix)", placeholder="e.g., El, Dra...")
        tag_dropdown = gr.Dropdown(label="Tag Filter", choices=["All"], value="All")

    output = gr.Textbox(label="Generated Names", lines=10)
    generate_btn = gr.Button("🔁 Generate")
    copy_btn = gr.Button("📋 Copy")

    # Trigger name generation
    generate_btn.click(
        fn=generate_name,
        inputs=[n_slider, temp_slider, topk_slider, dataset_dropdown, prefix_input, tag_dropdown],
        outputs=output
    )
    prefix_input.change(
        fn=generate_name,
        inputs=[n_slider, temp_slider, topk_slider, dataset_dropdown, prefix_input, tag_dropdown],
        outputs=output
    )

    # Update tags dynamically on dataset change
    dataset_dropdown.change(
        fn=get_tags_for_dataset,
        inputs=dataset_dropdown,
        outputs=tag_dropdown
    )

    # Copy to clipboard button
    copy_btn.click(
        fn=lambda txt: txt,
        inputs=output,
        outputs=None,
        js="navigator.clipboard.writeText(arguments[0]); alert('Copied!');"
    )

    gr.Markdown("---")
    theme_radio = gr.Radio(choices=["Light", "Dark"], value=current_theme_name, label="Choose Theme")
    loading_notice = gr.Textbox(value="", visible=True, interactive=False, show_label=False)
    theme_radio.change(fn=handle_theme_change, inputs=theme_radio, outputs=loading_notice)

if __name__ == "__main__":
    demo.launch()
