"""Gradient: Gradio UI. Run with `python app.py`.
Set GRADIENT_BACKEND=hf to use FLAN-T5 (needs transformers + torch)."""
import os

import gradio as gr

from gradient.generator import EchoBackend, HFBackend, generate
from gradient.rag import Retriever, load_corpus

retriever = Retriever(load_corpus(os.getenv("GRADIENT_DATA", "data/sample")))
backend = HFBackend() if os.getenv("GRADIENT_BACKEND") == "hf" else EchoBackend()


def run(mode, topic, grade, difficulty):
    res = generate(mode, topic, int(grade), difficulty, retriever, backend)
    status = "Grounded in: " + ", ".join(res.sources) if res.grounded else "Not grounded - refused"
    return res.text, status


with gr.Blocks(title="Gradient") as demo:
    gr.Markdown("# Gradient\nAI co-pilot for teachers. Drafts only; you approve and edit.")
    with gr.Row():
        mode = gr.Dropdown(["worksheet", "lesson plan"], value="worksheet", label="Output")
        grade = gr.Slider(1, 12, value=7, step=1, label="Class")
        difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
    topic = gr.Textbox(label="Topic", placeholder="e.g. photosynthesis")
    btn = gr.Button("Generate draft", variant="primary")
    out = gr.Textbox(label="Draft (editable)", lines=14, interactive=True)
    src = gr.Textbox(label="Sources", interactive=False)
    btn.click(run, [mode, topic, grade, difficulty], [out, src])

if __name__ == "__main__":
    demo.launch()
