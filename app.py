"""Gradient: Gradio UI. Run with `python app.py`.
See gradient/config.py for the GRADIENT_* environment switches."""
import gradio as gr

from gradient.config import make_backend, make_retriever
from gradient.generator import generate
from gradient.qa import generate_quiz

retriever = make_retriever()
backend = make_backend()


def run(mode, topic, grade, difficulty):
    if mode == "quiz (Q&A)":
        res = generate_quiz(topic, retriever, backend)
    else:
        res = generate(mode, topic, int(grade), difficulty, retriever, backend)
    status = "Grounded in: " + ", ".join(res.sources) if res.grounded else "Not grounded - refused"
    if res.rejected:
        status += f" | {res.rejected} question(s) dropped: not supported by your notes"
    return res.text, status


with gr.Blocks(title="Gradient") as demo:
    gr.Markdown("# Gradient\nAI co-pilot for teachers. Drafts only; you approve and edit.")
    with gr.Row():
        mode = gr.Dropdown(["quiz (Q&A)", "worksheet", "lesson plan"], value="quiz (Q&A)", label="Output")
        grade = gr.Slider(1, 12, value=7, step=1, label="Class")
        difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
    topic = gr.Textbox(label="Topic", placeholder="e.g. photosynthesis")
    btn = gr.Button("Generate draft", variant="primary")
    out = gr.Textbox(label="Draft (editable)", lines=14, interactive=True)
    src = gr.Textbox(label="Sources", interactive=False)
    btn.click(run, [mode, topic, grade, difficulty], [out, src])

if __name__ == "__main__":
    demo.launch()
