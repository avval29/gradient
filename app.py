"""Gradient: Gradio UI. Run with `python app.py`.
See gradient/config.py for the GRADIENT_* environment switches."""
import os

import gradio as gr

from gradient.config import make_backend, make_encoder, make_retriever
from gradient.generator import generate
from gradient.grading import finalize, grade_answer, log_final, parse_rubric_text
from gradient.qa import generate_quiz

retriever = make_retriever()
backend = make_backend()
encoder = make_encoder()
LOG_PATH = os.getenv("GRADIENT_LOG", "data/private/grading_log.jsonl")

EXAMPLE_RUBRIC = """# points | name | description | keywords (use | for alternatives, commas between keywords)
2 | Inputs | Plants use sunlight, water and carbon dioxide. | sunlight|light, water, carbon dioxide|co2
1 | Where | It happens in the leaves, in chloroplasts with chlorophyll. | leaf|leaves|chloroplast, chlorophyll
2 | Products | It produces glucose (food) and releases oxygen. | glucose|sugar|food|starch, oxygen"""


# ---------- tab 1: generate ----------
def run(mode, topic, grade, difficulty):
    if mode == "quiz (Q&A)":
        res = generate_quiz(topic, retriever, backend)
    else:
        res = generate(mode, topic, int(grade), difficulty, retriever, backend)
    status = "Grounded in: " + ", ".join(res.sources) if res.grounded else "Not grounded - refused"
    if res.rejected:
        status += f" | {res.rejected} question(s) dropped: not supported by your notes"
    return res.text, status


# ---------- tab 2: grade (draft only) ----------
def draft_grade(question, rubric_text, answer):
    if not answer.strip():
        return "Enter a student answer first.", None, "", None
    try:
        rubric = parse_rubric_text(question, rubric_text)
    except ValueError as e:
        return f"Rubric problem: {e}", None, "", None
    draft = grade_answer(answer, rubric, encoder)
    rows = ["| Criterion | Points | Evidence | Found | Missing |", "|---|---|---|---|---|"]
    for r in draft.results:
        rows.append(f"| {r.name} | {r.points_awarded:g}/{r.points_possible:g} | {r.evidence:.2f} | "
                    f"{', '.join(r.matched) or '-'} | {', '.join(r.missing) or '-'} |")
    notes = [f'- **{r.name}**: "{r.best_sentence}"' for r in draft.results if r.best_sentence]
    flag = ("\n\n**Please review carefully:** " + "; ".join(draft.review_reasons)) if draft.needs_review else ""
    table = "\n".join(rows) + flag + ("\n\nEvidence sentences:\n" + "\n".join(notes) if notes else "")
    return table, draft.suggested_score, draft.feedback, (draft, answer, rubric)


def approve(teacher, score, feedback, state):
    if state is None:
        return "Draft a grade first."
    draft, answer, rubric = state
    try:
        final = finalize(draft, teacher, None if score is None else float(score), feedback)
    except ValueError as e:
        return f"Not saved: {e}"
    log_final(final, draft, rubric, answer, LOG_PATH)
    note = f" (you changed the suggested {final.suggested_score:g})" if final.overridden else ""
    return f"Saved: {final.score:g}/{final.max_score:g} approved by {final.teacher}{note}."


with gr.Blocks(title="Gradient") as demo:
    gr.Markdown("# Gradient\nAI co-pilot for teachers. Drafts only; you approve and edit.")
    with gr.Tab("Generate"):
        with gr.Row():
            mode = gr.Dropdown(["quiz (Q&A)", "worksheet", "lesson plan"], value="quiz (Q&A)", label="Output")
            grade = gr.Slider(1, 12, value=7, step=1, label="Class")
            difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty")
        topic = gr.Textbox(label="Topic", placeholder="e.g. photosynthesis")
        btn = gr.Button("Generate draft", variant="primary")
        out = gr.Textbox(label="Draft (editable)", lines=14, interactive=True)
        src = gr.Textbox(label="Sources", interactive=False)
        btn.click(run, [mode, topic, grade, difficulty], [out, src])

    with gr.Tab("Grade (draft)"):
        gr.Markdown("Gradient suggests a score with evidence. **Nothing is final until you approve it.**")
        question = gr.Textbox(label="Question", value="Explain how plants make food by photosynthesis.")
        rubric_text = gr.Textbox(label="Rubric", value=EXAMPLE_RUBRIC, lines=6)
        answer = gr.Textbox(label="Student answer (typed)", lines=4)
        draft_btn = gr.Button("Draft grade", variant="primary")
        breakdown = gr.Markdown()
        with gr.Row():
            teacher = gr.Textbox(label="Your name (required to approve)")
            final_score = gr.Number(label="Final score (edit if you disagree)")
        final_feedback = gr.Textbox(label="Feedback to student (editable)", lines=4)
        approve_btn = gr.Button("Approve and save")
        status = gr.Markdown()
        state = gr.State()
        draft_btn.click(draft_grade, [question, rubric_text, answer], [breakdown, final_score, final_feedback, state])
        approve_btn.click(approve, [teacher, final_score, final_feedback, state], status)

if __name__ == "__main__":
    demo.launch()
