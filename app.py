"""Gradio web interface for ContextQA.  Run:  python app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from contextqa.config import DEFAULT_MODEL_KEY, MODELS  # noqa: E402
from contextqa.ui import answer_question, load_samples  # noqa: E402
from contextqa.utils import setup_logging  # noqa: E402


def build_demo():
    """Create the Gradio Blocks app (Gradio is imported lazily)."""
    import gradio as gr

    def run(model_key, context, question, progress=gr.Progress()):
        progress(0.1, desc="Loading model (first use downloads it) and running inference...")
        return answer_question(model_key, context, question)

    with gr.Blocks(title="ContextQA") as demo:
        gr.Markdown("# ContextQA\nExtractive question answering with Transformer encoders. "
                    "The answer is always a span copied from your context.")
        with gr.Row():
            with gr.Column(scale=3):
                context = gr.Textbox(label="Context", lines=10, placeholder="Paste a passage here...")
                question = gr.Textbox(label="Question", lines=1)
                model = gr.Dropdown(
                    choices=[(m.display_name, k) for k, m in MODELS.items()],
                    value=DEFAULT_MODEL_KEY, label="Model",
                )
                button = gr.Button("Answer", variant="primary")
            with gr.Column(scale=2):
                answer = gr.Textbox(label="Answer", interactive=False)
                score = gr.Textbox(label="Model score (not a probability of correctness)",
                                   lines=4, interactive=False)
                status = gr.Textbox(label="Status", interactive=False)
        samples = load_samples()
        if samples:
            gr.Examples(examples=samples, inputs=[context, question], label="Samples")
        inputs, outputs = [model, context, question], [answer, score, status]
        button.click(run, inputs, outputs, show_progress="full")
        question.submit(run, inputs, outputs, show_progress="full")
    return demo


if __name__ == "__main__":
    setup_logging()
    build_demo().launch()
