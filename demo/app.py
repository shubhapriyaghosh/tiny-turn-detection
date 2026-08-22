import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import TurnDetector

MODEL_PATH = "checkpoints/tiny_turn_net.onnx"
CONFIG_PATH = "configs/config.yaml"

detector = TurnDetector(MODEL_PATH, CONFIG_PATH)

LABELS = {False: "Still speaking / pause (don't interrupt)", True: "End of turn (safe to respond)"}


def predict(audio_path):
    if audio_path is None:
        return "No audio provided.", {}
    result = detector.predict_endpoint(audio_path)
    result_md = f"**{LABELS[result['is_end_of_turn']]}**\n\n_inference: {result['latency_ms']:.2f} ms_"
    confidence = {
        LABELS[True]: result["probs"]["end_of_turn"],
        LABELS[False]: result["probs"]["continuing"],
    }
    return result_md, confidence


with gr.Blocks(title="Tiny Turn Detection — Shiprocket Challenge") as demo:
    gr.Markdown(
        "# Tiny Turn Detection\n"
        "Record or upload the trailing ~2 seconds of a speaker's audio. "
        "TinyTurnNet (523K params, from-scratch CNN+GRU + prosodic features) predicts "
        "whether they've finished their turn or are pausing (including Hinglish fillers)."
    )
    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Trailing audio window")
    run_btn = gr.Button("Predict", variant="primary")
    result_md = gr.Markdown()
    confidence_out = gr.Label(label="Class probabilities")
    run_btn.click(fn=predict, inputs=audio_in, outputs=[result_md, confidence_out])

if __name__ == "__main__":
    demo.launch()
