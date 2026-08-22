"""
inference.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from src.features import FeatureConfig, extract


class TurnDetector:
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml", num_threads: int = 1):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.feature_cfg = FeatureConfig(
            sample_rate=cfg["audio"]["sample_rate"],
            window_seconds=cfg["audio"]["window_seconds"],
            n_mels=cfg["audio"]["n_mels"],
            n_fft=cfg["audio"]["n_fft"],
            hop_length=cfg["audio"]["hop_length"],
            fmin=cfg["audio"]["fmin"],
            fmax=cfg["audio"]["fmax"],
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def _softmax(self, logits):
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict_endpoint(self, audio_path: str, end_of_turn_threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()
        mel, prosody = extract(audio_path, self.feature_cfg)
        mel_input = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
        prosody_input = prosody[np.newaxis, :].astype(np.float32)

        logits = self.session.run(
            [self.output_name],
            {self.input_names[0]: mel_input, self.input_names[1]: prosody_input},
        )[0]
        probs = self._softmax(logits)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        is_end_of_turn = bool(probs[1] >= end_of_turn_threshold)
        return {
            "is_end_of_turn": is_end_of_turn,
            "confidence": float(probs[1]),
            "probs": {"continuing": float(probs[0]), "end_of_turn": float(probs[1])},
            "latency_ms": latency_ms,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/tiny_turn_net.onnx")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"{args.model} not found. Run the export step first.")

    detector = TurnDetector(args.model, args.config)
    result = detector.predict_endpoint(args.audio, end_of_turn_threshold=args.threshold)
    label = "END OF TURN" if result["is_end_of_turn"] else "STILL SPEAKING / PAUSE"
    print(f"{label}  (confidence={result['confidence']:.3f}, latency={result['latency_ms']:.2f}ms)")


if __name__ == "__main__":
    main()
