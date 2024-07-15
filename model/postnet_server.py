# app.py
from flask import Flask, request, jsonify
from model.Tacotron2_partitions.tacotron_postnet_inference import get_inference

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Request data must be JSON"}), 400
        data = request.get_json()
        mel_outputs = data.get("mel_outputs")
        alignments = data.get("alignments")
        if mel_outputs is None or alignments is None:
            return (
                jsonify({"error": "mel output and alignments is required"}),
                400,
            )

        mel_outputs_postnet, alignments = get_inference(
          mel_outputs, alignments
        )
        return jsonify(
            {
                "mel_outputs": mel_outputs_postnet.tolist(),
                "alignments": alignments.tolist(),
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
