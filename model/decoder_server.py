# app.py
from flask import Flask, request, jsonify
from model.Tacotron2_partitions.tacotron_decoder_inference import get_inference

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Request data must be JSON"}), 400
        data = request.get_json()
        encoded_inputs = data.get("encoded_input")
        input_lengths = data.get("input_lengths")
        if encoded_inputs is None or input_lengths is None:
            return (
                jsonify({"error": "encoded inupts and input lengths is required"}),
                400,
            )

        mel_outputs, gate_outputs, alignments, mel_lengths = get_inference(
            encoded_inputs, input_lengths
        )
        return jsonify(
            {
                "mel_outputs": mel_outputs.tolist(),
                "alignments": alignments.tolist(),
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
