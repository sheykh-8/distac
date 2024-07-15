# app.py
from flask import Flask, request, jsonify
from model.Tacotron2_partitions.tacotron_encoder_inference import get_inference

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Request data must be JSON"}), 400
        data = request.get_json()
        text = data.get("text")
        if text is None:
            return jsonify({"error": "No text field provided"}), 400
        print(f"this is the text: {text}")
        prediction = get_inference(text)
        return jsonify(
            {
                "encoded_input": prediction[0].tolist(),
                "input_lengths": prediction[1].tolist(),
            }
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
