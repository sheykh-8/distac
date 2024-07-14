from flask import Flask, jsonify, request
from model.Tacotron2_partitions.hifigan_inference import get_inference
import os

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def handler():
    data = request.get_json()

    spectrogram = data.get("mel_outputs")

    if spectrogram is None:
        return {"error": "No spectrogram provided"}, 400

    audio_path = get_inference(spectrogram)

    # Send the file and then delete it
    # response = send_file(audio_path, as_attachment=True, attachment_filename='generated_audio.wav')

    # @response.call_on_close
    # def remove_file():
    #     try:
    #         os.remove(audio_path)
    #     except Exception as e:
    #         app.logger.error(f"Error removing file {audio_path}: {e}")

    return jsonify({"path": audio_path})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
