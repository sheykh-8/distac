import functions_framework
from flask import jsonify, Request
from model.Tacotron2_partitions.hifigan_inference import get_inference

@functions_framework.http
def handler(request: Request):
    """HTTP Cloud Function for HiFi-GAN inference.
    Args:
        request (flask.Request): The request object.
    Returns:
        A JSON response containing the inference results or an error message.
    """
    if request.method != "POST":
        return jsonify({"error": "Only POST requests are accepted"}), 405
    if not request.is_json:
        return jsonify({"error": "Request data must be JSON"}), 400
    
    data = request.get_json()
    spectrogram = data.get("mel_outputs")
    if spectrogram is None:
        return jsonify({"error": "No spectrogram provided"}), 400
    
    audio_url = get_inference(spectrogram)
    
    return jsonify({
        "url": audio_url,
    })