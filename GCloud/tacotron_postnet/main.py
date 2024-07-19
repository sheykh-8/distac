import functions_framework
from flask import jsonify
from model.Tacotron2_partitions.tacotron_postnet_inference import get_inference


@functions_framework.http
def predict(request):
    """HTTP Cloud Function for Tacotron2 inference.
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
    mel_outputs = data.get("mel_outputs")
    alignments = data.get("alignments")

    if mel_outputs is None or alignments is None:
        return jsonify({"error": "mel outputs and alignments are required"}), 400

    mel_outputs_postnet, alignments = get_inference(mel_outputs, alignments)

    return jsonify(
        {
            "mel_outputs": mel_outputs_postnet.tolist(),
            "alignments": alignments.tolist(),
        }
    )
