import functions_framework
from flask import jsonify
from model.Tacotron2_partitions.tacotron_decoder_inference import get_inference

@functions_framework.http
def predict(request):
    """HTTP Cloud Function for Tacotron2 decoder inference.
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
    encoded_inputs = data.get("encoded_input")
    input_lengths = data.get("input_lengths")

    if encoded_inputs is None or input_lengths is None:
        return jsonify({"error": "encoded inputs and input lengths are required"}), 400

    mel_outputs, gate_outputs, alignments, mel_lengths = get_inference(
        encoded_inputs, input_lengths
    )

    return jsonify({
        "mel_outputs": mel_outputs.tolist(),
        "alignments": alignments.tolist(),
    })