import functions_framework
from flask import jsonify
from model.Tacotron2_partitions.tacotron_encoder_inference import get_inference


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
    text = data.get("text")

    if text is None:
        return jsonify({"error": "No text field provided"}), 400

    print(f"this is the text: {text}")
    prediction = get_inference(text)

    return jsonify({
        "encoded_input": prediction[0].tolist(),
        "input_lengths": prediction[1].tolist(),
    })