import functions_framework
from flask import jsonify
import requests

from google.auth.transport.requests import Request
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = 'keys/sheykh-c68e3-7ce67e219356.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

@functions_framework.http
def process(request):
    """HTTP Cloud Function for orchestrating audio processing.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # Check if request method is POST
    if request.method != 'POST':
        return jsonify({"error": "Only POST requests are accepted"}), 405

    # Check if request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    text = data.get("text")
    
    if text is None:
      return jsonify({"error": "Request must contain text"}), 400
  
    # Call the target function securely
    def get_auth_headers():
        # Refresh the credentials
        credentials.refresh(Request())
        print(credentials.token)
        return {
            'Authorization': f'Bearer {credentials.token}'
        }

    try:
        # Step 1: Call Encoder Service
        encoder_response = requests.post('https://us-central1-sheykh-c68e3.cloudfunctions.net/tacotron_encoder', json=data, headers=get_auth_headers())
        encoder_response.raise_for_status()
        encoded_data = encoder_response.json()

        # Step 2: Call Decoder Service
        decoder_response = requests.post('https://us-central1-sheykh-c68e3.cloudfunctions.net/tacotron_decoder', json=encoded_data)
        decoder_response.raise_for_status()
        decoded_data = decoder_response.json()

        # Step 3: Call Postnet Service
        postnet_response = requests.post('https://us-central1-sheykh-c68e3.cloudfunctions.net/tacotron_postnet', json=decoded_data)
        postnet_response.raise_for_status()
        postprocessed_data = postnet_response.json()

        # Step 4: Call hifigan Service
        vocoder_response = requests.post('https://us-central1-sheykh-c68e3.cloudfunctions.net/tacotron_hifigan', json=postprocessed_data)
        vocoder_response.raise_for_status()
        final_response = vocoder_response.json()

        # Return the JSON response from the hifigan service
        return jsonify(final_response)

    except requests.exceptions.RequestException as e:
        # Handle any errors that occurred during the requests
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500