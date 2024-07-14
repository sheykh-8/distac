from flask import Flask, request, send_file
import requests
import os

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    
    # Step 1: Call Encoder Service
    encoder_response = requests.post('http://encoder:80/predict', json=data)
    encoded_data = encoder_response.json()
    
    # Step 2: Call Decoder Service
    decoder_response = requests.post('http://decoder:80/predict', json=encoded_data)
    decoded_data = decoder_response.json()
    
    # Step 3: Call Postnet Service
    postnet_response = requests.post('http://postnet:80/predict', json=decoded_data)
    postprocessed_data = postnet_response.json()
    
    # Step 4: Call Feedforward Service
    feedforward_response = requests.post('http://hifigan:80/generate', json=postprocessed_data)
    final_data = feedforward_response.json()
    
    
    response = send_file(final_data.get("audio_path"), as_attachment=True, attachment_filename='generated_audio.wav')

    @response.call_on_close
    def remove_file():
        try:
            os.remove(final_data.get("audio_path"))
        except Exception as e:
            app.logger.error(f"Error removing file {final_data.get("audio_path")}: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
