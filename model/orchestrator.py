from flask import Flask, request, send_file, Response
import requests
import os

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    
    # Step 1: Call Encoder Service
    encoder_response = requests.post('http://encoder:80/predict', json=data)
    encoded_data = encoder_response.json()
    
    print(f"got response from encoder!")
    
    # Step 2: Call Decoder Service
    decoder_response = requests.post('http://decoder:80/predict', json=encoded_data)
    decoded_data = decoder_response.json()
    
    # Step 3: Call Postnet Service
    postnet_response = requests.post('http://postnet:80/predict', json=decoded_data)
    postprocessed_data = postnet_response.json()
    
    # Step 4: Call hifigan Service
    vocoder_response = requests.post('http://hifigan:80/generate', json=postprocessed_data)

    return Response(vocoder_response.iter_content(chunk_size=1024), 
                    content_type=vocoder_response.headers['Content-Type'],
                    headers={"Content-Disposition": vocoder_response.headers['Content-Disposition']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
