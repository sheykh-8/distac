import torch
from model.Tacotron2_partitions.hifigan import HifiganGenerator
import torchaudio
import uuid
import io
from google.cloud import storage

storage_client = storage.Client()
bucket_name = 'sheykh-c68e3.appspot.com'
bucket = storage_client.bucket(bucket_name)

vocoder = HifiganGenerator()
vocoder.load_state_dict(
    torch.load("model/Tacotron2_partitions/model_store/hifigan_weights.pt", map_location=torch.device('cpu'))
)
vocoder.eval()

def get_inference(spectrogram):
    
    spectrogram = torch.tensor(spectrogram)
    
    waveform = vocoder.decode_batch(spectrogram)
    
    filename = f"inferenced_audio_{uuid.uuid4().hex}.wav"
    
    audio_buffer = io.BytesIO()
    
    torchaudio.save(audio_buffer, waveform.squeeze(1), 22050, format="wav")
    
    audio_buffer.seek(0)
    
    blob = bucket.blob(filename)
    blob.upload_from_file(audio_buffer, content_type='audio/wav')
    
    blob.make_public()
    url = blob.public_url
    
    return url