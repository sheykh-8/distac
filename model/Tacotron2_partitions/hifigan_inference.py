import torch
from model.Tacotron2_partitions.hifigan import HifiganGenerator
import torchaudio
import uuid

vocoder = HifiganGenerator()
vocoder.load_state_dict(
    torch.load("model/Tacotron2_partitions/model_store/hifigan_weights.pt")
)
vocoder.eval()


def get_inference(spectrogram):
    # we need to convert input lengths and encoded inputs to tensors before continuing the process:
    spectrogram = torch.tensor(spectrogram)
    waveform = vocoder(
        spectrogram
    )

    audio_path = f"/mnt/shared/inferenced_audio{uuid.uuid4().hex}.wav"
    torchaudio.save(audio_path, waveform.squeeze(1), 22050)
    return audio_path