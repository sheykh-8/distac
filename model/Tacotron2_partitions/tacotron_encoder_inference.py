import torch
from model.Tacotron2_partitions.tacotron_encoder import TacotronEncoder


encoder = TacotronEncoder()
encoder.load_state_dict(torch.load("model/Tacotron2_partitions/model_store/tacotron_encoder_weights.pt"))

def get_inference(input_text):
    encoded_input, input_lengths = encoder.encode_batch([input_text])
    print(encoded_input.shape)
    
    return encoded_input, input_lengths
