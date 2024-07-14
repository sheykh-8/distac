import torch
from model.Tacotron2_partitions.tacotron_decoder import TacotronDecoder


decoder = TacotronDecoder()
decoder.load_state_dict(
    torch.load("model/Tacotron2_partitions/model_store/tacotron_decoder_weights.pt")
)
decoder.eval()


def get_inference(encoded_inputs, input_lengths):
    # we need to convert input lengths and encoded inputs to tensors before continuing the process:
    encoded_inputs = torch.tensor(encoded_inputs)
    input_lengths = torch.tensor(encoded_inputs)
    mel_outputs, gate_outputs, alignments, mel_lengths = decoder(
        encoded_inputs, input_lengths
    )

    return mel_outputs, gate_outputs, alignments, mel_lengths
