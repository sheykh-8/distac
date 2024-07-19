import torch
from model.Tacotron2_partitions.tacotron_post_net import TacotronPostNet


postnet = TacotronPostNet()
postnet.load_state_dict(
    torch.load("model/Tacotron2_partitions/model_store/tacotron_postnet_weights.pt")
)
postnet.eval()


def get_inference(mel_output, alignments):
    # we need to convert input lengths and encoded inputs to tensors before continuing the process:
    mel_output = torch.tensor(mel_output)
    alignments = torch.tensor(alignments)
    mel_outputs_postnet, alignments = postnet(
        mel_output, alignments
    )

    return mel_outputs_postnet, alignments
