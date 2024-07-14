
import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Tacotron2(nn.Module):
    """The Tactron2 text-to-speech model, based on the NVIDIA implementation.

    This class is the main entry point for the model, which is responsible
    for instantiating all submodules, which, in turn, manage the individual
    neural network layers

    Simplified STRUCTURE: input->word embedding ->encoder ->attention \
    ->decoder(+prenet) -> postnet ->output

    prenet(input is decoder previous time step) output is input to decoder
    concatenated with the attention output

    Arguments
    ---------
    mask_padding: bool
        whether or not to mask pad-outputs of tacotron
    n_mel_channels: int
        number of mel channels for constructing spectrogram
    n_symbols:  int=128
        number of accepted char symbols defined in textToSequence
    symbols_embedding_dim: int
        number of embedding dimension for symbols fed to nn.Embedding
    encoder_kernel_size: int
        size of kernel processing the embeddings
    encoder_n_convolutions: int
        number of convolution layers in encoder
    encoder_embedding_dim: int
        number of kernels in encoder, this is also the dimension
        of the bidirectional LSTM in the encoder
    attention_rnn_dim: int
        input dimension
    attention_dim: int
        number of hidden representation in attention
    attention_location_n_filters: int
        number of 1-D convolution filters in attention
    attention_location_kernel_size: int
        length of the 1-D convolution filters
    n_frames_per_step: int=1
        only 1 generated mel-frame per step is supported for the decoder as of now.
    decoder_rnn_dim: int
        number of 2 unidirectional stacked LSTM units
    prenet_dim: int
        dimension of linear prenet layers
    max_decoder_steps: int
        maximum number of steps/frames the decoder generates before stopping
    gate_threshold: int
        cut off level any output probability above that is considered
        complete and stops generation so we have variable length outputs
    p_attention_dropout: float
        attention drop out probability
    p_decoder_dropout: float
        decoder drop  out probability
    postnet_embedding_dim: int
        number os postnet dfilters
    postnet_kernel_size: int
        1d size of posnet kernel
    postnet_n_convolutions: int
        number of convolution layers in postnet
    decoder_no_early_stopping: bool
        determines early stopping of decoder
        along with gate_threshold . The logical inverse of this is fed to the decoder

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(213312)
    >>> from speechbrain.lobes.models.Tacotron2 import Tacotron2
    >>> model = Tacotron2(
    ...    mask_padding=True,
    ...    n_mel_channels=80,
    ...    n_symbols=148,
    ...    symbols_embedding_dim=512,
    ...    encoder_kernel_size=5,
    ...    encoder_n_convolutions=3,
    ...    encoder_embedding_dim=512,
    ...    attention_rnn_dim=1024,
    ...    attention_dim=128,
    ...    attention_location_n_filters=32,
    ...    attention_location_kernel_size=31,
    ...    n_frames_per_step=1,
    ...    decoder_rnn_dim=1024,
    ...    prenet_dim=256,
    ...    max_decoder_steps=32,
    ...    gate_threshold=0.5,
    ...    p_attention_dropout=0.1,
    ...    p_decoder_dropout=0.1,
    ...    postnet_embedding_dim=512,
    ...    postnet_kernel_size=5,
    ...    postnet_n_convolutions=5,
    ...    decoder_no_early_stopping=False
    ... )
    >>> _ = model.eval()
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> outputs, output_lengths, alignments = model.infer(inputs, input_lengths)
    >>> outputs.shape, output_lengths.shape, alignments.shape
    (torch.Size([2, 80, 1]), torch.Size([2]), torch.Size([2, 1, 5]))
    """

    def __init__(
        self,
        mask_padding=True,
        # mel generation parameter in data io
        n_mel_channels=80,
        # symbols
        n_symbols=148,
        symbols_embedding_dim=512,
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        # Decoder parameters
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        decoder_no_early_stopping=False,
    ):
        super().__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        
        # TODO: Embedding is moved to the encoder class for ease of use
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        
        # TODO: Send a request to get the encoder's result.
        self.encoder = Encoder(
            encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size
        )
        
        # TODO: Send a reqeust to get the decoder's result.
        self.decoder = Decoder(
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_rnn_dim,
            decoder_rnn_dim,
            prenet_dim,
            max_decoder_steps,
            gate_threshold,
            p_attention_dropout,
            p_decoder_dropout,
            not decoder_no_early_stopping,
        )
        
        # TODO: Send a request to get the postnets result.
        self.postnet = Postnet(
            n_mel_channels,
            postnet_embedding_dim,
            postnet_kernel_size,
            postnet_n_convolutions,
        )

    def parse_output(self, outputs, output_lengths, alignments_dim=None):
        """
        Masks the padded part of output

        Arguments
        ---------
        outputs: list
            a list of tensors - raw outputs
        output_lengths: torch.Tensor
            a tensor representing the lengths of all outputs
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training


        Returns
        -------
        mel_outputs: torch.Tensor
        mel_outputs_postnet: torch.Tensor
        gate_outputs: torch.Tensor
        alignments: torch.Tensor
            the original outputs - with the mask applied
        """
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(
                output_lengths, max_len=mel_outputs.size(-1)
            )
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_outputs.clone().masked_fill_(mask, 0.0)
            mel_outputs_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        if alignments_dim is not None:
            alignments = F.pad(
                alignments, (0, alignments_dim - alignments.size(-1))
            )

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def forward(self, inputs, alignments_dim=None):
        """Decoder forward pass for training

        Arguments
        ---------
        inputs: tuple
            batch object
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training

        Returns
        -------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        mel_outputs_postnet: torch.Tensor
            mel outputs from postnet
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        output_lengths: torch.Tensor
            length of the output without padding
        """

        inputs, input_lengths, targets, max_len, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths,
            alignments_dim,
        )

    def infer(self, inputs, input_lengths):
        """Produces outputs


        Arguments
        ---------
        inputs: torch.tensor
            text or phonemes converted

        input_lengths: torch.tensor
            the lengths of input parameters

        Returns
        -------
        mel_outputs_postnet: torch.Tensor
            final mel output of tacotron 2
        mel_lengths: torch.Tensor
            length of mels
        alignments: torch.Tensor
            sequence of attention weights
        """

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments




def get_mask_from_lengths(lengths, max_len=None):
    """Creates a binary mask from sequence lengths

    Arguments
    ---------
    lengths: torch.Tensor
        A tensor of sequence lengths
    max_len: int (Optional)
        Maximum sequence length, defaults to None.

    Returns
    -------
    mask: torch.Tensor
        the mask where padded elements are set to True.
        Then one can use tensor.masked_fill_(mask, 0) for the masking.

    Example
    -------
    >>> lengths = torch.tensor([3, 2, 4])
    >>> get_mask_from_lengths(lengths)
    tensor([[False, False, False,  True],
            [False, False,  True,  True],
            [False, False, False, False]])
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    seq_range = torch.arange(
        max_len, device=lengths.device, dtype=lengths.dtype
    )
    return ~(seq_range.unsqueeze(0) < lengths.unsqueeze(1))
