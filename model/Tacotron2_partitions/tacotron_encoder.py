from speechbrain.lobes.models.Tacotron2_partitions.utils import text_to_sequence
from speechbrain.lobes.models.Tacotron2_partitions.utils.padding import PaddedBatch


import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt


class ConvNorm(torch.nn.Module):
    """A 1D convolution layer with Xavier initialization

    Arguments
    ---------
    in_channels: int
        the number of input channels
    out_channels: int
        the number of output channels
    kernel_size: int
        the kernel size
    stride: int
        the convolutional stride
    padding: int
        the amount of padding to include. If not provided, it will be calculated
        as dilation * (kernel_size - 1) / 2
    dilation: int
        the dilation of the convolution
    bias: bool
        whether or not to use a bias
    w_init_gain: linear
        the weight initialization gain type (see torch.nn.init.calculate_gain)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import ConvNorm
    >>> layer = ConvNorm(in_channels=10, out_channels=5, kernel_size=3)
    >>> x = torch.randn(3, 10, 5)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 5, 5])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        """Computes the forward pass

        Arguments
        ---------
        signal: torch.Tensor
            the input to the convolutional layer

        Returns
        -------
        output: torch.Tensor
            the output
        """
        return self.conv(signal)


class Encoder(nn.Module):
    """The Tacotron2 encoder module, consisting of a sequence of  1-d convolution banks (3 by default)
    and a bidirectional LSTM

    Arguments
    ---------
    encoder_n_convolutions: int
        the number of encoder convolutions
    encoder_embedding_dim: int
        the dimension of the encoder embedding
    encoder_kernel_size: int
        the kernel size of the 1-D convolutional layers within
        the encoder

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Encoder
    >>> layer = Encoder()
    >>> x = torch.randn(2, 512, 128)
    >>> input_lengths = torch.tensor([128, 83])
    >>> outputs = layer(x, input_lengths)
    >>> outputs.shape
    torch.Size([2, 128, 512])

    """

    def __init__(
        self,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        encoder_kernel_size=5,
    ):
        super().__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        """Computes the encoder forward pass

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        """Performs a forward step in the inference context

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        device = x.device
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x.to(device))), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class TacotronEncoder(nn.Module):
    def __init__(
        self,
        n_symbols=148,
        symbols_embedding_dim=512,
    ):
        super().__init__()

        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder()

        self.encoder.eval()
        self.embedding.eval()

    def forward(self, inputs, input_lengths):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        return self.encoder(embedded_inputs, input_lengths)
    
    def text_to_seq(self, text):
        seq = text_to_sequence(text, ["english_cleaners"])
        return seq, len(seq)
    
    
    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.text_to_seq(item)[0], device="cpu"
                    )
                }
                for item in texts
            ]
            inputs = PaddedBatch(inputs)

            lens = [self.text_to_seq(item)[1] for item in texts]
            assert lens == sorted(
                lens, reverse=True
            ), "input lengths must be sorted in decreasing order"
            input_lengths = torch.tensor(lens, device="cpu")

            # mel_outputs_postnet, mel_lengths, alignments = self.infer(
            #     inputs.text_sequences.data, input_lengths
            # )
        return self.forward(inputs.text_sequences.data, input_lengths), input_lengths
