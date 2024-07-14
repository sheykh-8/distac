import torch
from torch import nn
from torch.nn import functional as F


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


class Postnet(nn.Module):
    """The Tacotron postnet consists of a number of 1-d convolutional layers
    with Xavier initialization and a tanh activation, with batch normalization.
    Depending on configuration, the postnet may either refine the MEL spectrogram
    or upsample it to a linear spectrogram

    Arguments
    ---------
    n_mel_channels: int
        the number of MEL spectrogram channels
    postnet_embedding_dim: int
        the postnet embedding dimension
    postnet_kernel_size: int
        the kernel size of the convolutions within the decoders
    postnet_n_convolutions: int
        the number of convolutions in the postnet

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Postnet
    >>> layer = Postnet()
    >>> x = torch.randn(2, 80, 861)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([2, 80, 861])
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        """Computes the forward pass of the postnet

        Arguments
        ---------
        x: torch.Tensor
            the postnet input (usually a MEL spectrogram)

        Returns
        -------
        output: torch.Tensor
            the postnet output (a refined MEL spectrogram or a
            linear spectrogram depending on how the model is
            configured)
        """
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x


class TacotronPostNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.postnet = Postnet()
        
        self.postnet.eval()

    def forward(self, mel_outputs, alignments):
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)
        return mel_outputs_postnet, alignments
