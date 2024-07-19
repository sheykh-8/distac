import torch
from torch import nn
from torch.nn import functional as F


class LinearNorm(torch.nn.Module):
    """A linear layer with Xavier initialization

    Arguments
    ---------
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    bias: bool
        whether or not to use a bias
    w_init_gain: linear
        the weight initialization gain type (see torch.nn.init.calculate_gain)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import LinearNorm
    >>> layer = LinearNorm(in_dim=5, out_dim=3)
    >>> x = torch.randn(3, 5)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 3])
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x: torch.Tensor
            a (batch, features) input tensor


        Returns
        -------
        output: torch.Tensor
            the linear layer output

        """
        return self.linear_layer(x)


class Prenet(nn.Module):
    """The Tacotron pre-net module consisting of a specified number of
    normalized (Xavier-initialized) linear layers

    Arguments
    ---------
    in_dim: int
        the input dimensions
    sizes: int
        the dimension of the hidden layers/output
    dropout: float
        the dropout probability

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Prenet
    >>> layer = Prenet()
    >>> x = torch.randn(862, 2, 80)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([862, 2, 256])
    """

    def __init__(self, in_dim=80, sizes=[256, 256], dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout = dropout

    def forward(self, x):
        """Computes the forward pass for the prenet

        Arguments
        ---------
        x: torch.Tensor
            the prenet inputs

        Returns
        -------
        output: torch.Tensor
            the output
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=True)
        return x


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


class LocationLayer(nn.Module):
    """A location-based attention layer consisting of a Xavier-initialized
    convolutional layer followed by a dense layer

    Arguments
    ---------
    attention_n_filters: int
        the number of filters used in attention

    attention_kernel_size: int
        the kernel size of the attention layer

    attention_dim: int
        the dimension of linear attention layers


    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import LocationLayer
    >>> layer = LocationLayer()
    >>> attention_weights_cat = torch.randn(3, 2, 64)
    >>> processed_attention = layer(attention_weights_cat)
    >>> processed_attention.shape
    torch.Size([3, 64, 128])

    """

    def __init__(
        self,
        attention_n_filters=32,
        attention_kernel_size=31,
        attention_dim=128,
    ):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        """Performs the forward pass for the attention layer

        Arguments
        ---------
        attention_weights_cat: torch.Tensor
            the concatenating attention weights

        Returns
        -------
        processed_attention: torch.Tensor
            the attention layer output

        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


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


class Attention(nn.Module):
    """The Tacotron attention layer. Location-based attention is used.

    Arguments
    ---------
    attention_rnn_dim: int
        the dimension of the RNN to which the attention layer
        is applied
    embedding_dim: int
        the embedding dimension
    attention_dim: int
        the dimension of the memory cell
    attention_location_n_filters: int
        the number of location filters
    attention_location_kernel_size: int
        the kernel size of the location layer

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import (
    ... Attention)
    >>> from speechbrain.lobes.models.transformer.Transformer import (
    ... get_mask_from_lengths)
    >>> layer = Attention()
    >>> attention_hidden_state = torch.randn(2, 1024)
    >>> memory = torch.randn(2, 173, 512)
    >>> processed_memory = torch.randn(2, 173, 128)
    >>> attention_weights_cat = torch.randn(2, 2, 173)
    >>> memory_lengths = torch.tensor([173, 91])
    >>> mask = get_mask_from_lengths(memory_lengths)
    >>> attention_context, attention_weights = layer(
    ...    attention_hidden_state,
    ...    memory,
    ...    processed_memory,
    ...    attention_weights_cat,
    ...    mask
    ... )
    >>> attention_context.shape, attention_weights.shape
    (torch.Size([2, 512]), torch.Size([2, 173]))
    """

    def __init__(
        self,
        attention_rnn_dim=1024,
        embedding_dim=512,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
    ):
        super().__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self, query, processed_memory, attention_weights_cat
    ):
        """Computes the alignment energies

        Arguments
        ---------
        query: torch.Tensor
            decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: torch.Tensor
            processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: torch.Tensor
            cumulative and prev. att weights (B, 2, max_time)

        Returns
        -------
        alignment : torch.Tensor
            (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(
                processed_query + processed_attention_weights + processed_memory
            )
        )

        energies = energies.squeeze(2)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """Computes the forward pass

        Arguments
        ---------
        attention_hidden_state: torch.Tensor
            attention rnn last output
        memory: torch.Tensor
            encoder outputs
        processed_memory: torch.Tensor
            processed encoder outputs
        attention_weights_cat: torch.Tensor
            previous and cumulative attention weights
        mask: torch.Tensor
            binary mask for padded data

        Returns
        -------
        result: tuple
            a (attention_context, attention_weights) tuple
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Decoder(nn.Module):
    """The Tacotron decoder

    Arguments
    ---------
    n_mel_channels: int
        the number of channels in the MEL spectrogram
    n_frames_per_step: int
        the number of frames in the spectrogram for each
        time step of the decoder
    encoder_embedding_dim: int
        the dimension of the encoder embedding
    attention_dim: int
        Size of attention vector
    attention_location_n_filters: int
        the number of filters in location-based attention
    attention_location_kernel_size: int
        the kernel size of location-based attention
    attention_rnn_dim: int
        RNN dimension for the attention layer
    decoder_rnn_dim: int
        the encoder RNN dimension
    prenet_dim: int
        the dimension of the prenet (inner and output layers)
    max_decoder_steps: int
        the maximum number of decoder steps for the longest utterance
        expected for the model
    gate_threshold: float
        the fixed threshold to which the outputs of the decoders will be compared
    p_attention_dropout: float
        dropout probability for attention layers
    p_decoder_dropout: float
        dropout probability for decoder layers
    early_stopping: bool
        Whether to stop training early.

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Decoder
    >>> layer = Decoder()
    >>> memory = torch.randn(2, 173, 512)
    >>> decoder_inputs = torch.randn(2, 80, 173)
    >>> memory_lengths = torch.tensor([173, 91])
    >>> mel_outputs, gate_outputs, alignments = layer(
    ...     memory, decoder_inputs, memory_lengths)
    >>> mel_outputs.shape, gate_outputs.shape, alignments.shape
    (torch.Size([2, 80, 173]), torch.Size([2, 173]), torch.Size([2, 173, 173]))
    """

    def __init__(
        self,
        n_mel_channels=80,
        n_frames_per_step=1,
        encoder_embedding_dim=512,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        attention_rnn_dim=1024,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        early_stopping=True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step, [prenet_dim, prenet_dim]
        )

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim, attention_rnn_dim
        )

        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory):
        """Gets all zeros frames to use as first decoder input

        Arguments
        ---------
        memory: torch.Tensor
            decoder outputs

        Returns
        -------
        decoder_input: torch.Tensor
            all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B,
            self.n_mel_channels * self.n_frames_per_step,
            dtype=dtype,
            device=device,
        )
        return decoder_input

    def initialize_decoder_states(self, memory):
        """Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        Arguments
        ---------
        memory: torch.Tensor
            Encoder outputs

        Returns
        -------
        attention_hidden: torch.Tensor
        attention_cell: torch.Tensor
        decoder_hidden: torch.Tensor
        decoder_cell: torch.Tensor
        attention_weights: torch.Tensor
        attention_weights_cum: torch.Tensor
        attention_context: torch.Tensor
        processed_memory: torch.Tensor
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )

        attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device
        )
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device
        )

        processed_memory = self.attention_layer.memory_layer(memory)

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        )

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepares decoder inputs, i.e. mel outputs

        Arguments
        ---------
        decoder_inputs: torch.Tensor
            inputs used for teacher-forced training, i.e. mel-specs

        Returns
        -------
        decoder_inputs: torch.Tensor
            processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepares decoder outputs for output

        Arguments
        ---------
        mel_outputs: torch.Tensor
            MEL-scale spectrogram outputs
        gate_outputs: torch.Tensor
            gate output energies
        alignments: torch.Tensor
            the alignment tensor

        Returns
        -------
        mel_outputs: torch.Tensor
            MEL-scale spectrogram outputs
        gate_outputs: torch.Tensor
            gate output energies
        alignments: torch.Tensor
            the alignment tensor
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        if gate_outputs.dim() == 1:
            gate_outputs = gate_outputs.unsqueeze(0)
        else:
            gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(
        self,
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    ):
        """Decoder step using stored states, attention and memory
        Arguments
        ---------
        decoder_input: torch.Tensor
            previous mel output
        attention_hidden: torch.Tensor
            the hidden state of the attention module
        attention_cell: torch.Tensor
            the attention cell state
        decoder_hidden: torch.Tensor
            the decoder hidden state
        decoder_cell: torch.Tensor
            the decoder cell state
        attention_weights: torch.Tensor
            the attention weights
        attention_weights_cum: torch.Tensor
            cumulative attention weights
        attention_context: torch.Tensor
            the attention context tensor
        memory: torch.Tensor
            the memory tensor
        processed_memory: torch.Tensor
            the processed memory tensor
        mask: torch.Tensor



        Returns
        -------
        mel_output: torch.Tensor
            the MEL-scale outputs
        gate_output: torch.Tensor
            gate output energies
        attention_weights: torch.Tensor
            attention weights
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell)
        )
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (
                attention_weights.unsqueeze(1),
                attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        attention_context, attention_weights = self.attention_layer(
            attention_hidden,
            memory,
            processed_memory,
            attention_weights_cat,
            mask,
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )

    @torch.jit.ignore
    def forward(self, memory, decoder_inputs, memory_lengths):
        """Decoder forward pass for training

        Arguments
        ---------
        memory: torch.Tensor
            Encoder outputs
        decoder_inputs: torch.Tensor
            Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: torch.Tensor
            Encoder output lengths for attention masking.

        Returns
        -------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments),
        )

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """Decoder inference

        Arguments
        ---------
        memory: torch.Tensor
            Encoder outputs
        memory_lengths: torch.Tensor
            The corresponding relative lengths of the inputs.

        Returns
        -------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        mel_lengths: torch.Tensor
            the length of MEL spectrograms
        """
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0
                )
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = (
                torch.le(torch.sigmoid(gate_output), self.gate_threshold)
                .to(torch.int32)
                .squeeze(1)
            )

            not_finished = not_finished * dec
            mel_lengths += not_finished
            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments, mel_lengths


class TacotronDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = Decoder()

        self.decoder.eval()

    def forward(self, memory, memory_lengths):
        return self.decoder.infer(memory, memory_lengths)
