import collections
import re
import torch
from torch.utils.data._utils.collate import default_convert
from torch.utils.data._utils.pin_memory import (
    pin_memory as recursive_pin_memory,
)


def pad_right_to(tensor, target_shape, mode="constant", value=0):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Arguments
    ---------
    tensor : torch.Tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals



def batch_pad_right(tensors: list, mode="constant", value=0):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Arguments
    ---------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """
    if not len(tensors):
        raise IndexError("Tensors list must not be empty")

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        all(
            [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)



PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def mod_default_collate(batch):
    """Makes a tensor from list of batch values.

    Note that this doesn't need to zip(*) values together
    as PaddedBatch connects them already (by key).

    Here the idea is not to error out.

    This is modified from:
    https://github.com/pytorch/pytorch/blob/c0deb231db76dbea8a9d326401417f7d1ce96ed5/torch/utils/data/_utils/collate.py#L42
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:  # Unequal size:
            return batch
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        try:
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    return batch
                return mod_default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        except RuntimeError:  # Unequal size
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    else:
        return batch

def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers.

    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data


class PaddedBatch:
    """Collate_fn when examples are dicts and have variable-length sequences.

    Different elements in the examples get matched by key.
    All numpy tensors get converted to Torch (PyTorch default_convert)
    Then, by default, all torch.Tensor valued elements get padded and support
    collective pin_memory() and to() calls.
    Regular Python data types are just collected in a list.

    Arguments
    ---------
    examples : list
        List of example dicts, as produced by Dataloader.
    padded_keys : list, None
        (Optional) List of keys to pad on. If None, pad all torch.Tensors
    device_prep_keys : list, None
        (Optional) Only these keys participate in collective memory pinning and moving with
        to().
        If None, defaults to all items with torch.Tensor values.
    padding_func : callable, optional
        Called with a list of tensors to be padded together. Needs to return
        two tensors: the padded data, and another tensor for the data lengths.
    padding_kwargs : dict
        (Optional) Extra kwargs to pass to padding_func. E.G. mode, value
    apply_default_convert : bool
        Whether to apply PyTorch default_convert (numpy to torch recursively,
        etc.) on all data. Default:True, usually does the right thing.
    nonpadded_stack : bool
        Whether to apply PyTorch-default_collate-like stacking on values that
        didn't get padded. This stacks if it can, but doesn't error out if it
        cannot. Default:True, usually does the right thing.

    Example
    -------
    >>> batch = PaddedBatch([
    ...     {"id": "ex1", "foo": torch.Tensor([1.])},
    ...     {"id": "ex2", "foo": torch.Tensor([2., 1.])}])
    >>> # Attribute or key-based access:
    >>> batch.id
    ['ex1', 'ex2']
    >>> batch["id"]
    ['ex1', 'ex2']
    >>> # torch.Tensors get padded
    >>> type(batch.foo)
    <class 'speechbrain.dataio.batch.PaddedData'>
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]])
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000])
    >>> # Batch supports collective operations:
    >>> _ = batch.to(dtype=torch.half)
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]], dtype=torch.float16)
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000], dtype=torch.float16)
    >>> # Numpy tensors get converted to torch and padded as well:
    >>> import numpy as np
    >>> batch = PaddedBatch([
    ...     {"wav": np.asarray([1,2,3,4])},
    ...     {"wav": np.asarray([1,2,3])}])
    >>> batch.wav  # +ELLIPSIS
    PaddedData(data=tensor([[1, 2,...
    >>> # Basic stacking collation deals with non padded data:
    >>> batch = PaddedBatch([
    ...     {"spk_id": torch.tensor([1]), "wav": torch.tensor([.1,.0,.3])},
    ...     {"spk_id": torch.tensor([2]), "wav": torch.tensor([.2,.3,-.1])}],
    ...     padded_keys=["wav"])
    >>> batch.spk_id
    tensor([[1],
            [2]])
    >>> # And some data is left alone:
    >>> batch = PaddedBatch([
    ...     {"text": ["Hello"]},
    ...     {"text": ["How", "are", "you?"]}])
    >>> batch.text
    [['Hello'], ['How', 'are', 'you?']]

    """

    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], torch.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.

        Returns
        -------
        Iterator over the batch.

        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Gets the position."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        """Returns the bach size"""
        return self.__length




