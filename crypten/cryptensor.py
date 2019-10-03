#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class CrypTensor(object):
    """
        Encrypted tensor type that is private and cannot be shown to the outside world.
    """

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new CrypTensor of same type.
        """
        raise NotImplementedError("new is not implemented")

    def abs(self):
        raise NotImplementedError("abs is not implemented")

    def __abs__(self):
        return self.abs()

    def pow(self):
        raise NotImplementedError("pow is not implemented")

    def __pow__(self, tensor):
        return self.pow(tensor)

    def __rpow__(self, scalar):
        raise NotImplementedError("__rpow__ is not implemented")

    def __init__(self):
        raise NotImplementedError("Cannot instantiate an CrypTensor")

    def get_plain_text(self):
        """Decrypts the encrypted tensor."""
        raise NotImplementedError("get_plain_text is not implemented")

    def shallow_copy(self):
        """Creates a shallow_copy of a tensor"""
        raise NotImplementedError("shallow_copy is not implemented")

    def add_(self, tensor):
        """Adds tensor to self (in-place) see :meth:`add`."""
        raise NotImplementedError("add_ is not implemented")

    def add(self, tensor):
        r"""Adds tensor to this :attr:`self`.

        Args:
            tensor: can be a torch tensor or a CrypTensor.

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        For a scalar `tensor`,

        .. math::
            \text{{out_i}} = \text{{input_i}} + \text{{tensor}}

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("add is not implemented")

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def sub_(self, tensor):
        """Subtracts tensor from `self` (in-place), see :meth:`sub`"""
        raise NotImplementedError("sub_ is not implemented")

    def sub(self, tensor):
        """Subtracts a scalar or tensor from :attr:`self` tensor.
        The shape of :attr:`tensor` must be
        `broadcastable`_ with the shape of :attr:`self`.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("sub is not implemented")

    def __sub__(self, tensor):
        """Subtracts tensor from this tensor."""
        return self.sub(tensor)

    def __rsub__(self, tensor):
        """Subtracts self from tensor."""
        return -self + tensor

    def __isub__(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        return self.sub_(tensor)

    def mul_(self, tensor):
        """Element-wise multiply with a tensor in-place, see :meth:`mul`."""
        raise NotImplementedError("mul_ is not implemented")

    def mul(self, tensor):
        r"""Element-wise multiply with a tensor.

        .. math::
            \text{out}_i = \text{tensor}_i \times \text{self}_i

        Args:
            tensor (Tensor or float): the tensor or value to multiply.

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("mul is not implemented")

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def div_(self, tensor):
        """Element-wise in-place divide by a tensor (see :meth:`div`)."""
        raise NotImplementedError("div_ is not implemented")

    def div(self, tensor):
        r"""
        Divides each element of the input :attr:`self` with the :attr:`tensor`
        and returns a new resulting tensor.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{tensor}_i}

        The shapes of :attr:`self` and :attr:`tensor` must be
        `broadcastable`_.

        Args:
            tensor (Tensor or float): the tensor or value in the denominator.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("div is not implemented")

    def __div__(self, tensor):
        """Element-wise divide by a tensor."""
        return self.div(tensor)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def neg(self):
        r"""
        Returns a new tensor with the negative of the elements of :attr:`self`.

        .. math::
            \text{out} = -1 \times \text{input}
        """
        raise NotImplementedError("neg is not implemented")

    def neg_(self):
        """Negative value of a tensor (in-place), see :meth:`neg`"""
        raise NotImplementedError("neg_ is not implemented")

    def __neg__(self):
        return self.neg()

    def matmul(self, tensor):
        r"""Performs matrix multiplication of `self` with `tensor`

        The behavior depends on the dimensionality of the tensors as follows:

        - If both tensors are 1-dimensional, the dot product (scalar) is returned.
        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is
          2-dimensional, a 1 is prepended to its dimension for the purpose of
          the matrix multiply. After the matrix multiply, the
          prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is 1-dimensional,
          the matrix-vector product is returned.
        - If both arguments are at least 1-dimensional and at least one argument
          is N-dimensional (where N > 2), then a batched matrix multiply is returned.
          If the first argument is 1-dimensional, a 1 is prepended to its dimension
          for the purpose of the batched matrix multiply and removed after.
          If the second argument is 1-dimensional, a 1 is appended to its dimension
          for the purpose of the batched matrix multiple and removed after.
          The non-matrix (i.e. batch) dimensions are broadcasted (and thus
          must be `broadcastable`_).  For example, if :attr:`self` is a
          :math:`(j \times 1 \times n \times m)` tensor and :attr:`tensor` is a :math:`(k \times m \times p)`
          tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.

        Arguments:
            tensor (Tensor): the tensor to be multiplied

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("matmul is not implemented")

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def eq(self, tensor):
        """Element-wise equality

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("eq is not implemented")

    def __eq__(self, tensor):
        """Element-wise equality"""
        return self.eq(tensor)

    def ne(self, tensor):
        """Element-wise inequality

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with the `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted boolean tensor containing a True at each location where comparison is true.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("ne is not implemented")

    def __ne__(self, tensor):
        """Element-wise inequality"""
        return self.ne(tensor)

    def ge(self, tensor):
        """Element-wise greater than or equal to

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted``BoolTensor`` containing a True at each location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("ge is not implemented")

    def __ge__(self, tensor):
        """Element-wise greater than or equal to"""
        return self.ge(tensor)

    def gt(self, tensor):
        """Element-wise greater than

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted``BoolTensor`` containing a True at each location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("gt is not implemented")

    def __gt__(self, tensor):
        """Element-wise greater than"""
        return self.gt(tensor)

    def le(self, tensor):
        """Element-wise less than or equal to

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted``BoolTensor`` containing a True at each location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("le is not implemented")

    def __le__(self, tensor):
        """Element-wise less than or equal to"""
        return self.le(tensor)

    def lt(self, tensor):
        """Element-wise less than

        The `tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with `self`.

        Args:
            tensor (Tensor or float): the tensor or value to compare.

        Returns:
            an encrypted``BoolTensor`` containing a True at each location where comparison is true

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("lt is not implemented")

    def __lt__(self, tensor):
        """Element-wise less than"""
        return self.lt(tensor)

    def dot(self, tensor, weights=None):
        """Perform (weighted) inner product with plain or cipher text."""
        raise NotImplementedError("dot is not implemented")

    def index_add(self, dim, index, tensor):
        """Perform out-of-place index_add: Accumulate the elements of tensor into
        self tensor by adding to the indices in the order given in index. """
        raise NotImplementedError("index_add is not implemented")

    def index_add_(self, dim, index, tensor):
        """Perform in-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index. """
        raise NotImplementedError("index_add_ is not implemented")

    # Regular functions:
    def clone(self):
        """
        Returns a copy of the :attr:`self` tensor.
        The copy has the same size and data type as :attr:`self`.

        .. note::
            This function is recorded in the computation graph. Gradients
            propagating to the cloned tensor will propagate to the original tensor.
        """
        raise NotImplementedError("clone is not implemented")

    def __getitem__(self, index):
        """
        Returns an encrypted tensor containing elements of self at `index`
        """
        raise NotImplementedError("__getitem__ is not implemented")

    def __setitem__(self, index, value):
        """
        Sets elements of an encrypted tensor `self` at index `index` to `value`.
        """
        raise NotImplementedError("__setitem__ is not implemented")

    def index_select(self, dim, index):
        """
        Returns a new tensor which indexes the `self` tensor along dimension
        `dim` using the entries in `index` which is a LongTensor.

        The returned tensor has the same number of dimensions as the original
        tensor (`self`). The `dim`th dimension has the same size as the length
        of `index`; other dimensions have the same size as in the original tensor.
        """
        raise NotImplementedError("index_select is not implemented")

    def view(self, *shape):
        r"""
        Returns a new encrypted tensor with the same data as the `self` tensor
        but of a different shape.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = 0, \dots, k-1`,

        .. math::
            \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Args:
            shape (torch.Size or int...): the desired
        """
        raise NotImplementedError("view is not implemented")

    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens a contiguous range of dims in a tensor.

        Args:
            start_dim (int): the first dim to flatten. Default is 0.
            end_dim (int): the last dim to flatten. Default is -1.
        """
        raise NotImplementedError("flatten is not implemented")

    def t(self):
        """
        Expects `self` to be <= 2-D tensor and transposes dimensions 0 and 1.

        0-D and 1-D tensors are returned as is and for 2-D tensors this can be
        seen as a short-hand function for `self.transpose(0, 1)`.
        """
        raise NotImplementedError("t is not implemented")

    def transpose(self, dim0, dim1):
        """
        Returns a tensor that is a transposed version of `self`.
        The given dimensions `dim0` and `dim1` are swapped.

        The resulting out tensor shares it’s underlying storage with the `self`
        tensor, so changing the content of one would change the content of the
        other.

        Args:
            dim0 (int): the first dimension to be transposed
            dim1 (int): the second dimension to be transposed
        """
        raise NotImplementedError("t is not implemented")

    def unsqueeze(self, dim):
        """
        Returns a new tensor with a dimension of size one inserted at the
        specified position.

        The returned tensor shares the same underlying data with this tensor.

        A `dim` value within the range `[-self.dim() - 1, self.dim() + 1)`
        can be used. Negative `dim` will correspond to `unsqueeze()` applied at
        `dim = dim + self.dim() + 1`

        Args:
            dim (int): the index at which to insert the singleton dimension
        """
        raise NotImplementedError("unsqueeze is not implemented")

    def squeeze(self, dim=None):
        """
        Returns a tensor with all the dimensions of `self` of size 1 removed.

        For example, if `self` is of shape:
        `(A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D)` then the
        returned tensor will be of shape: `(A \times B \times C \times D)(A×B×C×D)`.

        When `dim` is given, a `squeeze` operation is done only in the given
        dimension. If `self` is of shape: `(A \times 1 \times B)(A×1×B)` ,
        `squeeze(self, 0)` leaves the tensor unchanged, but `squeeze(self, 1)`
        will squeeze the tensor to the shape `(A \times B)(A×B)`
        """
        raise NotImplementedError("squeeze is not implemented")

    def repeat(self, *sizes):
        """
        Repeats this tensor along the specified dimensions.

        Unlike expand(), this function copies the tensor’s data.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this tensor along each
                dimension
        """
        raise NotImplementedError("repeat is not implemented")

    def narrow(self, dim, start, length):
        """
        Returns a new tensor that is a narrowed version of `input` tensor.
        The dimension `dim` is `input` from `start` to `start + length`.
        The returned `tensor` and `input` tensor share the same underlying storage.
        """
        raise NotImplementedError("narrow is not implemented")

    def expand(self, *sizes):
        """
        Returns a new view of the `self` tensor with singleton dimensions
        expanded to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the size
        cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a new
        view on the existing tensor where a dimension of size one is expanded to
        a larger size by setting the `stride` to 0. Any dimension of size 1 can
        be expanded to an arbitrary value without allocating new memory.
        """
        raise NotImplementedError("expand is not implemented")

    def roll(self, shifts, dims=None):
        """
        Roll the tensor along the given dimension(s). Elements that are shifted
        beyond the last position are re-introduced at the first position. If a
        dimension is not specified, the tensor will be flattened before rolling
        and then restored to the original shape.
        """
        raise NotImplementedError("roll is not implemented")

    def unfold(self, dimension, size, step):
        """
        Returns a tensor which contains all slices of size `size` from `self`
        tensor in the dimension `dimension`.

        Step between two slices is given by `step`.

        If sizedim is the size of dimension `dimension` for `self`, the size of
        dimension `dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size `size` is appended in the returned tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice
        """
        raise NotImplementedError("unfold is not implemented")

    def take(self, index):
        """
        Returns a new tensor with the elements of `input` at the given indices.
        The input tensor is treated as if it were viewed as a 1-D tensor.
        The result takes the same shape as the indices.
        """
        raise NotImplementedError("take is not implemented")

    def flip(self, input, dims):
        """
        Reverse the order of a n-D tensor along given axis in dims.

        Args:
            dims (a list or tuple): axis to flip on
        """
        raise NotImplementedError("flip is not implemented")

    def trace(self):
        """
        Returns the sum of the elements of the diagonal of the input 2-D matrix.
        """
        raise NotImplementedError("trace is not implemented")

    def sum(self, dim=None, keepdim=False):
        """
        Returns the sum of all elements in the `self` tensor.

        If :attr:`dim` is a list of dimensions,
        reduce over all of them.
        """
        raise NotImplementedError("sum is not implemented")

    def cumsum(self, dim):
        """
        Returns the cumulative sum of elements of :attr:`self` in the dimension
        :attr:`dim`.

        For example, if :attr:`self` is a vector of size N, the result will also be
        a vector of size N, with elements.

        .. math::
            y_i = x_1 + x_2 + x_3 + \dots + x_i

        Args:
            dim  (int): the dimension to do the operation over
        """
        raise NotImplementedError("cumsum is not implemented")

    def reshape(self, shape):
        """
        Returns a tensor with the same data and number of elements as `self`,
        but with the specified shape.

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        raise NotImplementedError("reshape is not implemented")

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by dim.

        For a 3-D tensor the output is specified by:
            - out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            - out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            - out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        """
        raise NotImplementedError("reshape is not implemented")

    # properties:
    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")

    def numel(self):
        raise NotImplementedError("numel is not implemented")

    def nelement(self):
        raise NotImplementedError("nelement is not implemented")

    def dim(self):
        raise NotImplementedError("dim is not implemented")

    def size(self):
        raise NotImplementedError("size is not implemented")

    @property
    def shape(self):
        return self.size()
