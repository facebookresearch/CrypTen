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
        """Adds :attr:`tensor` to :attr:`self` (in-place) see :meth:`add`."""
        raise NotImplementedError("add_ is not implemented")

    def add(self, tensor):
        r"""Adds :attr:`tensor` to this :attr:`self`.

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
        """Subtracts :attr:`tensor` from :attr:`self` (in-place), see :meth:`sub`"""
        raise NotImplementedError("sub_ is not implemented")

    def sub(self, tensor):
        """Subtracts a :attr:`tensor` from :attr:`self` tensor.
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
        """Element-wise multiply with a :attr:`tensor` in-place, see :meth:`mul`."""
        raise NotImplementedError("mul_ is not implemented")

    def mul(self, tensor):
        r"""Element-wise multiply with a :attr:`tensor`.

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
        """Element-wise in-place divide by a :attr:`tensor` (see :meth:`div`)."""
        raise NotImplementedError("div_ is not implemented")

    def div(self, tensor):
        r"""
        Divides each element of :attr:`self` with the :attr:`tensor`
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
        r"""Performs matrix multiplication of :attr:`self` with :attr:`tensor`

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

    def sqrt(self):
        """
        Computes the square root of :attr:`self`
        """
        raise NotImplementedError("sqrt is not implemented")

    def square(self):
        """
        Computes the square of :attr:`self`
        """
        raise NotImplementedError("square is not implemented")

    def norm(self, p="fro", dim=None, keepdim=False):
        """
        Computes the p-norm of the :attr:`self` (or along a dimension)

        Args:
            p (str, int, or float): specifying type of p-norm
            dim (int): optional dimension along which to compute p-norm
            keepdim (bool): whether the output tensor has `dim` retained or not
        """
        raise NotImplementedError("norm is not implemented")

    def mean(self, dim=None):
        """Compute mean."""
        raise NotImplementedError("mean is not implemented")

    def var(self, dim=None):
        """Compute variance."""
        raise NotImplementedError("var is not implemented")

    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        raise NotImplementedError("relu is not implemented")

    def argmax(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the maximum value of all elements in
        :attr:`self`

        If multiple values are equal to the maximum, ties will be broken
        (randomly). Note that this deviates from PyTorch's implementation since
        PyTorch does not break ties randomly, but rather returns the lowest
        index of a maximal value.

        If :attr:`keepdim` is True, the output tensor are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors having 1
        fewer dimension than :attr:`self`.

        If :attr:`one_hot` is True, the output tensor will have the same size as the
        :attr:`self` and contain elements of value `1` on argmax indices (with random
        tiebreaking) and value `0` on other indices.
        """
        raise NotImplementedError("argmax is not implemented")

    def argmin(self, dim=None, keepdim=False, one_hot=False):
        """Returns the indices of the minimum value of all elements in the
        :attr:`self`

        If multiple values are equal to the minimum, ties will be broken
        (randomly). Note that this deviates from PyTorch's implementation since
        PyTorch does not break ties randomly, but rather returns the lowest
        index of a minimal value.

        If :attr:`keepdim` is True, the output tensor are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors having 1
        fewer dimension than :attr:`self`.

        If :attr:`one_hot` is True, the output tensor will have the same size as the
        :attr:`self` and contain elements of value `1` on argmin indices (with random
        tiebreaking) and value `0` on other indices.
        """
        raise NotImplementedError("argmin is not implemented")

    def max(self, dim=None, keepdim=False, one_hot=False):
        """Returns the maximum value of all elements in :attr:`self`

        If :attr:`dim` is specified, returns a tuple `(values, indices)` where
        `values` is the maximum value of each row of :attr:`self` in the
        given dimension :attr:`dim`. And `indices` is the result of an :meth:`argmax` call with
        the same keyword arguments (:attr:`dim`, :attr:`keepdim`, and :attr:`one_hot`)

        If :attr:`keepdim` is True, the output tensors are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors having 1
        fewer dimension than :attr:`self`
        """
        raise NotImplementedError("max is not implemented")

    def min(self, dim=None, keepdim=False, one_hot=False):
        """Returns the minimum value of all elements in :attr:`self`.

        If `dim` is sepcified, returns a tuple `(values, indices)` where
        `values` is the minimum value of each row of :attr:`self` tin the
        given dimension :attr:`dim`. And :attr:`indices` is the result of an :meth:`argmin` call with
        the same keyword arguments (:attr:`dim`, :attr:`keepdim`, and :attr:`one_hot`)

        If `keepdim` is True, the output tensors are of the same size as
        :attr:`self` except in the dimension :attr:`dim` where they are of size 1.
        Otherwise, :attr:`dim` is squeezed, resulting in the output tensors having 1
        fewer dimension than :attr:`self`
        """
        raise NotImplementedError("min is not implemented")

    def batchnorm(
        self,
        ctx,
        weight,
        bias,
        running_mean=None,
        running_var=None,
        training=False,
        eps=1e-05,
        momentum=0.1,
    ):
        """Batch normalization."""
        raise NotImplementedError("batchnorm is not implemented")

    def conv2d(self, *args, **kwargs):
        """2D convolution."""
        raise NotImplementedError("conv2d is not implemented")

    def max_pool2d(self, kernel_size, padding=None, stride=None, return_indices=False):
        """Applies a 2D max pooling over an input signal composed of several
        input planes.

        If ``return_indices`` is True, this will return the one-hot max indices
        along with the outputs.

        These indices will be returned as with dimensions equal to the
        ``max_pool2d`` output dimensions plus the kernel dimensions. This is because
        each returned index will be a one-hot kernel for each element of the
        output that corresponds to the maximal block element of the corresponding
        input block.

        A max pool with output tensor of size :math:`(i, j, k, l)` with kernel
        size :math:`m` and will return an index tensor of size
        :math:`(i, j, k, l, m, m)`.

        [ 0,  1,  2,  3]                    [[0, 0], [0, 0]]
        [ 4,  5,  6,  7]         ->         [[0, 1], [0, 1]]
        [ 8,  9, 10, 11]         ->         [[0, 0], [0, 0]]
        [12, 13, 14, 15]                    [[0, 1], [0, 1]]

        Note: This deviates from PyTorch's implementation since PyTorch returns
        the index values for each element rather than a one-hot kernel. This
        deviation is useful for implementing ``_max_pool2d_backward`` later.
        """
        raise NotImplementedError("max_pool2d is not implemented")

    def _max_pool2d_backward(
        self, indices, kernel_size, padding=None, stride=None, output_size=None
    ):
        """Implements the backwards for a `max_pool2d` call where `self` is
        the output gradients and `indices` is the 2nd result of a `max_pool2d`
        call where `return_indices` is True.

        The output of this function back-propagates the gradient (from `self`)
        to be computed with respect to the input parameters of the `max_pool2d`
        call.

        `max_pool2d` can map several input sizes to the same output sizes. Hence,
        the inversion process can get ambiguous. To accommodate this, you can
        provide the needed output size as an additional argument `output_size`.
        Otherwise, this will return a tensor the minimal size that will produce
        the correct mapping.
        """
        raise NotImplementedError("_max_pool2d_backward is not implemented")

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or CrypTensor): values selected at indices
                where condition is False.

        Returns: CrypTensor or torch.tensor
        """
        raise NotImplementedError("where is not implemented")

    def sigmoid(self, reciprocal_method="log"):
        """Computes the sigmoid function on the input value
                sigmoid(x) = (1 + exp(-x))^{-1}
        """
        raise NotImplementedError("sigmoid is not implemented")

    def tanh(self, reciprocal_method="log"):
        """Computes tanh from the sigmoid function:
            tanh(x) = 2 * sigmoid(2 * x) - 1
        """
        raise NotImplementedError("tanh is not implemented")

    def softmax(self, dim, **kwargs):
        """Compute the softmax of a tensor's elements along a given dimension
        """
        raise NotImplementedError("softmax is not implemented")

    def cos(self):
        """Computes the cosine of the input."""
        raise NotImplementedError("cos is not implemented")

    def sin(self):
        """Computes the sine of the input."""
        raise NotImplementedError("sin is not implemented")

    # Approximations:
    def exp(self):
        """Computes exponential function on the tensor."""
        raise NotImplementedError("exp is not implemented")

    def log(self):
        """Computes the natural logarithm of the tensor."""
        raise NotImplementedError("log is not implemented")

    def reciprocal(self):
        """Computes the reciprocal of the tensor."""
        raise NotImplementedError("reciprocal is not implemented")

    def eq(self, tensor):
        """Element-wise equality

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

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

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted boolean tensor containing a True at each location where
            comparison is true.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        """
        raise NotImplementedError("ne is not implemented")

    def __ne__(self, tensor):
        """Element-wise inequality"""
        return self.ne(tensor)

    def ge(self, tensor):
        """Element-wise greater than or equal to

        The :attr:`tensor` argument can be a number or a tensor whose shape is
        `broadcastable`_ with :attr:`self`

        Args:
            tensor (Tensor or float): the tensor or value to compare

        Returns:
            an encrypted boolean valued tensor containing a True at each location where
            comparison is true

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
            an encrypted boolean valued tensor containing a True at each location where
            comparison is true

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
            an encrypted boolean valued tensor containing a True at each location where
            comparison is true

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
            an encrypted boolean valued tensor containing a True at each location where
            comparison is true

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

    def ger(self, tensor):
        """Compute outer product."""
        raise NotImplementedError("ger is not implemented")

    def index_add(self, dim, index, tensor):
        """Accumulate the elements of :attr:`tensor` into
        :attr:`self` by adding to the indices in the order given in :attr:`index`

        Example: if ``dim == 0`` and ``index[i] == j``,
            then the ``i``\ th row of tensor is added to the ``j``\ th row of :attr:`self`

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of tensor to select from
            tensor (MPCTensor or torch.Tensor): containing values to add
        """
        raise NotImplementedError("index_add is not implemented")

    def index_add_(self, dim, index, tensor):
        """Accumulate the elements of :attr:`tensor` into
        :attr:`self` by adding to the indices in the order given in :attr:`index`

        Example: if ``dim == 0`` and ``index[i] == j``,
            then the ``i``\ th row of tensor is added to the ``j``\ th row of :attr:`self`

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of tensor to select from
            tensor (MPCTensor or torch.Tensor): containing values to add
        """
        raise NotImplementedError("index_add_ is not implemented")

    def scatter_add(self, dim, index, other):
        """Adds all values from the :attr:`other` into :attr:`self` at the indices
        specified in :attr:`index`. This an out-of-place version of
        :meth:`scatter_add_`. For each value in :attr:`other`, it is added to an
        index in :attr:`self` which is specified by its index in :attr:`other`
        for ``dimension != dim`` and by the corresponding
        value in index for ``dimension = dim``.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add,
                can be either empty or the same size of src.
                When empty, the operation returns identity.
            other (Tensor): the source elements to scatter and add
        """
        raise NotImplementedError("scatter_add is not implemented")

    def scatter_add_(self, dim, index, other):
        """Adds all values from the :attr:`other` into :attr:`self` at the indices
        specified in :attr:`index`.
        For each value in :attr:`other`, it is added to an
        index in :attr:`self` which is specified by its index in :attr:`other`
        for ``dimension != dim`` and by the corresponding
        value in index for ``dimension = dim``.


        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add,
                can be either empty or the same size of src.
                When empty, the operation returns identity.
            other (Tensor): the source elements to scatter and add
        """
        raise NotImplementedError("scatter_add_ is not implemented")

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
        Sets elements of `self` at index `index` to `value`.
        """
        raise NotImplementedError("__setitem__ is not implemented")

    def index_select(self, dim, index):
        """
        Returns a new tensor which indexes the :attr:`self` tensor along dimension
        :attr:`dim` using the entries in :attr:`index`.

        The returned tensor has the same number of dimensions as :attr:`self`
        The dimension :attr:`dim` has the same size as the length
        of :attr:`index`; other dimensions have the same size as in :attr:`self`.
        """
        raise NotImplementedError("index_select is not implemented")

    def view(self, *shape):
        r"""
        Returns a new encrypted tensor with the same data as the :attr:`self` tensor
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
        Expects :attr:`self` to be <= 2D tensor and transposes dimensions 0 and 1.

        0D and 1D tensors are returned as is and for 2D tensors this can be
        seen as a short-hand function for `self.transpose(0, 1)`.
        """
        raise NotImplementedError("t is not implemented")

    def transpose(self, dim0, dim1):
        """
        Returns a tensor that is a transposed version of :attr:`self`
        The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

        The resulting tensor shares it’s underlying storage with :attr:`self`,
        so changing the content of one would change the content of the
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

        The returned tensor shares the same underlying data with :attr:`self`

        A :attr:`dim` value within the range `[-self.dim() - 1, self.dim() + 1)`
        can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze` applied at
        `dim = dim + self.dim() + 1`

        Args:
            dim (int): the index at which to insert the singleton dimension
        """
        raise NotImplementedError("unsqueeze is not implemented")

    def squeeze(self, dim=None):
        """
        Returns a tensor with all the dimensions of :attr:`self` of size 1 removed.

        For example, if :attr:`self` is of shape:
        `(A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D)` then the
        returned tensor will be of shape: `(A \times B \times C \times D)(A×B×C×D)`.

        When :attr:`dim` is given, a :meth:`squeeze` operation is done only in the given
        dimension. If :attr:`self` is of shape: `(A \times 1 \times B)(A×1×B)` ,
        `squeeze(self, 0)` leaves the tensor unchanged, but `squeeze(self, 1)`
        will squeeze the tensor to the shape `(A \times B)(A×B)`
        """
        raise NotImplementedError("squeeze is not implemented")

    def repeat(self, *sizes):
        """
        Repeats :attr:`self` along the specified dimensions.

        Unlike expand(), this function copies the tensor’s data.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this tensor along each
                dimension
        """
        raise NotImplementedError("repeat is not implemented")

    def narrow(self, dim, start, length):
        """
        Returns a new tensor that is a narrowed version of :attr:`self`
        The dimension :attr:`dim` is input from :attr:`start` to :attr:`start + length`.
        The returned tensor and :attr:`self` share the same underlying storage.
        """
        raise NotImplementedError("narrow is not implemented")

    def expand(self, *sizes):
        """
        Returns a new view of :attr:`self` with singleton dimensions
        expanded to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the size
        cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a new
        view on the existing tensor where a dimension of size one is expanded to
        a larger size by setting the :attr:`stride` to 0. Any dimension of size 1 can
        be expanded to an arbitrary value without allocating new memory.
        """
        raise NotImplementedError("expand is not implemented")

    def roll(self, shifts, dims=None):
        """
        Roll :attr:`self` along the given dimensions :attr:`dims`. Elements that are shifted
        beyond the last position are re-introduced at the first position. If a
        dimension is not specified, the tensor will be flattened before rolling
        and then restored to the original shape.
        """
        raise NotImplementedError("roll is not implemented")

    def unfold(self, dimension, size, step):
        """
        Returns a tensor which contains all slices of size :attr:`size` from :attr:`self`
        in the dimension :attr:`dimension`.

        Step between two slices is given by :attr:`step`

        If `sizedim` is the size of :attr:`dimension` for :attr:`self`, the size of
        :attr:`dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size :attr:`size` is appended in the returned tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice
        """
        raise NotImplementedError("unfold is not implemented")

    def take(self, index, dimension=None):
        """
        Returns a new tensor with the elements of :attr:`input` at the given indices.
        When the dimension is None, :attr:`self` tensor is treated as if it were
        viewed as a 1D tensor, and the result takes the same shape as the indices.
        When the dimension is an integer, the result take entries of tensor along a
        dimension according to the :attr:`index`.
        """
        raise NotImplementedError("take is not implemented")

    def flip(self, input, dims):
        """
        Reverse the order of a n-D tensor along given axis in dims.

        Args:
            dims (a list or tuple): axis to flip on
        """
        raise NotImplementedError("flip is not implemented")

    def pad(self, pad, mode="constant", value=0):
        """Pads tensor with constant."""
        raise NotImplementedError("pad is not implemented")

    def trace(self):
        """
        Returns the sum of the elements of the diagonal of :attr:`self`.
        :attr:`self` has to be a 2D tensor.
        """
        raise NotImplementedError("trace is not implemented")

    def sum(self, dim=None, keepdim=False):
        """
        Returns the sum of all elements in the :attr:`self`

        If :attr:`dim` is a list of dimensions,
        reduce over all of them.
        """
        raise NotImplementedError("sum is not implemented")

    def cumsum(self, dim):
        """
        Returns the cumulative sum of elements of :attr:`self` in the dimension
        :attr:`dim`

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
        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified :attr:`shape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        raise NotImplementedError("reshape is not implemented")

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by :attr:`dim`

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
