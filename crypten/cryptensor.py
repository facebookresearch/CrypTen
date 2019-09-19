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
        """Adds tensor to this tensor (in-place)."""
        raise NotImplementedError("add_ is not implemented")

    def add(self, tensor):
        """Adds tensor to this tensor."""
        raise NotImplementedError("add is not implemented")

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def sub_(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        raise NotImplementedError("sub_ is not implemented")

    def sub(self, tensor):
        """Subtracts tensor from this tensor."""
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
        """Element-wise multiply with a tensor (in-place)."""
        raise NotImplementedError("mul_ is not implemented")

    def mul(self, tensor):
        """Element-wise multiply with a tensor."""
        raise NotImplementedError("mul is not implemented")

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def div_(self, scalar):
        """Element-wise divide by a tensor (in-place)."""
        raise NotImplementedError("div_ is not implemented")

    def div(self, scalar):
        """Element-wise divide by a tensor."""
        raise NotImplementedError("div is not implemented")

    def __div__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def neg(self):
        """Negative value of a tensor"""
        raise NotImplementedError("neg is not implemented")

    def neg_(self):
        """Negative value of a tensor (in-place)"""
        raise NotImplementedError("neg_ is not implemented")

    def __neg__(self):
        return self.neg()

    def matmul(self, tensor):
        """Perform matrix multiplication using some tensor"""
        raise NotImplementedError("matmul is not implemented")

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def eq(self, tensor):
        """Element-wise equality"""
        raise NotImplementedError("eq is not implemented")

    def __eq__(self, tensor):
        """Element-wise equality"""
        return self.eq(tensor)

    def ne(self, tensor):
        """Element-wise inequality"""
        raise NotImplementedError("ne is not implemented")

    def __ne__(self, tensor):
        """Element-wise inequality"""
        return self.ne(tensor)

    def ge(self, tensor):
        """Element-wise greater than or equal to"""
        raise NotImplementedError("ge is not implemented")

    def __ge__(self, tensor):
        """Element-wise greater than or equal to"""
        return self.ge(tensor)

    def gt(self, tensor):
        """Element-wise greater than"""
        raise NotImplementedError("gt is not implemented")

    def __gt__(self, tensor):
        """Element-wise greater than"""
        return self.gt(tensor)

    def le(self, tensor):
        """Element-wise less than or equal to"""
        raise NotImplementedError("le is not implemented")

    def __le__(self, tensor):
        """Element-wise less than or equal to"""
        return self.le(tensor)

    def lt(self, tensor):
        """Element-wise less than"""
        raise NotImplementedError("lt is not implemented")

    def __lt__(self, tensor):
        """Element-wise less than"""
        return self.lt(tensor)

    def dot(self, tensor, weights=None):
        """Perform (weighted) inner product with plain or cipher text."""
        raise NotImplementedError("dot is not implemented")

    # Regular functions:
    def clone(self):
        """
        Returns a deep copy of the `self` tensor.
        The copy has the same size and ptype as `self`.
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
        """
        Returns a new encrypted tensor with the same data as the `self` tensor
        but of a different shape.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size.
        """
        raise NotImplementedError("view is not implemented")

    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens a contiguous range of dims in a tensor."""
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
        """
        raise NotImplementedError("t is not implemented")

    def unsqueeze(self, dim):
        """
        Returns a new tensor with a dimension of size one inserted at the
        specified position.

        The returned tensor shares the same underlying data with this tensor.

        A `dim` value within the range `[-self.dim() - 1, self.dim() + 1)`
        can be used. Negative `dim` will correspond to `unsqueeze()`` applied at
        `dim = dim + self.dim() + 1`
        """
        raise NotImplementedError("unsqueeze is not implemented")

    def squeeze(self, dim=None):
        """
        Returns a tensor with all the dimensions of `self` of size 1 removed.

        For example, if `self` is of shape:
        `(A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D)` then the
        returned tensor will be of shape: ``(A \times B \times C \times D)(A×B×C×D)`.

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
        ``(sizedim - size) / step + 1`.

        An additional dimension of size `size` is appended in the returned tensor.
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
        """
        raise NotImplementedError("sum is not implemented")

    def cumsum(self, dim):
        """
        Returns the cumulative sum of elements of `self` in the dimension `dim`.
        """
        raise NotImplementedError("cumsum is not implemented")

    def reshape(self, shape):
        """
        Returns a tensor with the same data and number of elements as `self`,
        but with the specified shape.
        """
        raise NotImplementedError("reshape is not implemented")

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by dim.

        For a 3-D tensor the output is specified by:
        ```
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        ```
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
