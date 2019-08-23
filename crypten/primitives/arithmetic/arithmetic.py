#!/usr/bin/env python3
from functools import reduce

import crypten.common.constants as constants

# dependencies:
import torch
from crypten import comm
from crypten.common import EncryptedTensor, FixedPointEncoder
from crypten.common.rng import generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor, is_int_tensor
from crypten.trusted_third_party import TrustedThirdParty

from .beaver import Beaver


SENTINEL = -1


# MPC tensor where shares additive-sharings.
class ArithmeticSharedTensor(EncryptedTensor):
    """
        Encrypted tensor object that uses additive sharing to perform computations.

        Additive shares are computed by splitting each value of the input tensor
        into n separate random values that add to the input tensor, where n is
        the number of parties present in the protocol (world_size).
    """

    # constructors:
    def __init__(self, tensor=None, size=None, precision=constants.PRECISION, src=0):
        if src == SENTINEL:
            return

        self.encoder = FixedPointEncoder(precision_bits=precision)
        if tensor is not None:
            if is_int_tensor(tensor) and precision != 0:
                tensor = tensor.float()
            tensor = self.encoder.encode(tensor)
            size = tensor.size()

        # Generate psuedo-random sharing of zero (PRZS) and add source's tensor
        self._tensor = ArithmeticSharedTensor.PRZS(size)._tensor
        if self.rank == src:
            assert tensor is not None, "Source must provide a data tensor"
            self._tensor += tensor

    @staticmethod
    def from_shares(share, precision=constants.PRECISION, src=0):
        """Generate an AdditiveSharedTensor from a share from each party"""
        result = ArithmeticSharedTensor(src=SENTINEL)
        result._tensor = share
        result.encoder = FixedPointEncoder(precision_bits=precision)
        return result

    @staticmethod
    def PRZS(*size):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. One of these parties adds
        this number while the other subtracts this number.
        """
        tensor = ArithmeticSharedTensor(src=SENTINEL)
        current_share = generate_random_ring_element(*size, generator=comm.g0)
        next_share = generate_random_ring_element(*size, generator=comm.g1)
        tensor._tensor = current_share - next_share
        return tensor

    @property
    def rank(self):
        return comm.get_rank()

    def shallow_copy(self):
        """Create a shallow copy"""
        result = ArithmeticSharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor
        return result

    def __repr__(self):
        return "[%s] ArithmeticSharedTensor" % self.size()

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if isinstance(value, (int, float)) or torch.is_tensor(value):
            value = ArithmeticSharedTensor(value)
        assert isinstance(
            value, ArithmeticSharedTensor
        ), "Unsupported input type %s for __setitem__" % type(value)
        self._tensor.__setitem__(index, value._tensor)

    def pad(self, pad, mode="constant", value=0):
        """
            Pads the input tensor with values provided in `value`.
        """
        assert mode == "constant", (
            "Padding with mode %s is currently unsupported" % mode
        )

        result = self.shallow_copy()
        if isinstance(value, (int, float)):
            value = self.encoder.encode(value).item()
            if result.rank == 0:
                result._tensor = torch.nn.functional.pad(
                    result._tensor, pad, mode=mode, value=value
                )
            else:
                result._tensor = torch.nn.functional.pad(
                    result._tensor, pad, mode=mode, value=0
                )
        elif isinstance(value, ArithmeticSharedTensor):
            assert (
                value.dim() == 0
            ), "Private values used for padding must be 0-dimensional"
            value = value._tensor.item()
            result._tensor = torch.nn.functional.pad(
                result._tensor, pad, mode=mode, value=value
            )
        else:
            raise TypeError(
                "Cannot pad ArithmeticSharedTensor with a %s value" % type(value)
            )

        return result

    @staticmethod
    def stack(tensors, *args, **kwargs):
        """Perform tensor stacking"""
        for i, tensor in enumerate(tensors):
            if torch.is_tensor(tensor):
                tensors[i] = ArithmeticSharedTensor(tensor)
            assert isinstance(
                tensors[i], ArithmeticSharedTensor
            ), "Can't stack %s with ArithmeticSharedTensor" % type(tensor)

        result = tensors[0].shallow_copy()
        result._tensor = torch.stack(
            [tensor._tensor for tensor in tensors], *args, **kwargs
        )
        return result

    def reveal(self):
        """Get plaintext without any downscaling"""
        tensor = self._tensor.clone()
        return comm.all_reduce(tensor)

    def get_plain_text(self):
        """Decrypt the tensor"""
        return self.encoder.decode(self.reveal())

    def _arithmetic_function_(self, y, op, *args, **kwargs):
        return self._arithmetic_function(y, op, inplace=True, *args, **kwargs)

    def _arithmetic_function(self, y, op, inplace=False, *args, **kwargs):
        assert op in [
            "add",
            "sub",
            "mul",
            "matmul",
            "conv2d",
        ], f"Provided op `{op}` is not a supported arithmetic function"

        additive_func = op in ["add", "sub"]
        public = isinstance(y, (int, float)) or torch.is_tensor(y)
        private = isinstance(y, ArithmeticSharedTensor)

        if inplace:
            result = self
            if additive_func or (op == "mul" and public):
                op += "_"
        else:
            result = self.clone()

        if public:
            y = result.encoder.encode(y)

            if additive_func:  # ['add', 'sub']
                if result.rank == 0:
                    result._tensor = getattr(result._tensor, op)(y)
                else:
                    result._tensor = torch.broadcast_tensors(result._tensor, y)[0]
            elif op == "mul_":  # ['mul_']
                result._tensor = result._tensor.mul_(y)
            else:  # ['mul', 'matmul', 'conv2d']
                result._tensor = getattr(torch, op)(result._tensor, y, *args, **kwargs)
        elif private:
            if additive_func:  # ['add', 'sub', 'add_', 'sub_']
                result._tensor = getattr(result._tensor, op)(y._tensor)
            else:  # ['mul', 'matmul', 'conv2d'] - Note: 'mul_' calls 'mul' here
                # Must copy _tensor.data here to support 'mul_' being inplace
                result._tensor.data = getattr(Beaver, op)(
                    result, y, *args, **kwargs
                )._tensor.data
        else:
            raise TypeError("Cannot %s %s with %s" % (op, type(y), type(self)))

        if not additive_func:
            return result.div_(result.encoder.scale)
        return result

    def add(self, y):
        """Perform element-wise addition"""
        return self._arithmetic_function(y, "add")

    def add_(self, y):
        """Perform element-wise addition"""
        return self._arithmetic_function_(y, "add")

    def sub(self, y):
        """Perform element-wise subtraction"""
        return self._arithmetic_function(y, "sub")

    def sub_(self, y):
        """Perform element-wise subtraction"""
        return self._arithmetic_function_(y, "sub")

    def mul(self, y):
        """Perform element-wise multiplication"""
        if isinstance(y, int) or is_int_tensor(y):
            return self.clone().mul_(y)
        return self._arithmetic_function(y, "mul")

    def mul_(self, y):
        """Perform element-wise multiplication"""
        if isinstance(y, int) or is_int_tensor(y):
            self._tensor *= y
            return self
        return self._arithmetic_function_(y, "mul")

    def div(self, y):
        """Divide by a given tensor"""
        result = self.clone()
        if isinstance(y, EncryptedTensor):
            result._tensor = torch.broadcast_tensors(result._tensor, y._tensor)[
                0
            ].clone()
        elif torch.is_tensor(y):
            result._tensor = torch.broadcast_tensors(result._tensor, y)[0].clone()
        return result.div_(y)

    def div_(self, y):
        """Divide two tensors element-wise"""
        # Truncate protocol for dividing by public integers:
        if isinstance(y, int) or is_int_tensor(y):
            if comm.get_world_size() > 2:
                wraps = self.wraps()
                self._tensor /= y
                # NOTE: The multiplication here must be split into two parts
                # to avoid long out-of-bounds when y <= 2 since (2 ** 63) is
                # larger than the largest long integer.
                self -= wraps * 4 * (int(2 ** 62) // y)
            else:
                self._tensor /= y
            return self

        # Otherwise multiply by reciprocal
        if isinstance(y, float):
            y = torch.FloatTensor([y])

        assert is_float_tensor(y) or isinstance(
            y, ArithmeticSharedTensor
        ), "Unsupported type for div_: %s" % type(y)

        return self.mul_(y.reciprocal())

    def wraps(self):
        """Privately computes the number of wraparounds for a set a shares"""
        return Beaver.wraps(self)

    def matmul(self, y):
        """Perform matrix multiplication using some tensor"""
        return self._arithmetic_function(y, "matmul")

    def mean(self, *args, **kwargs):
        """Computes mean of given tensor"""
        result = self.sum(*args, **kwargs)
        sizes_summed = self.size(*args, **kwargs)
        if isinstance(sizes_summed, int):
            divisor = sizes_summed
        else:
            assert len(sizes_summed) > 0, "size of input tensor cannot be 0"
            divisor = reduce(lambda x, y: x * y, sizes_summed)
        return result.div(divisor)

    def conv2d(self, kernel, **kwargs):
        """Perform a 2D convolution using the given kernel"""
        return self._arithmetic_function(kernel, "conv2d", **kwargs)

    def avg_pool2d(self, kernel_size, *args, **kwargs):
        """Perform an average pooling on each 2D matrix of the given tensor"""
        z = self.sum_pool2d(kernel_size, *args, **kwargs)
        pool_size = kernel_size * kernel_size
        return z / pool_size

    def sum_pool2d(self, *args, **kwargs):
        """Perform a sum pooling on each 2D matrix of the given tensor"""
        result = self.shallow_copy()
        result._tensor = torch.nn.functional.avg_pool2d(
            self._tensor, *args, **kwargs, divisor_override=1
        )
        return result

    def softmax(self, apprx_max=5.0, **kwargs):
        """Compute the max of a tensor's elements (or along a given dimension)

        This is computed by
            softmax(x)
                = exp(x) * reciprocal(sum(exp(x)))
                = exp(x) * exp( -log( sum(exp(x)))

        For large x, exp(x) will be extremely large and therefore sum(exp(x))
        will be out of the accurate range for log(). Therefore we note that the
        above can be corrected by subtracting a constant:

                = exp(x - c) * exp( -log( sum(exp(x - c))))
        """
        numerator = self.exp()
        # correction should be approximately the maximum value
        denominator = (self - apprx_max).exp().sum().reciprocal()
        result = numerator * denominator

        correction = torch.empty(size=result.size()).fill_(apprx_max)
        correction = (-correction).exp()
        return result * correction

    # negation and reciprocal:
    def neg_(self):
        """Negate the tensor's values"""
        self._tensor.neg_()
        return self

    def neg(self):
        """Negate the tensor's values"""
        return self.clone().neg_()

    # Approximated functions
    def exp(self, iterations=8):
        """Approximates the exponential function using a limit approximation:

            exp(x) = lim (1 + x / n) ^ n

        Here we compute exp by choosing n = 2 ** d for some large d and then
        computing (1 + x / n) once and squaring d times.
        """
        result = 1 + self.div(2 ** iterations)
        for _ in range(iterations):
            result = result.square()
        return result

    def log(self, iterations=1, exp_iterations=8):
        """Approximates the natural logarithm using 6th order modified
        Householder iterations:

        Iterations are computed by:
                              h = 1 - x * exp(-y_n)
        y_{n+1} = y_n - h * (1 + h / 2 + h^2 / 3 + h^3 / 6 + h^4 / 5 + h^5 / 7)
        """

        # Initialization to a decent estimate (found by qualitative inspection):
        #                ln(x) = x/40 - 8exp(-2x - .3) + 1.9
        term1 = self / 40
        term2 = 8 * (-2 * self - 0.3).exp()
        y = term1 - term2 + 1.9

        # 6th order Householder iterations
        for _ in range(iterations):
            h = 1 - self * (-y).exp(iterations=exp_iterations)
            h2 = h.square()
            h3 = h2 * h
            h4 = h2.square()
            h5 = h4 * h
            y -= h * (1 + h.div(2) + h2.div_(3) + h3.div_(6) + h4.div_(5) + h5.div_(7))

        return y

    def pow(self, p):
        """
        Approximates self ^ p by computing:
            x ^ p = exp(p * log(x))
        """
        return self.log().mul_(p).exp(iterations=9)

    def reciprocal(self, method="NR", nr_iters=7, log_iters=1, exp_iters=9):
        """
        Methods:
            'NR' : Newton Raphson method computes the reciprocal using iterations
                    of x[i+1] = (2x[i] - self * x[i]^2) and uses
                    3exp(-(x-.5)) + 0.003 as an initial guess

            'log' : Computes the reciprocal of the input from the observation that:
                    x ^ -1 = exp(-log(x))
        """
        if method == "NR":
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(.5 - x) + 0.003
            result = 3 * (0.5 - self).exp() + 0.003
            for _ in range(nr_iters):
                result += result - result.square().mul_(self)
            return result
        elif method == "log":
            return (-self.log(iterations=log_iters)).exp(iterations=exp_iters)
        else:
            raise ValueError("Invalid method %s given for reciprocal function" % method)

    def sqrt(self):
        """
        Computes the square root of the input by raising it to the 0.5 power
        """
        return self.pow(0.5)

    def square(self):
        result = Beaver.square(self).div_(self.encoder.scale)
        return result

    def norm(self, *args, **kwargs):
        return self.square().sum(*args, **kwargs).pow(0.5)

    def _eix(self, iterations=10):
        """Computes e^(i * self) where i is the imaginary unit.
        Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
        """
        re = 1
        im = self.div(2 ** iterations)

        # First iteration uses knowledge that `re` is public and = 1
        re -= im.square()
        im *= 2

        # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
        for _ in range(iterations - 1):
            a2 = re.square()
            b2 = im.square()
            im = im.mul_(re)
            im._tensor *= 2
            re = a2 - b2

        return re, im

    def cos(self, iterations=10):
        """Computes the cosine of the input using cos(x) = Re{exp(i * x)}"""
        return self._eix(iterations=iterations)[0]

    def sin(self, iterations=10):
        """Computes the sine of the input using sin(x) = Im{exp(i * x)}"""
        return self._eix(iterations=iterations)[1]

    # copy between CPU and GPU:
    def cuda(self):
        raise NotImplementedError("CUDA is not supported for ArithmeticSharedTensors")

    def cpu(self):
        raise NotImplementedError("CUDA is not supported for ArithmeticSharedTensors")

    def dot(self, y, weights=None):
        """Compute a dot product between two tensors"""
        assert self.size() == y.size(), "Number of elements do not match"
        if weights is not None:
            assert weights.size() == self.size(), "Incorrect number of weights"
            result = self * weights
        else:
            result = self.clone()

        return result.mul_(y).sum()

    def ger(self, y):
        """Computer an outer product between two vectors"""
        assert self.dim() == 1 and y.dim() == 1, "Outer product must be on 1D tensors"
        return self.view((-1, 1)).matmul(y.view((1, -1)))
