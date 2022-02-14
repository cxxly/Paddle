# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
import numbers
import typing

import paddle
import paddle.nn.functional as F
from paddle.distribution import (constraint, distribution, tool,
                                 transformed_distribution, variable)


__all__ = [  # noqa
    'Transform',
    'AbsTransform',
    'AffineTransform',
    'ChainTransform',
    'CorrelationCholeskyTransform',
    'ExpTransform',
    'IndependentTransform',
    'PowerTransform',
    'ReshapeTransform',
    'PowerTransform',
    'SigmoidTransform',
    'SoftmaxTransform',
    'StackTransform',
    'StickBreakingTransform',
    'TanhTransform'
]


class Type(enum.Enum):
    """Mapping type of a transformation.
    """
    BIJECTION = 'bijection'  # bijective(injective and surjective)
    INJECTION = 'injection'  # injective-only
    SURJECTION = 'surjection'  # surjective-inly
    OTHER = 'other'  # general, neither injective nor surjective

    @classmethod
    def is_injective(cls, _type):
        """Both bijection and injection are injective mapping.
        """
        return _type in (cls.BIJECTION, cls.INJECTION)


class Transform(object):
    """Base class for the random varlable transformation.

    ``Transform`` can be used to represent any differentiable and injective 
    function from the subset of :math:`R^n` to subset of :math:`R^m`, generally 
    used for transforming a random sample generated by ``Distribution`` 
    instance. 

    Suppose :math:`X` is a K-dimensional random variable with probability 
    density function :math:`p_X(x)`. A new random variable :math:`Y = f(X)` may 
    be defined by transforming :math:`X` with a suitably well-behaved funciton 
    :math:`f`. It suffices for what follows to note that if f is one-to-one and 
    its inverse :math:`f^{-1}` have a well-defined Jacobian, then the density of 
    :math:`Y` is

    .. math::

        p_Y(y) = p_X(f^{-1}(y)) |det J_{f^{-1}}(y)|

    where det is the matrix determinant operation and :math:`J_{f^{-1}}(y)` is 
    the Jacobian matrix of :math:`f^{-1}` evaluated at :math:`y`.
    Taking :math:`x = f^{-1}(y)`, the Jacobian matrix is defined by

    .. math::

        J(y) = \begin{bmatrix}
        {\frac{\partial x_1}{\partial y_1}} &{\frac{\partial x_1}{\partial y_2}} 
        &{\cdots} &{\frac{\partial x_1}{\partial y_K}} \\
        {\frac{\partial x_2}{\partial y_1}}  &{\frac{\partial x_2}
        {\partial y_2}}&{\cdots} &{\frac{\partial x_2}{\partial y_K}} \\
        {\vdots} &{\vdots} &{\ddots} &{\vdots}\\
        {\frac{\partial x_K}{\partial y_1}} &{\frac{\partial x_K}{\partial y_2}} 
        &{\cdots} &{\frac{\partial x_K}{\partial y_K}} 
        \end{bmatrix}

    A ``Transform`` can be characterized by three operations:

        #. forward
           Forward implements :math:`x \rightarrow f(x)`, and is used to convert 
           one random outcome into another.
        #. inverse
           Undoes the transformation :math:`y \rightarrow f^{-1}(y)`.  
        #. log_det_jacobian
           The log of the absolute value of the determinant of the matrix of all
           first-order partial derivatives of the inverse function.

    Subclass typically implement follow methods:

        * _forward
        * _inverse
        * _forward_log_det_jacobian
        * _inverse_log_det_jacobian (optional)

    If the transform changes the shape of the input, you must also implemented:

        * _forward_shape
        * _inverse_shape

    Non-injective tansformation currently are not supported.

    Examples:

        .. code-block:: python

            class ExpTransform(Transform):

                _type = _Type.BIJECTION

                def __init__(self):
                    super(ExpTransform, self).__init__()

                def _forward(self, x):
                    return x.exp()

                def _inverse(self, y):
                    return y.log()

                def _forward_log_det_jacobian(self, x):
                    return x

                # Optional, this has been implemented in Base classs.
                def _inverse_log_det_jacobian(self, y):
                    return -self._forward_log_det_jacobian(self._inverse(y))

    """
    _type = Type.INJECTION

    def __init__(self):
        super(Transform, self).__init__()

    @classmethod
    def _is_injective(cls):
        """Is the transformation type one-to-one or not.

        Returns:
            bool: ``True`` denotes injective. ``False`` denotes non-injective.
        """
        return Type.is_injective(cls._type)

    def __call__(self, input):
        """Make this instance as a callable object. The return value is 
        depening on the input type. 

        * If the input is a ``Tensor`` instance, return 
          ``self.forward(input)`` .
        * If the input is a ``Distribution`` instance, return 
          ``TransformedDistribution(base=input, transforms=[self])`` .
        * If the input is a ``Transform`` instance, return 
          ``ChainTransform([self, input])`` .

        Args:
            input (Tensor|Distribution|Transform): The input value.

        Returns:
            [Tensor|TransformedDistribution|ChainTransform]: The return value.
        """
        if isinstance(input, distribution.Distribution):
            return transformed_distribution.TransformedDistribution(input, self)
        if isinstance(input, Transform):
            return ChainTransform([self, input])
        return self.forward(x)

    def forward(self, x):
        """Forward transformation with mapping :math:`y = f(x)`. 

        Useful for turning one random outcome into another.

        Args:
            x (Tensos): Input parameter, generally is a sample generated 
                from ``Distribution``.

        Returns:
            Tensor: Outcome of forward transform.
        """
        if not isinstance(x, paddle.Tensor):
            raise TypeError(
                f"Expected 'x' is a Tensor or Real, but got {type(x)}.")
        if x.dim() < self._domain.event_rank:
            raise ValueError(
                f'The dimensions of x({x.dim()}) should be '
                f'grater than or equal to {self._domain.event_rank}')
        return self._forward(x)

    def inverse(self, y):
        """Inverse transform :math:`x = f^{-1}(y)`. It's useful for 'reversing' 
        a transformation to compute one probability in terms of another.

        Args:
            y (Tensor): Input parameter for inverse transformation.

        Returns:
            Tensor: Outcome of inverse transform.
        """
        if not isinstance(y, paddle.Tensor):
            raise TypeError(
                f"Expected 'y' is a Tensor or Real, but got {type(y)}.")
        if y.dim() < self._codomain.event_rank:
            raise ValueError(
                f'The dimensions of y({y.dim()}) should be '
                f'grater than or equal to {self._codomain.event_rank}')
        return self._inverse(y)

    def forward_log_det_jacobian(self, x):
        """The log of the absolute value of the determinant of the matrix of all 
        first-order partial derivatives of the inverse function.

        Args:
            x (Tensor): Input tensor, generally is a sample generated from 
                ``Distribution``

        Returns:
            Tensor: The log of the absolute value of Jacobian determinant. 
        """
        if not isinstance(x, paddle.Tensor):
            raise TypeError(
                f"Expected 'y' is a Tensor or Real, but got {type(x)}.")
        if isinstance(x, paddle.Tensor) and x.dim() < self._domain.event_rank:
            raise ValueError(
                f'The dimensions of x({x.dim()}) should be '
                f'grater than or equal to {self._domain.event_rank}')
        if not self._is_injective():
            raise NotImplementedError(
                "forward_log_det_jacobian can't be implemented for non-injective"
                "transforms.")

        return self._call_forward_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y):
        """Compute :math:`log|det J_{f^{-1}}(y)|`.
        Note that ``forward_log_det_jacobian`` is the negative of this function, 
        evaluated at :math:`f^{-1}(y)`.

        Args:
            y (Tensor): The input to the 'inverse' Jacobian determinant 
                evaluation.

        Returns:
            Tensor: The value of :math:`log|det J_{f^{-1}}(y)|`.
        """
        if not isinstance(y, paddle.Tensor):
            raise TypeError(f"Expected 'y' is a Tensor, but got {type(y)}.")
        if y.dim() < self._codomain.event_rank:
            raise ValueError(
                f'The dimensions of y({y.dim()}) should be '
                f'grater than or equal to {self._codomain.event_rank}')
        return self._call_inverse_log_det_jacobian(y)

    def forward_shape(self, shape):
        """Infer the shape of forward transformation.

        Args:
            shape (Sequence[int]): The input shape.

        Returns:
            Sequence[int]: The output shape.
        """
        if not isinstance(shape, typing.Sequence):
            raise TypeError(
                f"Expected shape is Sequence[int] type, but got {type(shape)}.")
        return self._forward_shape(shape)

    def inverse_shape(self, shape):
        """Infer the shape of inverse transformation.

        Args:
            shape (Sequence[int]): The input shape of inverse transformation.

        Returns:
            Sequence[int]: The output shape of inverse transformation.
        """
        if not isinstance(shape, typing.Sequence):
            raise TypeError(
                f"Expected shape is Sequence[int] type, but got {type(shape)}.")
        return self._inverse_shape(shape)

    @property
    def _domain(self):
        """The domain of this transformation"""
        return variable.real

    @property
    def _codomain(self):
        """The codomain of this transformation"""
        return variable.real

    def _forward(self, x):
        """Inner method for publid API ``forward``, subclass should 
        overwrite this method for supporting forward transformation.
        """
        raise NotImplementedError('Forward not implemented')

    def _inverse(self, y):
        """Inner method of public API ``inverse``, subclass should 
        overwrite this method for supporting inverse transformation.
        """
        raise NotImplementedError('Inverse not implemented')

    def _call_forward_log_det_jacobian(self, x):
        """Inner method called by ``forward_log_det_jacobian``."""
        if hasattr(self, '_forward_log_det_jacobian'):
            return self._forward_log_det_jacobian(x)
        if hasattr(self, '_inverse_log_det_jacobian'):
            return -self._inverse_log_det_jacobian(self.forward(y))
        raise NotImplementedError(
            'Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian'
            'is implemented. One of them is required.')

    def _call_inverse_log_det_jacobian(self, y):
        """Inner method called by ``inverse_log_det_jacobian``"""
        if hasattr(self, '_inverse_log_det_jacobian'):
            return self._inverse_log_det_jacobian(y)
        if hasattr(self, '_forward_log_det_jacobian'):
            return -self._forward_log_det_jacobian(self._inverse(y))
        raise NotImplementedError(
            'Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian '
            'is implemented. One of them is required')

    def _forward_shape(self, shape):
        """Inner method called by ``forward_shape``, which is used to infer the 
        forward shape. Subclass should overwrite this method for supporting 
        ``forward_shape``.
        """
        return shape

    def _inverse_shape(self, shape):
        """Inner method called by ``inverse_shape``, whic is used to infer the 
        invese shape. Subclass should overwrite this method for supporting 
        ``inverse_shape``.
        """
        return shape


class AbsTransform(Transform):
    """Absolute transformation with formula :math:`y = f(x) = abs(x)`, 
    element-wise.

    This non-injective transformation allows for transformaitons of scalar 
    distributions with the absolute value function, which maps ``(-inf, inf)`` 
    to ``[0, inf)`` .

    * For ``y`` in ``(0, inf)`` , ``AbsTransform.inverse(y)`` returns the set invese 
      ``{x  in (-inf, inf) : |x| = y}`` as a tuple, ``-y, y`` .
    * For ``y`` equal ``0`` , ``AbsTransform.inverse(0)`` returns ``0, 0``, which is not 
      the set inverse (the set inverse is the singleton {0}), but "works" in 
      conjunction with ``TransformedDistribution`` to produce a left 
      semi-continuous pdf.
    * For ``y`` in ``(-inf, 0)`` , ``AbsTransform.inverse(y)`` returns the 
      wrong thing ``-y, y``. This is done for efficiency.

    Examples:

        .. code-block:: python

            import paddle

            abs = paddle.distribution.AbsTransform()

            print(abs.forward(paddle.to_tensor([-1., 0., 1.])))
            # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1., 0., 1.])

            print(abs.inverse(paddle.to_tensor(1.)))
            # (Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [-1.]), Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.]))

            # The |dX/dY| is constant 1. So Log|dX/dY| == 0
            print(abs.inverse_log_det_jacobian(paddle.to_tensor(1.)))
            # (Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        0.), Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        0.))

            #Special case handling of 0.
            print(abs.inverse(paddle.to_tensor(0.)))
            # (Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0.]), Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0.]))
            print(abs.inverse_log_det_jacobian(paddle.to_tensor(0.)))
            # (Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        0.), Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        0.))

    """
    _type = Type.SURJECTION

    def _forward(self, x):
        return x.abs()

    def _inverse(self, y):
        return -y, y

    def _inverse_log_det_jacobian(self, y):
        zero = paddle.zeros([], dtype=y.dtype)
        return zero, zero

    @property
    def _domain(self):
        return variable.real

    @property
    def _codomain(self):
        return variable.positive


class AffineTransform(Transform):
    """Affine transformation with mapping 
    :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor): The location parameter.
        scale (Tensor): The scale parameter.
    """
    _type = Type.BIJECTION

    def __init__(self, loc, scale):
        if not isinstance(loc, paddle.Tensor):
            raise TypeError(f"Expected 'loc' is a Tensor, but got {type(loc)}")
        if not isinstance(scale, paddle.Tensor):
            raise TypeError(
                f"Expected scale is a Tensor, but got {type(scale)}")
        self._loc = loc
        self._scale = scale
        super(AffineTransform, self).__init__()

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    def _forward(self, x):
        return self._loc + self._scale * x

    def _inverse(self, y):
        return (y - self._loc) / self._scale

    def _forward_log_det_jacobian(self, x):
        return paddle.abs(self._scale).log()

    def _forward_shape(self, shape):
        return paddle.broadcast_shape(
            paddle.broadcast_shape(shape, self._loc.shape), self._scale.shape)

    def _inverse_shape(self, shape):
        return paddle.broadcast_shape(
            paddle.broadcast_shape(shape, self._loc.shape), self._scale.shape)

    @property
    def _domain(self):
        return variable.real

    @property
    def _codomain(self):
        return variable.real


class ChainTransform(Transform):
    """Composes multiple transforms in a chain.

    Args:
        transforms (Sequence[int]): A sequence of transformations.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        super(Chain, self).__init__()

    def _is_injective(self):
        return all(t._is_injective() for t in self.transforms)

    def _forward(self, x):
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def _inverse(self, y):
        for transform in reversed(self.transforms):
            y = transform.inverse(y)

    def _forward_log_det_jacobian(self, x):
        value = 0
        event_rank = self._domain.event_rank
        for t in self.transforms:
            value += tool._sum_rightmost(
                t.forward_log_det_jacobian(x), event_rank - t.domain.event_rank)
            x = t.forward(x)
            event_rank += t.codomain.event_rank - t.domain.event_rank
        return value

    def _forward_shape(self, shape):
        for transform in self.transforms:
            shape = transform.forward_shape(shape)
        return shape

    def _inverse_shape(self, shape):
        for transform in self.transforms:
            shape = transform.inverse_shape(shape)
        return shape

    def _domain(self):
        domain = self.transforms[0].domain

        # Compute the lower bound of input dimensions for chain transform.
        #
        # Suppose the dimensions of input tensor is N, and chain [t0,...ti,...tm],
        # ti(in) denotes ti.domain.event_rank, ti(out) denotes ti.codomain.event_rank,
        # delta(ti) denotes (ti(out) - ti(in)).
        # For transform ti, N shoud satisfy the constraint:
        #   N + delta(t0) + delta(t1)...delta(t(i-1)) >= ti(in)
        # So, for all transform in chain, N shoud satisfy follow constraints:
        #   t0: N >= t0(in)
        #   t1: N >= t1(in) - delta(t0)
        #   ...
        #   tm: N >= tm(in) - ... - delta(ti) - ... - delta(t0)
        #
        # Above problem can be solved more effectively use dynamic programming.
        # Let N(i) denotes lower bound of transform ti, than the state
        # transition equation is:
        #   N(i) = max{N(i+1)-delta(ti), ti(in)}
        event_rank = self.transforms[-1].codomain.event_rank
        for t in reversed(self.transforms):
            event_rank -= t.codomain.event_rank - t.domain.event_rank
            event_rank = max(event_rank, t.domain.event_rank)

        return constraint.independent(domain, event_rank - domain.event_rank)

    def _codomain(self):
        codomain = self.transforms[-1].codomain

        event_rank = self.transforms[0].domain.event_rank
        for t in self.transforms:
            event_rank += t.codomain.event_rank - t.domain.event_rank
            event_rank = max(event_rank, t.codomain.event_rank)

        return constraint.independent(codomain,
                                      event_rank - codomain.event_rank)


class CorrelationCholeskyTransform(Transform):
    """[summary]

    Args:
        transform ([type]): [description]
    """

    # _domain = constraint.real_vector
    # _codomain = constraint.correlation_cholesky

    def _forward(self, x):
        pass

    def _inverse(self, y):
        pass

    def _forward_log_det_jacobian(self, x):
        pass

    def _forward_shape(self, x):
        pass

    def _inverse_shape(self, x):
        pass

    def _fill_triangular(self):
        """Creates a (batch of) lower triangular matrix from a vector in row 
        order
        """
        pass

    def _fill_triangular_inverse(self):
        """Creates a vector from a (batch of) lower triangular matrix.
        """
        pass


class ExpTransform(Transform):
    """Exponent transformation with mapping :math:`y = \exp(x)`.

    Exapmles:

        .. code-block:: python
    
            import paddle

            exp = paddle.distribution.ExpTransform()
            print(exp.forward(paddle.to_tensor([1., 2., 3.])))
            # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [2.71828175 , 7.38905621 , 20.08553696])

            print(exp.inverse(paddle.to_tensor([1., 2., 3.])))
            # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0.        , 0.69314718, 1.09861231])

            print(exp.forward_log_det_jacobian(paddle.to_tensor([1., 2., 3.])))
            # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1., 2., 3.])

            print(exp.inverse_log_det_jacobian(paddle.to_tensor([1., 2., 3.])))
            # Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [ 0.        , -0.69314718, -1.09861231])
    """
    _type = Type.BIJECTION

    def __init__(self):
        super(ExpTransform, self).__init__()

    @property
    def _domain(self):
        return variable.real

    @property
    def _codomain(self):
        return variable.positive

    def _forward(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def _forward_log_det_jacobian(self, x):
        return x


class IndependentTransform(Transform):
    """``IndependentTransform` wraps a base transformation, reinterprets 
    some of the rightmost batch axes as event axes.

    Generally, it is used to expand the event axes. This has no effect on the
    forward or inverse transformaion, but does sum out the 
    ``reinterpretd_bach_rank`` rightmost dimensions in computing the determinant 
    of Jacobian matrix.

    To see this, consider the ``ExpTransform`` applied to a Tensor which has 
    sample, batch, and event ``(S,B,E)`` shape semantics. Suppose the Tensor's 
    paritioned-shape is ``(S=[4], B=[2], E=[3,3])`` .

    Args:
        base (Transform): 
    """

    def __init__(self, base, reinterpreted_batch_rank):
        self._base = base
        self._reinterpreted_batch_rank = reinterpreted_batch_rank
        super(IndependentTransform, self).__init__()

    def _forward(self, x):
        if x.dim() < self._domain.event_rank:
            raise ValueError("Input dimensions is less than event dimensions.")
        return self._base.forward(x)

    def _inverse(self, y):
        if y.dim() < self._codomain.event_rank:
            raise ValueError("Input dimensions is less than event dimensions.")
        return self._base.inverse(x)

    def _forward_log_det_jacobian(self, x):
        return self.base.forward_log_det_jacobian(x).sum(
            list(range(-self.reinterpreted_batch_ndims, 0)))

    def _forward_shape(self, shape):
        return self.base.forward_shape(shape)

    def _inverse_shape(self, shape):
        return self.base.inverse_shape(shape)

    def _domain(self):
        return constraint.independent(self.base.domain,
                                      self.reinterpreted_batch_ndims)

    def _codomain(self):
        return constraint.indenpendent(self.base.codomain,
                                       self.reinterpreted_batch_ndims)


class PowerTransform(Transform):
    """Power transformation with mapping :math:`y = x^{\text{exponent}}`

    Args:
        power (Tensor): The power parameter.
    """
    _type = Type.BIJECTIVE

    def __init__(self, power):
        if not isinstance(power, paddle.Tensor):
            raise TypeError(
                f"Expected 'power' is a tensor, but got {type(pwoer)}")
        self._power = power
        super(PowerTransform, self).__init__()

    @property
    def power(self):
        return self._power

    @property
    def _domain(self):
        return variable.positive

    @property
    def _codomain(self):
        return variable.positive

    def _forward(self, x):
        return x.exp(self.power)

    def _inverse(self, y):
        return y.pow(1 / self.power)

    def _forward_log_det_jacobian(self, x):
        return (self.power * x.power(self.power - 1)).abs().log()


class ReshapeTransform(Transform):
    """Reshape event shape of a tensor.

    Args:
        in_event_shape(Sequence[int]): The input event shape.
        out_event_shape(Sequence[int]): The output event shape.
    """

    def __init__(self, in_event_shape, out_event_shape):
        if not isinstance(in_event_shape, typing.Sequence) or not isinstance(
                out_event_shape, typing.Sequence):
            raise TypeError(
                f"Expected type of 'in_event_shape' and 'out_event_shape' is "
                f"Squence[int], but got 'in_event_shape': {in_event_shape}, "
                f"'out_event_shape': {out_event_shape}")
        if math.prod(in_event_shape) != math.prod(out_event_shape):
            raise ValueError(
                f"The numel of 'in_event_shape' should be 'out_event_shape', "
                f"but got {math.prod(in_event_shape)}!={math.prod(out_event_shape)}"
            )

        self._in_event_shape = tuple(in_event_shape)
        self._out_event_shape = tuple(out_event_shape)
        super(ReshapeTransform, self).__init__()

    def _domain(self):
        return constraint.independent(constraint.real,
                                      len(self._in_event_shape))

    def _codomain(self):
        return constraint.independent(constraint.real,
                                      len(self._out_event_shape))

    def _forward(self, x):
        return x.reshape(
            tuple(x.shape)[:x.dim() - len(self._in_event_shape)] +
            self._out_event_shape)

    def _inverse(self, y):
        return y.reshape(
            tuple(y.shape)[:y.dim() - len(self._out_event_shape)] +
            self._in_event_shape)

    def _forward_shape(self, shape):
        if len(shape) < len(self._in_event_shape):
            raise ValueError(
                f"Expected length of 'shape' is not less than {len(self._in_event_shape)}, but got {len(shape)}"
            )
        if shape[-len(self._in_event_shape):] != self._in_event_shape:
            raise ValueError(
                f"Event shape mismatch, expected: {self._in_event_shape}, but got {shape[-len(self._in_event_shape):]}"
            )
        return tuple(shape[:-len(self._in_event_shape)]) + self._out_event_shape

    def _inverse_shape(self, shape):
        if len(shape) < len(self._out_event_shape):
            raise ValueError(
                f"Expected 'shape' length is not less than {len(self._out_event_shape)}, but got {len(shape)}"
            )
        if shape[-len(self._out_event_shape):] != self._out_event_shape:
            raise ValueError(
                f"Event shape mismatch, expected: {self._out_event_shape}, but got {shape[-len(self._out_event_shape):]}"
            )
        return tuple(shape[:-len(self._out_event_shape)]) + self._in_event_shape

    def _forward_log_det_jacobian(self, x):
        return paddle.zeros(
            x.shape[:x.dim() - len(self._in_event_shape)], dtype=x.dtype)


class SigmoidTransform(Transform):
    # _domain = constrain.real
    # _codomian = constraint.unit_interval

    def _forward(self, x):
        return paddle.sigmoid(x)

    def _inverse(self, y):
        return y.log() - (-y).log1p()

    def _forward_log_det_jacobian(self, x):
        return paddle.nn.functional.softplus(
            -x) - paddle.nn.functional.softplus(x)


class SoftmaxTransform(Transform):
    def _domain(self):
        return constraint.real_vector

    def _codomain(self):
        return constraint.simplex

    def _forward(self, x):
        x = (x - x.max(-1, True)[0]).exp()
        return x / x.sum(-1, True)

    def _inverse(self, y):
        return y.log()


class StackTransform(Transform):
    def __init__(self, transforms, axis=0):
        if not transforms or not isinstance(transforms, typing.Iterable):
            raise TypeError('transforms must be Iterable')
        if not all(isinstance(t, tansform.Transform) for t in transforms):
            raise TypeError('All element in transforms must be Transform Type.')

        self._transforms = transforms
        self._axis = axis

    def _forward(self, x):
        self._check_size(x)
        return paddle.stack([
            t.forward(v)
            for v, t in zip(paddle.unstack(x, self._axis), self._transforms)
        ], self._axis)

    def _inverse(self, y):
        self._check_size(y)
        return paddle.stack([
            t.inverse(v)
            for v, t in zip(paddle.unstack(y, self._axis), self._transforms)
        ], self._axis)

    def _forward_log_det_jacobian(self, x):
        self._check_size(x)
        return paddle.stack([
            t.forward_log_det_jacobian(v)
            for v, t in zip(paddle.unstack(x, self._axis), self._transforms)
        ], self._axis)

    def _check_size(self, v):
        if not (-v.dim() <= self._axis < v.dim()):
            raise ValueError(
                f'Input dimensions {v.dim()} should be grater than stack '
                f'transform axis {self._axis}.')
        if x.shape[self._axis] != len(self._transforms):
            raise ValueError(
                f'Input size along {self._axis} should be equal to the '
                f'length of transforms.')

    def _domain(self):
        return constraint.stack([t.domain for t in self._transforms],
                                self._axis)

    def _codomain(self):
        return constraint.stack([t.codomain for t in self._transforms],
                                self.axis)


class StickBreakingTransform(Transform):
    """Convert an unconstrained vector to the simplex with one additional 
    dimension by the stick-breaking construction.

    Args:
        transform ([type]): [description]

    Returns:
        [type]: [description]
    """

    # _domain = constraint.real_vector
    # _codomain = constraint.simplex

    def _forward(self, x):
        beta = (1 - F.sogmoid(x - paddle.arange(-x.shape[-1], 0, -1).log()))
        return F.pad(beta, [0, 1], value=1) * F.pad(beta.cumprod(-1), [1, 0],
                                                    value=1)

    def _inverse(self, y):
        return y[..., :-1].log() - (1 - y[..., :-1].cumsum(-1)
                                    ).log + paddle.arange(-y.shape[-1] - 1, 0,
                                                          -1).log()

    def _forward_log_det_jacobian(self, x):
        x = x - paddle.arange(-x.shape[-1], 0, -1)
        return (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)

    def _forward_shape(self, shape):
        if not shape:
            raise ValueError(f"Expected 'shape' is not empty, but got {shape}")
        return shape[:-1] + (shape[-1] + 1)

    def _inverse_shape(self, shape):
        if not shape:
            raise ValueError(f"Expected 'shape' is not empty, but got {shape}")
        return shape[:-1] + (shape[-1] - 1)


class TanhTransform(Transform):
    # _domain = constraint.real
    # _codomain = constraint.interval(-1.0, 1.0)

    def _forward(self, x):
        return x.tanh()

    def _inverse(self, y):
        return y.atanh()

    def _forward_log_det_jacobian(self, x):
        return 2. * (math.log(2.)) - x - paddle.nn.functional.softplus(-2. * x)
