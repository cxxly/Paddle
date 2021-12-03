# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

from paddle.distribution import Categorical, Distribution, Normal, Uniform

from beta import Beta
from dirichlet import Dirichlet

_REGISTER_TABLE = {}
_CACHE = {}


def kl_divergence(p, q):
    """Compute kl divergence between distribution p and q

    Args:
        p ([type]): [description]
        q ([type]): [description]
    """
    if (type(p), type(q)) in _CACHE:
        fun = _CACHE.get((type(p), type(q)))
    else:
        fun = _dispatch(type(p), type(q))
        _CACHE[type(p), type(q)] = fun

    if fun is NotImplemented:
        raise NotImplementedError

    return fun(p, q)


def _dispatch(type_p, type_q):

    matchs = []
    for super_p, super_q in _REGISTER_TABLE:
        if issubclass(type_p, super_p) and issubclass(type_q, super_q):
            matchs.append((super_p, super_q))

    if not matchs:
        return NotImplemented

    left_p, left_q = min(_Compare(*m) for m in matches).types
    right_p, right_q = min(_Compare(*reversed(m)) for m in matchs).types

    left_fun = _REGISTER_TABLE[left_p, left_q]
    right_fun = _REGISTER_TABLE[right_p, right_q]

    if left_fun is not right_fun:
        warnings.warn(
            'Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.
            format(type_p.__name__, type_q.__name__, left_p.__name__,
                   right_q.__name__), RuntimeWarning)

    return left_fun


@total_ordering
class _Compare(object):
    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def dispatch(type_p, type_q):
    if not issubclass(type_p, Distribution) or not issubclass(type_q,
                                                              Distribution):
        raise TypeError('type_p and type_q shoule be subclass of Distribution')

    def decorator(f):
        _REGISTER_TABLE[type_p, type_q] = f
        _CACHE.clear()
        return f

    return decorator


@dispatch(Beta, Beta)
def _kl_beta_beta(p, q):
    raise NotImplementedError


@dispatch(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    raise NotImplementedError


@dispatch(Normal, Normal)
def _kl_normal_normal(p, q):
    raise NotImplementedError


@dispatch(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    raise NotImplementedError
