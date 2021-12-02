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

from beta import Beta
from dirichlet import Dirichlet
from paddle.distribution import Uniform, Normal, Categorical, Distribution

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
    pass


@total_ordering
class _Compare(object):
    def __init__(self, *types):
        self._types = types

    def __eq__(self, other):
        return self._types == other.types

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
        return f

    return decorator


@dispatch(Beta, Beta)
def _kl_beta_beta(p, q):
    pass


@dispatch(Dirichlet, Dirichlet)
def _kl_beta_beta(p, q):
    pass


@dispatch(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    pass


@dispatch(Normal, Normal)
def _kl_normal_normal(p, q):
    pass


@dispatch(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    pass
