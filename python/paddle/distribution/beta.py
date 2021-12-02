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

import paddle

import dirichlet
import exponential_family


class Beta(exponential_family.ExponentialFamily):
    """Beta distribution parameterized by

    Args:
        alpha (float|list|Tensor): alpha parameter of beta distribution
        beta (float|list|Tensor): beta parameter of beta distribution
    """

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta
        self._dirichlet = dirichlet.Dirichlet(paddle.stack([alpha, beta], -1))

        super(Beta, self).__init__(self._dirichlet._batch_shape,
                                   self._dirichlet._event_shape)

    @property
    def alpha(self):
        """Return alpha parameter of beta distribution.

        Returns:
            alpha parameter
        """
        return self._alpha

    @property
    def beta(self):
        """Return beta parameter of beta distribution.

        Returns:
            beta parameter
        """
        return self._beta

    @property
    def mean(self):
        """mean of beta distribution.
        """
        return self._alpha / (self._alpha + self._beta)

    @property
    def vairance(self):
        """variance of beat distribution
        """
        sum = self._alpha + self._beta
        return self._alpha * self._beta / (sum.pow(2) * (sum + 1))

    def prob(self, value):
        """probability density funciotn evaluated at value

        Args:
            value (Tensor): value to be evaluated
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """log probability density funciton evaluated at value

        Args:
            value (Tensor): value to be evaluated
        """
        return self._dirichlet.log_prob(paddle.stack([value, 1.0 - value], -1))

    def sample(self, shape=None):
        """sample from beta distribution with sample shape 

        Args:
            shape (Tensor): sample shape

        Returns:
            sampled data
        """
        return self._dirichlet.sample(shape).select(-1, 0)
