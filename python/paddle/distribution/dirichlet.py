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

import exponential_family
import paddle


class Dirichlet(exponential_family.ExponentialFamily):
    """Dirichlet distribution with parameter concentration

    Args:
        concentration (Tensor): concentration parameter of dirichlet 
        distribution
    """

    def __init__(self, concentration):
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one dimensional")

        self._concentration = concentration
        super(Dirichlet, self).__init__(concentration.shape[:-1],
                                        concentration.shape[-1:])

    @property
    def concentration(self):
        """Return concentration parameter of Dirichlet distribution.

        Returns:
            [Tensor]: parameter of Dirichlet distribution.
        """
        return self._concentration

    @property
    def mean(self):
        """mean of Dirichelt distribution.

        Returns:
            mean value of distribution.
        """
        return self._concentration / self._concentration.sum(-1, keepdim=True)

    @property
    def variance(self):
        """variance of Dirichlet distribution.

        Returns:
            variance value of distribution.
        """
        concentration0 = self._concentration.sum(-1, keepdim=True)
        return (self._concentration *
                (concentration0 - self._concentration)) / (
                    concentration0.pow(2) * (concentration0 + 1))

    def sample(self, shape=None):
        """[summary]

        Args:
            shape ([type], optional): [description]. Defaults to None.
        """
        raise NotImplementedError

    def prob(self, value):
        """Probability density function(pdf) evaluated at value.

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            pdf evaluated at value.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """log of probability densitiy function

        Args:
            value ([type]): [description]
        """
        return ((paddle.log(value) * (self._concentration - 1.0)
                 ).sum(-1) + paddle.lgamma(self._concentration.sum(-1)) -
                paddle.lgamma(self._concentration).sum(-1))

    def entropy(self):
        """entropy of Dirichlet distribution.

        Returns:
            entropy of distribution.
        """
        concentration0 = self._concentration.sum(-1)
        k = self._concentration.shape[-1]
        return (paddle.lgamma(self._concentration).sum(-1) -
                paddle.lgamma(concentration0) -
                (k - concentration0) * paddle.digamma(concentration0) - (
                    (self._concentration - 1.0
                     ) * paddle.digamma(self._concentration)).sum(-1))
