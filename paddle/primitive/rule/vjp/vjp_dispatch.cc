// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <vector>

#include "paddle/fluid/ir/dialect/pd_api.h"
#include "paddle/ir/core/operation.h"
#include "paddle/primitive/backend/backend.h"
#include "paddle/primitive/rule/vjp/vjp_dispatch.h"
#include "paddle/primitive/type/desc_tensor.h"

namespace paddle {
namespace primitive {
namespace experimental {
paddle::optional<paddle::Tensor> tanh_vjp(
    const Tensor& out,
    const Tensor& grad_out,
    const std::vector<std::vector<int>>& stop_gradients) {
  // get tanh_grad res.
  Tensor op_res =
      backend::experimental::tanh_grad<primitive::experimental::DescTensor>(
          out, grad_out);

  // set op stop_gradient info
  // TODO(wanghao107): Replace with more generic code.
  // Support set stop_gradients for all ops.
  ir::Operation* grad_op =
      std::static_pointer_cast<primitive::experimental::DescTensor>(
          op_res.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>()
          .owner();
  uint32_t num_res = grad_op->num_results();
  std::vector<ir::Attribute> ir_stop_gradients(num_res);
  for (size_t i = 0; i < num_res; i++) {
    if (stop_gradients[0][i]) {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), true);
    } else {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), false);
    }
  }
  grad_op->set_attribute(
      "stop_gradient",
      ir::ArrayAttribute::get(ir::IrContext::Instance(), ir_stop_gradients));

  // construct vjp result by op result and stop_gradients info
  paddle::optional<paddle::Tensor> vjp_res;
  if (!stop_gradients[0][0]) {
    vjp_res = paddle::make_optional<paddle::Tensor>(op_res);
  }
  return vjp_res;
}
}  // namespace experimental
}  // namespace primitive
}  // namespace paddle
