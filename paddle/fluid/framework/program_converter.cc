// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/program_converter.h"

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace framework {

template <typename T>
std::vector<T> ExtractPlainVector(
    const std::vector<paddle::experimental::Scalar>& values) {
  std::vector<T> results;
  results.reserve(values.size());
  for (const auto& item : values) {
    results.push_back(item.to<T>());
  }
  return results;
}

template <typename T>
std::vector<paddle::experimental::Scalar> WrapAsScalars(
    const std::vector<T>& values) {
  std::vector<paddle::experimental::Scalar> results;
  results.reserve(values.size());
  for (const auto& item : values) {
    results.push_back(paddle::experimental::Scalar(item));
  }
  return results;
}

namespace no_scalar {
void ConvertSetValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values = PADDLE_GET_CONST(
      std::vector<paddle::experimental::Scalar>, op->GetAttr("values", false));
  op->RemoveAttr("values");
  op->SetAttr("bool_values", std::vector<int>());
  op->SetAttr("fp32_values", std::vector<float>());
  op->SetAttr("int32_values", std::vector<int>());
  op->SetAttr("int64_values", std::vector<int64_t>());
  op->SetAttr("fp64_values", std::vector<double>());
  op->SetAttr("fp16_values", std::vector<float>());

  paddle::experimental::DataType dtype =
      paddle::experimental::DataType::FLOAT32;
  if (values.size()) {
    dtype = values.at(0).dtype();
  }

  switch (dtype) {
    case paddle::experimental::DataType::BOOL:
      op->SetAttr("bool_values", ExtractPlainVector<int>(values));
      break;
    case paddle::experimental::DataType::FLOAT32:
      op->SetAttr("fp32_values", ExtractPlainVector<float>(values));
      break;
    case paddle::experimental::DataType::INT32:
      op->SetAttr("int32_values", ExtractPlainVector<int>(values));
      break;
    case paddle::experimental::DataType::INT64:
      op->SetAttr("int64_values", ExtractPlainVector<int64_t>(values));
      break;
    case paddle::experimental::DataType::FLOAT64:
      op->SetAttr("fp64_values", ExtractPlainVector<double>(values));
      break;
    case paddle::experimental::DataType::FLOAT16:
      op->SetAttr("fp16_values", ExtractPlainVector<float>(values));
      break;
    default:
      PD_THROW("Invalid data type `", dtype, "`.");
  }
}

void ConvertAssignValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values = PADDLE_GET_CONST(
      std::vector<paddle::experimental::Scalar>, op->GetAttr("values", false));
  op->RemoveAttr("values");
  op->SetAttr("bool_values", std::vector<int>());
  op->SetAttr("fp32_values", std::vector<float>());
  op->SetAttr("int32_values", std::vector<int>());
  op->SetAttr("int64_values", std::vector<int64_t>());

  paddle::experimental::DataType dtype =
      paddle::experimental::DataType::FLOAT32;
  if (values.size()) {
    dtype = values.at(0).dtype();
  }

  switch (dtype) {
    case paddle::experimental::DataType::BOOL:
      op->SetAttr("bool_values", ExtractPlainVector<int>(values));
      break;
    case paddle::experimental::DataType::FLOAT32:
    case paddle::experimental::DataType::FLOAT64:
      op->SetAttr("fp32_values", ExtractPlainVector<float>(values));
      break;
    case paddle::experimental::DataType::INT32:
      op->SetAttr("int32_values", ExtractPlainVector<int>(values));
      break;
    case paddle::experimental::DataType::INT64:
      op->SetAttr("int64_values", ExtractPlainVector<int64_t>(values));
      break;
    default:
      PD_THROW("Invalid data type `", dtype, "`.");
  }
}

void ConvertFillConstantOp(OpDesc* op) {
  paddle::experimental::Scalar value = PADDLE_GET_CONST(
      paddle::experimental::Scalar, op->GetAttr("value", false));
  op->RemoveAttr("value");

  paddle::experimental::DataType dtype = value.dtype();
  switch (dtype) {
    case paddle::experimental::DataType::BOOL:
    case paddle::experimental::DataType::INT32:
    case paddle::experimental::DataType::INT64:
    case paddle::experimental::DataType::FLOAT32:
    case paddle::experimental::DataType::FLOAT64:
    case paddle::experimental::DataType::FLOAT16:
      op->SetAttr("value", value.to<float>());
      op->SetAttr("str_value", value.ToRawString());
      break;
    default:
      PD_THROW("Cannot convert `", dtype, "` back to float.");
  }
}

void ConvertProgram(ProgramDesc* program) {
  // bool has_op_version = program->Proto()->has_op_version_map();
  // if (has_op_version) {
  //   const proto::OpVersionMap& op_versions =
  //   program->Proto()->op_version_map(); for (int i = 0; i <
  //   op_versions.pair_size(); i++) {
  //     VLOG(0) << "op name: " << op_versions.pair(i).op_name()
  //             << "\tversion: " << op_versions.pair(i).op_version().version();
  //   }
  // }

  const size_t num_blocks = program->Size();
  for (size_t i = 0; i < num_blocks; i++) {
    BlockDesc* block = program->MutableBlock(i);
    const size_t num_ops = block->OpSize();
    for (size_t j = 0; j < num_ops; j++) {
      OpDesc* op = block->Op(j);
      const std::string op_type = op->Type();
      if (op_type == "set_value") {
        ConvertSetValueOp(op);
      } else if (op_type == "fill_constant") {
        ConvertFillConstantOp(op);
      } else if (op_type == "assign_value") {
        ConvertAssignValueOp(op);
      }
    }
  }
}
}  // namespace no_scalar

namespace scalar {
void ConvertSetValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values;
  std::vector<int> bool_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values", false));
  std::vector<float> fp32_values =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values", false));
  std::vector<int> int32_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values", false));
  std::vector<int64_t> int64_values = PADDLE_GET_CONST(
      std::vector<int64_t>, op->GetAttr("int64_values", false));
  std::vector<double> fp64_values =
      PADDLE_GET_CONST(std::vector<double>, op->GetAttr("fp64_values", false));
  std::vector<float> fp16_values =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp16_values", false));

  if (bool_values.size()) {
    values = WrapAsScalars(bool_values);
  } else if (fp32_values.size()) {
    values = WrapAsScalars(fp32_values);
  } else if (int32_values.size()) {
    values = WrapAsScalars(int32_values);
  } else if (int64_values.size()) {
    values = WrapAsScalars(int64_values);
  } else if (fp64_values.size()) {
    values = WrapAsScalars(fp64_values);
  } else if (fp16_values.size()) {
    values = WrapAsScalars(fp16_values);
  }
  op->RemoveAttr("bool_values");
  op->RemoveAttr("fp32_values");
  op->RemoveAttr("int32_values");
  op->RemoveAttr("int64_values");
  op->RemoveAttr("fp64_values");
  op->RemoveAttr("fp16_values");
  op->SetAttr("values", values);
}

void ConvertAssignValueOp(OpDesc* op) {
  std::vector<paddle::experimental::Scalar> values;
  std::vector<int> bool_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values", false));
  std::vector<float> fp32_values =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values", false));
  std::vector<int> int32_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values", false));
  std::vector<int64_t> int64_values = PADDLE_GET_CONST(
      std::vector<int64_t>, op->GetAttr("int64_values", false));

  if (bool_values.size()) {
    values = WrapAsScalars(bool_values);
  } else if (fp32_values.size()) {
    values = WrapAsScalars(fp32_values);
  } else if (int32_values.size()) {
    values = WrapAsScalars(int32_values);
  } else if (int64_values.size()) {
    values = WrapAsScalars(int64_values);
  }
  op->RemoveAttr("bool_values");
  op->RemoveAttr("fp32_values");
  op->RemoveAttr("int32_values");
  op->RemoveAttr("int64_values");
  op->SetAttr("values", values);
}

void ConvertFillConstantOp(OpDesc* op) {
  float value = PADDLE_GET_CONST(float, op->GetAttr("value", false));
  std::string str_value =
      PADDLE_GET_CONST(std::string, op->GetAttr("str_value", false));

  paddle::experimental::Scalar scalar_value;
  if (!str_value.empty()) {
    scalar_value = paddle::experimental::Scalar(str_value);
  } else {
    scalar_value = paddle::experimental::Scalar(value);
  }

  op->RemoveAttr("value");
  op->RemoveAttr("str_value");
  op->SetAttr("value", scalar_value);
}

void ConvertProgram(ProgramDesc* program) {
  const size_t num_blocks = program->Size();
  for (size_t i = 0; i < num_blocks; i++) {
    BlockDesc* block = program->MutableBlock(i);
    const size_t num_ops = block->OpSize();
    for (size_t j = 0; j < num_ops; j++) {
      OpDesc* op = block->Op(j);
      const std::string op_type = op->Type();
      if (op_type == "set_value") {
        ConvertSetValueOp(op);
      } else if (op_type == "fill_constant") {
        ConvertFillConstantOp(op);
      } else if (op_type == "assign_value") {
        ConvertAssignValueOp(op);
      }
    }
  }
}
}  // namespace scalar
}  // namespace framework
}  // namespace paddle
