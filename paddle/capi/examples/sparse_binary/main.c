#include <paddle/capi.h>
#include <time.h>
#include "../common/common.h"

#define CONFIG_BIN "./trainer_config.bin"

int main() {
  // Initalize Paddle
  char* argv[] = {"--use_gpu=False"};
  CHECK(paddle_init(1, (char**)argv));

  // Reading config binary file. It is generated by `convert_protobin.sh`
  long size;
  void* buf = read_config(CONFIG_BIN, &size);

  // Create a gradient machine for inference.
  paddle_gradient_machine machine;
  CHECK(paddle_gradient_machine_create_for_inference(&machine, buf, (int)size));
  CHECK(paddle_gradient_machine_randomize_param(machine));

  // Loading parameter. Uncomment the following line and change the directory.
  // CHECK(paddle_gradient_machine_load_parameter_from_disk(machine,
  //                                                "./some_where_to_params"));
  paddle_arguments in_args = paddle_arguments_create_none();

  // There is only one input of this network.
  CHECK(paddle_arguments_resize(in_args, 1));

  // Create input matrix.
  paddle_matrix mat = paddle_matrix_create_sparse(1, 784, 3, true, false);
  srand(time(0));
  paddle_real* array;
  int colBuf[] = {9, 93, 109};
  int rowBuf[] = {0, sizeof(colBuf) / sizeof(int)};

  CHECK(paddle_matrix_sparse_copy_from(mat,
                                       rowBuf,
                                       sizeof(rowBuf) / sizeof(int),
                                       colBuf,
                                       sizeof(colBuf) / sizeof(int),
                                       NULL,
                                       0));

  CHECK(paddle_arguments_set_value(in_args, 0, mat));

  paddle_arguments out_args = paddle_arguments_create_none();
  CHECK(paddle_gradient_machine_forward(machine,
                                        in_args,
                                        out_args,
                                        /* isTrain */ false));
  paddle_matrix prob = paddle_matrix_create_none();

  CHECK(paddle_arguments_value(out_args, 0, prob));

  CHECK(paddle_matrix_get_row(prob, 0, &array));

  printf("Prob: ");
  for (int i = 0; i < 10; ++i) {
    printf("%.2f ", array[i]);
  }
  printf("\n");

  return 0;
}
