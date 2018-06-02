#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "common.h"
#include "util.h"
#include "open_cl_computing.h"

#include "root_directory.h"

#define DTYPE cl_float

static DeviceData *ddata = NULL;
static FunctionData *fdata = NULL;

static bool
gpu_add(DTYPE *A,
	DTYPE *B,
	DTYPE *C,
	size_t size)
{
  // Create memory buffers on the device for each vector 
  cl_mem a_mem_obj = create_cl_mem_r(ddata, size*sizeof(DTYPE));
  cl_mem b_mem_obj = create_cl_mem_r(ddata, size*sizeof(DTYPE));
  cl_mem c_mem_obj = create_cl_mem_w(ddata, size*sizeof(DTYPE));

  // write data to input args
  memcpy_host_to_device(ddata, a_mem_obj, A, size*sizeof(DTYPE));
  memcpy_host_to_device(ddata, b_mem_obj, B, size*sizeof(DTYPE));
  execute_command_queue(ddata);

  // call function
  call_kernel_function(fdata, ddata, size,
		       KERNEL_ARGS(a_mem_obj, b_mem_obj, c_mem_obj));
  
  // copy result
  memcpy_device_to_host(ddata, c_mem_obj, C, size*sizeof(DTYPE));
  execute_command_queue(ddata);
  
  // free allocated resources
  cl_int ret = 0;
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseMemObject(c_mem_obj);
  return true;
}

static bool
cpu_add(DTYPE *A,
	DTYPE *B,
	DTYPE *C,
	size_t size) {
  for (size_t i = 0; i < size; ++i) {
    C[i] = sqrt(A[i]*A[i] + B[i]*B[i]);
  }
  return true;
}
 
int
main(int argc,
     char **argv)
{
  bool (*add)(DTYPE *, DTYPE *, DTYPE *, size_t) = NULL;
  char *device = NULL;
  bool use_opencl = false;
  size_t list_size = 0;
  for (int i = 1; i < argc; ++i) {
    char *arg = argv[i];
    
    if (strncmp(arg, "device=", 7) == 0) {
      device = arg + 7;
      if (strcmp(device, "gpu") == 0) {
	use_opencl = true;
	add = gpu_add;
      } else if (strcmp(device, "cpu") == 0) {
	add = cpu_add;
      }
    } else if (strncmp(arg, "size=", 5) == 0) {
      list_size = atoll(arg + 5);
    }
  }
  
  if (!add) {
    PRIINFO("usage: %s [device=cpu/gpu] [size=array-size]\n", argv[0]);
    return EXIT_FAILURE;
  }

  
  if (use_opencl) {
    ddata = set_up_cl_device();
    if (!ddata) {
      fprintf(stderr, "Fatal: no device data available. Exiting.\n");
      return EXIT_FAILURE;
    }
    fdata = set_up_cl_kernel_function(ddata, SYS_PATH("/res/vector_pythagoras_kernel.cl"), "vector_pythagoras");
    if (!fdata) {
      fprintf(stderr, "Fatal: no function data available. Exiting.\n");
      return EXIT_FAILURE;
    }
  }
  
  DTYPE *A = (DTYPE *) calloc(list_size, sizeof(DTYPE));
  DTYPE *B = (DTYPE *) calloc(list_size, sizeof(DTYPE));
  DTYPE *C = (DTYPE *) calloc(list_size, sizeof(DTYPE));
  for(size_t i = 0; i < list_size; i++) {
    A[i] = i;
    B[i] = list_size - i;
  }
  
  double t0 = get_time();
  add(A, B, C, list_size);
  double t1 = get_time();

  for(size_t i = 0; i < list_size; i++) {
    PRINRM("sqrt(sqr(%.3f) + sqr(%.3f)) = %.3f\n", A[i], B[i], C[i]);
  }
  
  PRIINFO("Adding length %ld array on the %s took %.3f s\n", list_size, device, t1-t0);

  if (use_opencl) {
    tear_down_kernel_function(fdata);
    tear_down_cl_device(ddata);
  }
  
  free(A);
  free(B);
  free(C);
  return EXIT_SUCCESS;
}
