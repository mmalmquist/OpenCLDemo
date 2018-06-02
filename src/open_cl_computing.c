#include "open_cl_computing.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "common.h"
#include "util.h"

struct open_cl_device_data_t
{
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue;

  cl_uint num_compute_units;
  size_t num_work_groups;
  cl_uint num_work_dim;
  size_t *num_work_sizes;
};

struct open_cl_kernel_function_data_t
{
  cl_program program;
  cl_kernel kernel;
};


static bool
get_device(cl_device_id *device);

static char *
load_cl_kernel(char const *file_name,
	       size_t *src_size)
{
  FILE *fp = fopen(file_name, "r");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s': %s\n", file_name, strerror(errno));
    return NULL;
  }
  char *kernel_data = read_file(fp, src_size);
  fclose(fp);
  return kernel_data;
}

static bool
get_device(cl_device_id *device)
{
  cl_uint ret_num_platforms = 0;
  clGetPlatformIDs(0, NULL, &ret_num_platforms);
  if (ret_num_platforms == 0) {
    fprintf(stderr, "No platforms available\n");
    return false;
  }
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
    
  cl_uint ret_num_devices = 0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
  if (ret_num_devices == 0) {
    fprintf(stderr, "No devices available\n");
    return false;
  }
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
  return true;
}

extern DeviceData *
set_up_cl_device(void)
{
  DeviceData *device_data = (DeviceData *) calloc(1, sizeof(DeviceData));
  if (!get_device(&device_data->device)) {
    return NULL;
  }
  
  clGetDeviceInfo(device_data->device, CL_DEVICE_MAX_COMPUTE_UNITS,
		  sizeof(cl_uint), (void *) &device_data->num_compute_units, NULL);
  PRIDBG("%s: %u\n", "CL_DEVICE_MAX_COMPUTE_UNITS", device_data->num_compute_units);
  
  clGetDeviceInfo(device_data->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		  sizeof(size_t), (void *) &device_data->num_work_groups, NULL);
  PRIDBG("%s: %lu\n", "CL_DEVICE_MAX_WORK_GROUP_SIZE", device_data->num_work_groups);
  
  clGetDeviceInfo(device_data->device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
		  sizeof(size_t), (void *) &device_data->num_work_dim, NULL);
  PRIDBG("%s: %u\n", "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", device_data->num_work_dim);
  
  device_data->num_work_sizes = (size_t *) calloc(device_data->num_work_dim, sizeof(size_t));
  clGetDeviceInfo(device_data->device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
		  device_data->num_work_dim*sizeof(size_t), (void *) device_data->num_work_sizes, NULL);
  PRIDBG("%s:\n", "CL_DEVICE_MAX_WORK_ITEM_SIZES");
  for (cl_uint i = 0; i < device_data->num_work_dim; ++i) {
    PRIDBG("\t%lu\n", device_data->num_work_sizes[i]);
  }

  cl_int ret = 0;
  device_data->context = clCreateContext( NULL, 1, &device_data->device, NULL, NULL, &ret);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clCreateContext", ret);
  device_data->command_queue = clCreateCommandQueue(device_data->context, device_data->device, 0, &ret);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clCreateCommandQueue", ret);
  return device_data;
}

extern void
tear_down_cl_device(DeviceData *device_data)
{
  if (!device_data) { return; }
  
  cl_int ret = 0;
  ret = clReleaseCommandQueue(device_data->command_queue);
  ret = clReleaseContext(device_data->context);
  
  free(device_data->num_work_sizes);
  free(device_data);
}

extern FunctionData *
set_up_cl_kernel_function(DeviceData *device_data,
			  char const *file_path,
			  char const *function_name)
{
  FunctionData *function_data = (FunctionData *) calloc(1, sizeof(FunctionData));
  size_t source_size = 0;
  char *source_str = load_cl_kernel(file_path, &source_size);
  if (!source_str) {
    return NULL;
  }
  
  cl_int ret = 0;
  function_data->program = clCreateProgramWithSource(device_data->context, 1,
						     (const char **)&source_str,
						     (const size_t *)&source_size, &ret);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clCreateProgramWithSource", ret);
 
  ret = clBuildProgram(function_data->program, 1, &device_data->device, NULL, NULL, NULL);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clBuildProgram", ret);
 
  function_data->kernel = clCreateKernel(function_data->program, function_name, &ret);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clCreateKernel", ret);
  
  free(source_str);
  return function_data;
}

extern void
tear_down_kernel_function(FunctionData *function_data)
{
  if (!function_data) { return; }
  
  cl_int ret = 0;
  ret = clReleaseKernel(function_data->kernel);
  ret = clReleaseProgram(function_data->program);
  free(function_data);
}

extern cl_mem
create_cl_mem(DeviceData *ddata,
	      cl_mem_flags flags,
	      size_t size)
{
  cl_int ret = 0;
  cl_mem m = clCreateBuffer(ddata->context, flags, size, NULL, &ret);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clCreateBuffer", ret);
  return m;
}

extern void
memcpy_host_to_device(DeviceData *ddata,
		      cl_mem device_obj,
		      void const *host_obj,
		      size_t size)
{
  cl_int ret = 0;
  ret = clEnqueueWriteBuffer(ddata->command_queue, device_obj, CL_TRUE, 0, size, host_obj, 0, NULL, NULL);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clEnqueueWriteBuffer", ret);
}

extern void
memcpy_device_to_host(DeviceData *ddata,
		      cl_mem device_obj,
		      void *host_obj,
		      size_t size)
{
  cl_int ret = 0;
  ret = clEnqueueReadBuffer(ddata->command_queue, device_obj, CL_TRUE, 0, size, host_obj, 0, NULL, NULL);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clEnqueueReadBuffer", ret);
}

extern void
set_kernel_arg(FunctionData *fdata,
	       cl_uint arg_index,
	       cl_mem mem_obj)
{
  cl_int ret = 0;
  ret = clSetKernelArg(fdata->kernel, arg_index, sizeof(cl_mem), (void const *)&mem_obj);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clSetKernelArg", ret);
}

extern void
call_kernel(FunctionData *fdata,
	   DeviceData *ddata,
	   size_t array_length)
{
  cl_int ret = 0;
  
  // work must be powers of 2.
  size_t global_work_offset = 0;
  for (size_t mask = 1; mask <= array_length; mask <<= 1) {
    size_t global_work_size = array_length & mask;
    if (global_work_size > 0) {
      size_t local_work_size = MIN(global_work_size, ddata->num_work_groups);
      //size_t local_work_size = MIN(global_work_size, 64);
      ret = clEnqueueNDRangeKernel(ddata->command_queue,
				   fdata->kernel, 1,
				   &global_work_offset,
				   &global_work_size,
				   &local_work_size,
				   0, NULL, NULL);
      PRIDBG_WHEN(ret != 0, "%s: %d\n", "clEnqueueNDRangeKernel", ret);
      
      global_work_offset += global_work_size;
    }
  }
}

extern void
execute_command_queue(DeviceData *ddata)
{
  cl_int ret = 0;
  ret = clFlush(ddata->command_queue);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clFlush", ret);
  ret = clFinish(ddata->command_queue);
  PRIDBG_WHEN(ret != 0, "%s: %d\n", "clFinish", ret);
}

extern void
call_kernel_function(FunctionData *fdata,
		     DeviceData *ddata,
		     size_t array_length,
		     ...)
{
  va_list valist;
  va_start(valist, array_length);

  int i = 0;
  for (cl_mem mem_obj; (mem_obj = va_arg(valist, cl_mem)) != NULL; ++i) {
    set_kernel_arg(fdata, i, mem_obj);
  }
  va_end(valist);

  call_kernel(fdata, ddata, array_length);
  execute_command_queue(ddata);
}
