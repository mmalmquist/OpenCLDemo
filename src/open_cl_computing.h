#ifndef OPEN_CL_COMPUTING_H_
#define OPEN_CL_COMPUTING_H_


#include <stdarg.h> 
#include <stdbool.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct open_cl_device_data_t DeviceData;
typedef struct open_cl_kernel_function_data_t FunctionData;

extern DeviceData *
set_up_cl_device(void);
extern void
tear_down_cl_device(DeviceData *device_data);
extern FunctionData *
set_up_cl_kernel_function(DeviceData *device_data,
			  char const *file_path,
			  char const *function_name);
extern void
tear_down_kernel_function(FunctionData *function_data);

extern cl_mem
create_cl_mem(DeviceData *ddata,
	      cl_mem_flags flags,
	      size_t size);

#define create_cl_mem_r(ddata, size) create_cl_mem(ddata, CL_MEM_READ_ONLY, size)
#define create_cl_mem_w(ddata, size) create_cl_mem(ddata, CL_MEM_WRITE_ONLY, size)
#define create_cl_mem_rw(ddata, size) create_cl_mem(ddata, CL_MEM_READ_WRITE, size)

extern void
memcpy_host_to_device(DeviceData *ddata,
		      cl_mem device_obj,
		      void const *host_obj,
		      size_t size);

extern void
memcpy_device_to_host(DeviceData *ddata,
		      cl_mem device_obj,
		      void *host_obj,
		      size_t size);

extern void
set_kernel_arg(FunctionData *fdata,
	       cl_uint arg_index,
	       cl_mem mem_obj);

extern void
call_kernel(FunctionData *fdata,
	    DeviceData *ddata,
	    size_t array_length);

extern void
execute_command_queue(DeviceData *ddata);

extern void
call_kernel_function(FunctionData *fdata,
		     DeviceData *ddata,
		     size_t array_length,
		     ...);

#define KERNEL_ARGS(...) __VA_ARGS__, NULL

#endif
