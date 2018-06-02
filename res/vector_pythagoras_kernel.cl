
#define DTYPE float

DTYPE
sqr(DTYPE x)
{
  return x * x;
}

__kernel void
vector_pythagoras(__global __read_only DTYPE const *A,
		  __global __read_only DTYPE const *B,
		  __global __write_only DTYPE *C)
{
  int i = get_global_id(0);

  C[i] = sqrt(sqr(A[i]) + sqr(B[i]));
}
