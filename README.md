# OpenCLDemo
A simple program that uses OpenCL to perform computations on the GPU.<br>
The current demo uses Pythagoras' theorem to compute the hypothenuse for a given number of generated triangles.

Requires an OpenCL 1.2 compatible GPU.

## Build
Any of
+ <code>$ make debug</code>
+ <code>$ make release</code>

## Run
+ Supports CPU and GPU computations.
+ Supports different sizes.
<br>
<code>$ ./build/OpenCLDemo device=[device] size=[size]</code>
