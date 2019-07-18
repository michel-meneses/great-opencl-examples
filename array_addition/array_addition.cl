/**
 * This kernel function sums two arrays of integers and returns its result
 * through a third array.
 **/

 __kernel void sumArrays(__global int* a, __global int* b, __global int* c){
     int index = get_global_id(0);
     c[index] = a[index] + b[index];
 }