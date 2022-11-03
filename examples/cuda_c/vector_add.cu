#include <stdio.h>

#define N 10000

__global__ void vector_add(int *out, int *a, int *b, int n) {
   for (int i = 0; i < n; i++) {
      out[i] = a[i] + b[i];
   }
}

int main() {
   int a[N], b[N], out[N];
   int *d_a, *d_b, *d_out;

   for (int i = 0; i < N; i++) {
      a[i] = 0-i;
      b[i] = i * i;
   } 

   // allocate the memory on the GPU
   cudaMalloc((void**)&d_a, sizeof(int) * N);
   cudaMalloc((void**)&d_b, sizeof(int) * N);
   cudaMalloc((void**)&d_out, sizeof(int) * N); 

   // copy the arrays 'a' and 'b' to the GPU
   cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

   vector_add<<<N, 1>>>(d_out, d_a, d_b, N);

   // copy the array 'c' back from the GPU to the CPU
   cudaMemcpy(out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

   int i;
   for (i=0; i<N; ++i) {
      printf ("%d ", out[i]);
   }
   printf ("\n");

   // free the memory allocated on the GPU
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_out);
   return 0;
}

