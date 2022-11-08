import pycuda.driver as cuda
import pycuda.autoinit, pycuda.compiler
import numpy

a = numpy.random.randn(4,4).astype(numpy.float32)   # creates 4x4 array of random floats
a_dev = cuda.mem_alloc(a.nbytes)                    # allocates memory on device
cuda.memcpy_htod(a_dev, a)                          # copy data from host to device

# kernel code that gets sent to nvcc
# notice that this is really just c code;
# it's commented out so python interpreter doesn't see it

# doubles the value of every element in the matrix
mod = pycuda.compiler.SourceModule("""
__global__ void twice(float *a) {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
}
""")

func = mod.get_function("twice")    # alias the kernel function
func(a_dev, block=(4,4,1))          # invoke kernel code with a 4x4 block size

a_result = numpy.empty_like(a)      # creates an empty array w/ same dimensions as a
cuda.memcpy_dtoh(a_result, a_dev)   # copy data from device to host, into a_result

# print results
print("Input Matrix")
print(a)
print("Output Matrix")
print(a_result)
