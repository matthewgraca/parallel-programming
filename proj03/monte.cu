#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>

using namespace std;

/******************************************************************************
  * kernel function defintions
  ****************************************************************************/
// initializes random states for rng
__global__ void random_init(curandState *states){
  int index = blockIdx.x * blockDim.x + threadIdx.x ;
  int seed = 1337;
  // gives each thread its own random state
  curand_init(seed, index, 0, &states[index]);
}

// checks if two floats are in the range of a quarter unit circle
__device__ bool in_range(float x, float y){
  return x * x + y * y <= 1;
}

// counts the number of randomly generate numbers in range
__global__ void count_in_range(bool *d_a, const int n, curandState *states){
  int index = blockIdx.x * blockDim.x + threadIdx.x ;
  if (index < n){
    float x = curand_uniform(&states[index]);
    float y = curand_uniform(&states[index]);
    d_a[index] = (in_range(x, y)) ? 1 : 0;
  }
}


/******************************************************************************
  * host function defintions
  ****************************************************************************/
void validate_inputs(int argc, char *argv[]);
bool is_int(char* args);

// host function definitions
// ensure inputs are correct
void validate_inputs(int argc, char *argv[]){
  string usage = "usage: monte [10...1000000]";
  // ensure the number of args fit (2)
  if (argc != 2){
    cout << "one argument required; " << usage << endl;
    exit(0);
  }

  // ensure the number is an int
  if (!is_int(argv[1])){
    cout << "arguments must be 32-bit integers; " << usage << endl;
    exit(0);
  }

  // ensure the int is in range
  int a = stoi(argv[1]);
  if (a < 10 || a > 1000000){
    cout << "out of range; " << usage << endl;
    exit(0);
  }
}

// returns true if given characters are integers, false if otherwise
bool is_int(char* args){
  try{
    stoi(args);
  }
  catch (...){
    return false;
  }
  return true;
}

/******************************************************************************
  * main 
  ****************************************************************************/
int main(int argc, char* argv[]){
  // ensure command line args are valid, then determine partitions
  // how many threads should be allow user to pick?
  validate_inputs(argc, argv);
  int points = stoi(argv[1]);

  // allocate host memory 
  bool host_cnt[points];
  
  // device arrays and rng for gpu
  bool *device_cnt;
  curandState *device_rand;
  auto start = chrono::system_clock::now();

  // allocate device memory
  size_t device_cnt_size = points * sizeof(bool);
  cudaMalloc((void **) &device_rand, points*sizeof(curandState));
  cudaMalloc((void **) &device_cnt, device_cnt_size);

  // copy host data (not necessary)
  // allocate threads and blocks on gpu 
  int threads = 1024;
  int blocks = ceil(points / (float)threads);
  dim3 dimBlock(threads);
  dim3 dimGrid(blocks);

  // run kernel functions
  random_init<<<dimGrid, dimBlock>>>(device_rand);
  count_in_range<<<dimGrid, dimBlock>>>(device_cnt, points, device_rand);

  // copy result from device to host
  cudaMemcpy(host_cnt, device_cnt, device_cnt_size, cudaMemcpyDeviceToHost);
  chrono::duration<double> dur = chrono::system_clock::now() - start;

  // count the number of points in the range
  int count = 0;
  for (int i = 0; i < points; i++){
    count += (host_cnt[i]) ? 1 : 0;
  }

  // print results
  double estimate = 4 * (double)count / points;
  double delta = abs(M_PI - estimate);
  printf("Time for computation %.6f\n", dur.count());
  printf("Result: %.4f\n", estimate);
  printf("Delta: %.4f\n", delta);

  // free device space
  cudaFree(device_cnt);
  cudaFree(device_rand);
  return 0;
}
