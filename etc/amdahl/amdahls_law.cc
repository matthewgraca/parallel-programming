#include "amdahls_law.h"
#include <thread>
#include <cmath>

constexpr int PROBLEM_SIZE = 1000000;
void operations(int);
void parallel_init(int*, int, int);
void parallelize(int);

// initializes array slice, as well as runs additional operations
void parallel_init(int* a, int beg, int end){
  for (int i = beg; i < end; i++){
    a[i] = i;
    operations(i);
  }
}

// additional operations to pad out runtime so tests aren't running at 1ms
void operations(int i){
  i = cos(i);
  i = sin(i);
  i = tan(i);
  i = exp(i);
  i = pow(i, i);
}

// splits our experiment into threads
void parallelize(int n){
  int a[PROBLEM_SIZE];

  // determine partitioning indices
  int threads = std::thread::hardware_concurrency();
  int remainder = n % threads;
  int slice = n / threads;
  int start_idx = 0;
  std::thread t[threads];

  // split the array's initialization to be parallelized, based on n
  for (int i = 0; i < threads; i++){
    if (i == threads-1)
      slice += remainder;
    t[i] = std::thread(parallel_init, std::ref(a), start_idx, start_idx+slice);
    start_idx += slice;
  }

  // join threads
  for (int i = 0; i < threads; i++){
    t[i].join();
  }

  // complete the rest of the array's initialization serially
  for (int i = n; i < PROBLEM_SIZE; i++){
    a[i] = i;
    operations(i);
  }
}

int no_parallel(){
  parallelize(0);
  return 1;
}

int quarter_parallel(){
  parallelize(PROBLEM_SIZE*0.25);
  return 1;
}

int half_parallel(){
  parallelize(PROBLEM_SIZE*0.5);
  return 1;
}

int three_quarters_parallel(){
  parallelize(PROBLEM_SIZE*0.75);
  return 1;
}

int full_parallel(){
  parallelize(PROBLEM_SIZE);
  return 1;
}
