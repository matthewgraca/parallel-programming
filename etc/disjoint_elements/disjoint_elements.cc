#include "disjoint_elements.h"
#include <unordered_map>
#include <thread>
#include <mutex>

std::mutex naive_mutex;
std::mutex parallel_mutex;

/******************************************************************************
 * sequential count
 *****************************************************************************/

// counts the number of disjoint elements between two int arrays, same size
int sequential_count(int* a, int* b, int n){
  int count = 0;
  // store b into a hashmap
  std::unordered_map<int, int> umap;
  for (int i = 0; i < n; i++){
    umap.insert({b[i], i});
  }
  // count the number of elements in a that are contained in b
  for (int i = 0; i < n; i++){
    if (!umap.contains(a[i]))
      count += 2; // (1)
  }
  return count;
}
/* (1) we count twice since if a doesn't appear in b, 
 b also doesn't appear in a if both sets are the same size 
 (otherwise, we'd store both a and b in separate hashmaps 
 and count elements of a not in b, then elements in b not in a)
*/

/******************************************************************************
 * parallel count
 *****************************************************************************/

// threaded function
void parallel_threaded_count(int& count, 
    int* a, const std::unordered_map<int, int>& umap,
    int beg, int end){
  int local_count = 0;
  for (int i = beg; i < end; i++){
    if (!umap.contains(a[i]))
      local_count += 2;
  }
  std::lock_guard<std::mutex> lock(parallel_mutex);
  count += local_count;
}

// uses threads to count disjoint elements
int parallel_count(int* a, int* b, int n){
  int count = 0;
  // store b into a hashmap
  std::unordered_map<int, int> umap;
  for (int i = 0; i < n; i++){
    umap.insert({b[i], i});
  }

  // determine partitioning indices
  int threads = 8;
  std::thread t[threads];
  int slice = n / threads;
  int remainder = n % threads;
  int start_idx = 0;

  // if n <= threads, just run the sequential version
  if (n <= threads){
    count = sequential_count(a, b, n);
  }
  else{
    // initialize threads
    for (int i = 0; i < threads; i++){
      if (i == threads-1)
        slice += remainder;
      t[i] = std::thread(parallel_threaded_count, 
          std::ref(count), std::ref(a), std::ref(umap), 
          start_idx, start_idx+slice);
      start_idx += slice;
    }

    // join threads
    for (int i = 0; i < threads; i++){
      t[i].join();
    }
  }

  return count;
}

/******************************************************************************
 * naive parallel count
 *****************************************************************************/

// threaded version of naive count
void naive_threaded_count(int& count, int* a, int* b, 
    int beg, int end, int n){
  int local_count = 0;
  for (int i = beg; i < end; i++){
    for (int j = 0; j < n; j++){
      if (a[i] == b[j]){
        local_count += 2;
        j = n;
      }
    }
  }
  // local sum := number of non-disjoint elements
  // sum -= local sum removes all non-disjoint elements from sum
  std::lock_guard<std::mutex> lock(naive_mutex);
  count -= local_count;
}

// naive solution to counting disjoint elements
int naive_parallel_count(int* a, int* b, int n){
  // determine partitioning indices
  int threads = 8;
  std::thread t[threads];
  int slice = n / threads;
  int remainder = n % threads;
  int start_idx = 0;
  int count = n * 2;  // let all elements be disjoint

  // if n <= threads, just run the sequential version
  if (n <= threads){
    count = sequential_count(a, b, n);
  }
  else{
    // initialize threads
    for (int i = 0; i < threads; i++){
      if (i == threads-1)
        slice += remainder;
      t[i] = std::thread(naive_threaded_count, 
          std::ref(count), std::ref(a), std::ref(b), 
          start_idx, start_idx+slice, n);
      start_idx += slice;
    }

    // join threads
    for (int i = 0; i < threads; i++){
      t[i].join();
    }
  }
  return count;
}
