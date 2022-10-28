#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <utility>
#include <vector>
#include <thread>
#include <string>

using namespace std;

/******************************************************************************
 * Declarations
 *****************************************************************************/
mutex countMutex;
bool inRange(double x, double y);
void countInRange(int& globalCount, int& partSize);
void validateInputs(int argc, char *argv[]);
bool isInt(char* args);

/******************************************************************************
 * Function definitions
 *****************************************************************************/
// thread function - counts number of random nums in the quarter unit circle
void countInRange(int& globalCount, int& partSize){
  int localCount = 0;

  for (int i = 0; i < partSize; i++){
    double x = ((double)rand()) / ((double)RAND_MAX);
    double y = ((double)rand()) / ((double)RAND_MAX);
    if (inRange(x, y))
      localCount++;
  }

  // lock and add to global count
  lock_guard<mutex> countLock(countMutex);
  globalCount += localCount;
}

// in quarter unit circle if x^2 + y^2 <= 1
bool inRange(double x, double y){
  return x * x + y * y <= 1;
}

// ensure inputs are correct
void validateInputs(int argc, char *argv[]){
  // ensure the number of args fit (2)
  if (argc != 3){
    cout << "two arguments required; "
         << "usage: monte [1...10] [10...1000000]" << endl;
    exit(0);
  }

  // ensure the 2 numbers are ints
  if (!isInt(argv[1]) || !isInt(argv[2])){
    cout << "arguments must be 32-bit integers; "
         << "usage: monte [1...10] [10...1000000]" << endl;
    exit(0);
  }

  // ensure the 2 ints are in range
  int a = stoi(argv[1]);
  int b = stoi(argv[2]);
  if ((a < 1 || a > 10) || (b < 10 || b > 1000000)){
    cout << "out of range; "
         << "usage: monte [1...10] [10...1000000]" << endl;
    exit(0);
  }
}

// returns true if given characters are integers, false if otherwise
bool isInt(char* args){
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
 *****************************************************************************/

int main(int argc, char *argv[]){
  // ensure command line arguments are valid
  validateInputs(argc, argv);
  int threads = stoi(argv[1]);  
  int points = stoi(argv[2]);   

  // determine partitions
  int partSize = points / threads;   
  int remainder = points % threads;

  // spawn threads and divide work
  int globalCount = 0;
  srand(time(0));
  thread t[threads];
  auto start = chrono::system_clock::now();
  for (int i = 0; i < threads; ++i) {
    // add points that don't partition to the last thread
    if (i == threads-1)
      partSize += remainder;
    t[i] = thread(countInRange, ref(globalCount), ref(partSize));
  }

  // join threads
  for (int i = 0; i < threads; ++i)
     t[i].join();
  chrono::duration<double> dur = chrono::system_clock::now() - start;

  // output data
  double estimate = 4 * (double)globalCount / points;
  double delta = abs(M_PI - estimate);
  printf("Time for computation %.6f\n", dur.count());
  printf("Result: %.4f\n", estimate);
  printf("Delta: %.4f\n", delta);
}
