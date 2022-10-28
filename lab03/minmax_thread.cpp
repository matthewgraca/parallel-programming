#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <utility>
#include <vector>
#include <thread>
#include <climits>

using namespace std;

constexpr long long value= 1000000;   
mutex myMutex;

void minMax(long long& max, long long& min, const vector<int>& val, 
   unsigned long long beg, unsigned long long end){
   long long localMin = LLONG_MAX;
   long long localMax = LLONG_MIN;
    for (auto it= beg; it < end; ++it){
      long long value = val[it];
      if (localMin > value)
        localMin  = value;
      if (localMax < value)
        localMax = value;
    }
    lock_guard<mutex> myLock(myMutex);
    if (min > localMin)
      min = localMin;
    if (max < localMax)
      max = localMax;
}

int main(){

  cout << endl;

  vector<int> randValues;
  randValues.reserve(value);

  mt19937 engine (0);
  uniform_int_distribution<> uniformDist(INT_MIN,INT_MAX);
  for ( long long i=0 ; i< value ; ++i)
     randValues.push_back(uniformDist(engine));
 
  long long min = LLONG_MAX;
  long long max = LLONG_MIN;
  auto start = chrono::system_clock::now();

  int threads = 8;
  thread t[threads];
  long long slice = value / threads;
  int startIdx=0;
  for (int i = 0; i < threads; ++i) {
    cout << "Thread[" << i << "] - slice ["
         << startIdx << ":" << startIdx+slice-1 << "]" << endl;
    t[i] = thread(minMax, ref(max), ref(min), ref(randValues), startIdx, startIdx+slice-1);
    startIdx += slice;
  }

  for (int i = 0; i < threads; ++i)
     t[i].join();

  chrono::duration<double> dur= chrono::system_clock::now() - start;
  cout << "Time for minmax " << dur.count() << " seconds" << endl;
  cout << "Result: " << "min: " << min << " max: " << max << endl;

  cout << endl;
}
