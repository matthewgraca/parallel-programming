#include "mpi.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define NUM_OF_TRIALS 5

// in quarter unit circle if x^2 + y^2 <= 1
bool in_range(double x, double y){
  return x * x + y * y <= 1;
}

int main (int argc, char *argv[]){
  int ARRAY_SIZE = 100;
  double x, y;

  // initialize mpi
  int myid, numprocs;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processor_name, &namelen);

  // rng seed = myid for testing consistency
  srand((unsigned) (myid));

  // run trials, each with larger number of points
  for (int j = 0; j < NUM_OF_TRIALS; j++, ARRAY_SIZE *= 10){
    // number of points each node is responsible for
    int part_size = (int)floor(ARRAY_SIZE / numprocs);

    // start counting execution time
    double startwtime;
    if (myid == 0) {
      startwtime = MPI_Wtime();
    }

    // master/slave node work
    // find number of points in the quarter unit circle
    int part_count = 0;
    if (myid == 0){
      // master work
      for (int i = 0; i < part_size; i++){
        x = (double)rand() / RAND_MAX;  
        y = (double)rand() / RAND_MAX;  
        if (in_range(x, y))
          part_count++;
      }
    }
    else{
      // slave work
      for (int i = 0; i < part_size; i++){
        x = (double)rand() / RAND_MAX;  
        y = (double)rand() / RAND_MAX;  
        if (in_range(x, y))
          part_count++;
      }
    }

    // collect data from all processes
    int total_count = part_count;
    MPI_Reduce(&part_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // get process execution time
    if (myid == 0){
      double estimate = 4 * (double)total_count / ARRAY_SIZE;
      double delta = std::abs(estimate - M_PI);
      double runTime = MPI_Wtime() - startwtime;
      printf("Execution time (sec) = %f "
          "total_count = %-6d "
          "total points = %-7d " 
          "estimation = %.4f "
          "delta = %.4f\n", 
          runTime, total_count, ARRAY_SIZE, estimate, delta);
    }
  }
  MPI_Finalize();
}
