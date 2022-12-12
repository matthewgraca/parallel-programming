#include "mpi.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <climits>

#define ARRAY_SIZE 1000000

int main (int argc,  char *argv[]) {

   int myid, numprocs;
   int namelen;
   int* numbers = new int[ARRAY_SIZE];
   char processor_name[MPI_MAX_PROCESSOR_NAME];

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Get_processor_name(processor_name, &namelen);
   
   srand(myid); 
 
   printf("Process %d on %s\n", myid, processor_name);
 
   for (int i=0; i<ARRAY_SIZE; i++)
      numbers[i] = rand();  //could be randomly generated

   int s = (int)floor(ARRAY_SIZE/numprocs);
   int s0 = s + ARRAY_SIZE%numprocs;

   int startIndex = s0 + (myid-1)*s;
   int endIndex = startIndex + s;

   double startwtime;
   if (myid == 0) {
      startwtime = MPI_Wtime();
   }

   int i;
   int part_min = INT_MAX;
   int part_max = INT_MIN;
   
   if (myid == 0) {
      // master worker - compute the master's partial min 
      for (i=0; i<s0; i++) {
         part_min = (numbers[i] < part_min) ? numbers[i] : part_min;
         part_max = (numbers[i] > part_max) ? numbers[i] : part_max;
      }
      printf("Process %d - startIndex 0 endIndex %d; part_min %ld part_max %ld\n",
             myid, s0-1, part_min, part_max);
   } else {
      //slave's work - compute partial min
      for (i= startIndex; i<endIndex; i++) {
         part_min = (numbers[i] < part_min) ? numbers[i] : part_min;
         part_max = (numbers[i] > part_max) ? numbers[i] : part_max;
      }
      printf ("Process %d - startIndex %d endIndex %d; part_min %ld part_max %ld\n",
              myid, startIndex, endIndex-1, part_min, part_max);
   }

   int min = part_min;
   int max = part_max;
   MPI_Reduce(&part_min, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&part_max, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

   if (myid == 0) {
      double runTime = MPI_Wtime() - startwtime;
      printf("Execution time (sec) = %f min = %ld max = %ld\n",
             runTime, min, max);
   }

   delete[] numbers;

   MPI_Finalize();
}

