This is a C++ example of using Message Passing Interface (MPI).
Each node splices the array into partitions, then finds the local minimum and maximum value for their partition.
Then, the local min and max are reduced, using built-in functionality, to find the global min and max values.
