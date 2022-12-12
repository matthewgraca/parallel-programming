Simple python example of using Message Passing Interface (MPI) to communicate with different processors. 
Each node splices the data from an array into partitions, and finds the sum of the values in their partition.
The master and slave nodes are responsible for computing their local sums.
The partial sums are then reduced, combining into a global sum. 
