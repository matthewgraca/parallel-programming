import random
import threading
import time

lock = threading.Lock()
sum = 0

def partsum(count, id):
   local = 0
   for i in range(count):
      local = local + int(random.random()*100)

   with lock:
      global sum
      sum = sum + local

if __name__ == "__main__":
   size = 100000000  # Number of random numbers to add
   threads = 100   # Number of threads to create

   # Create a list of jobs and then iterate through
   # the number of threads appending each thread to
   # the job list
   jobs = []
   for i in range(0, threads):
      out_list = list()
      thread = threading.Thread(target=partsum(int(size/threads), i))
      jobs.append(thread)

   start_time = time.time()
   # Start the threads (i.e. calculate the random number lists)
   for j in jobs:
      j.start()

   # Ensure all of the threads have finished
   for j in jobs:
      j.join()

   print ("Final sum: %d" % sum)
   print ("Time for addition %s seconds" % (time.time() - start_time))
   print ("List processing complete.")