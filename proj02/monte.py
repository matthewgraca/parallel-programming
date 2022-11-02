import threading
import random
import time
import math
 
mutex = threading.Lock()
globalCount = 0

# in quarter unit circle if x^2+y^2<=1
def inRange(x, y):
    return x * x + y * y <= 1
 
# thread function - counts number of random nums in the quarter unit circle
def countInRange(partSize):
    localCount = 0
    for i in range(partSize):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        if inRange(x,y):
            localCount += 1

    # lock and add to global count
    mutex.acquire()
    global globalCount
    globalCount += localCount
    mutex.release()
 
if __name__ =="__main__":
    # TODO: validate inputs
    threads = 2
    points = 100000

    # determine partitions
    partSize = int(points / threads)
    remainder = points % threads

    # spawn threads and divide work
    random.seed()
    start = time.time()
    threadList = []
    for i in range(threads):
        t = threading.Thread(target=countInRange, args=(partSize,))
        threadList.append(t)
        t.start()

    # join threads
    for i in range(threads):
        threadList[i].join()
    dur = time.time() - start

    # output data
    estimate = 4 * float(globalCount) / points
    delta = abs(math.pi - estimate)
    print("Time for computation; %f\n", dur)
    print("Result: %f\n", estimate)
    print("Delta: %f\n", delta)
