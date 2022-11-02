import threading
import random
import time
import math
import sys
import getopt
 
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

def validateInputs(argv):
    usage = "Usage: python monte.py [1...] [10...1000000]"
    # ensure num of args is 2
    if len(argv) != 2:
        sys.exit("Two arguments required. " + usage)
    # ensure two args are ints
    try:
        threads = int(argv[0])
        points = int(argv[1])
    except:
        sys.exit("Arguments must be integers. " + usage)
    # ensure two int args are w/in range
    if not 1 <= threads <= 10 or not 10 <= points <= 1000000:
        sys.exit("Arguments must be in range. " + usage)
    return points, threads

if __name__ =="__main__":
    # validate inputs
    argv = sys.argv[1:]
    points, threads = validateInputs(argv)

    # determine partitions
    partSize = int(points / threads)
    remainder = points % threads

    # spawn threads and divide work
    random.seed()
    start = time.time()
    threadList = []
    for i in range(threads):
        # ensure unevenly divided part gets given to last thread
        if i == threads-1:
            partSize += remainder
        # create and start threads
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
    print("Time for computation: %.6f seconds" % dur)
    print("Result: %.4f" % estimate)
    print("Delta: %.4f" % delta)
