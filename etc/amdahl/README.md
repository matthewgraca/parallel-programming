## Goal
This was an attempt to test out the speedup factor of parallelized programs, using [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law), in the contest of the GoogleTest framework. You can play around with this using this [online calculator](https://www.omnicalculator.com/other/amdahls-law).

This was run using an i5-2540M; sporting 4 threads, thus a speedup factor of 4.

## Results

With our code run fully sequentially, we saw a runtime of 86ms.

At 25% parallelized code, we can expect a speedup multiplier of 1.2308. Thus our expected time is 69.87ms, which is in line with our result of 67ms.

At 50% parallelized code, we can expect a speedup multiplier of 1.6. Thus our expected time is 53.75ms, which is in line with our result of 55ms.

At 75% parallelized code, we can expect a speedup multiplier of 2.2857. Thus our expected time is 37.625ms. This is where we begin to see significant deviations, with our result displaying 43ms. For perspective, we would expect this number for 66% parallelized code.

At 100% parallelized code, we can expect a speedup multiplier of 4. Thus our expected time is 21.5ms. However, our result showed 31ms, a time more consistent with 85% parallelized code.

So next time you are working on a project with parallelized elements, keep in mind that Amdahl's Law is only a brief guide on expected speedup. The more parallelized the code, the more error you can expect. From these results, you can assume that 25%-50% parallelized code is roughly the same; but for 75%, assume the value given is in line with 66% and for 100%, assume it's more in line with 85% for your speedup gains.
