# TimeHistogram

High-performance estimation of percentiles of large time series data, varying over time. Freely available under the MIT license.

Suppose you are looking for trends in a very large time series of data. With many data points, it is difficult to understand the behavior at a glance. For example, here is a dataset of 50,000 (x, y) points, plotted as a line graph.

<img src="https://raw.githubusercontent.com/mattiverson/TimeHistogram/master/figures/InitialData.png" width="80%"></img>

There are some clear changes in behavior at x=20 and x=80, but it's hard to see anything other than the min and max values.
For example, it's not obvious whether the variance of the data increased or decreased after x=20; the minima and maxima extend further than they do for x < 20, but there are fewer data points reaching those extreme values.

This is the motivation for TimeHistogram: we can use it to calculate and plot percentiles of this time series. Here are its estimates of the 10th, 25th, 50th, 75th, and 90th percentile:

<img src="https://raw.githubusercontent.com/mattiverson/TimeHistogram/master/figures/ComputedQuantiles.png" width="80%"></img>

Now we can see a much more clear picture of the data's behavior over time. It is now obvious that variance decreased at x=20. The oscillations from x=40 to x=60, which were easy to miss before, are now obvious.
We can even go a step further, and make specific quantitative claims -- the interquartile range was steady around 0.52 for x < 20, before reducing to around 0.27 for 20 < x < 40.

Of course this is only useful if it's actually true. In this example, the original data was produced from some simple distributions:

```Python
import numpy as np
np.random.seed(12345)
x1 = np.linspace(0, 20, 10000, endpoint=False)
y1 = np.random.uniform(1, 2, 10000)
x2 = np.linspace(20, 40, 10000, endpoint=False)
y2 = np.random.normal(1.5, 0.2, 10000)
x3 = np.linspace(40, 60, 10000, endpoint=False)
y3 = np.random.normal(1.5 + 0.1 * np.sin(0.2*np.pi*x3), 0.2, 10000)
x4 = np.linspace(60, 80, 10000, endpoint=False)
y4 = np.random.normal(1.5, 0.2 + 0.05 * np.sin(0.2*np.pi*x3), 10000)
x5 = np.linspace(80, 100, 10000, endpoint=False)
y5 = np.random.uniform(1 - (x5-80) * 0.01, 2 + (x5-80) * 0.01, 10000)
```

From which, we can compute the exact theoretical quantiles of the data:
```Python
from scipy.special import erfinv
def invnorm(x):
    return np.sqrt(2) * erfinv(2*x-1)

one = np.ones_like(x1)
q = np.array((0.9, 0.75, 0.5, 0.25, 0.1)).reshape(5, 1)
ty1 = one + q
ty2 = 1.5 * one + 0.2*invnorm(q)
ty3 = 1.5 + 0.1 * np.sin(0.2*np.pi*x3) + 0.2*invnorm(q)
ty4 = 1.5 + (0.2 + 0.05 * np.sin(0.2*np.pi*x3))*invnorm(q)
ty5 = 1.5 + (0.5 + 0.01*(x5-80))*(2*q-1)
```

So let's see how well our estimates, computed from the randomly-generated data, were able to match these theoretical values:

<img src="https://raw.githubusercontent.com/mattiverson/TimeHistogram/master/figures/ExactQuantiles.png" width="80%"></img>

We can see that the computed estimates are very well aligned with the theoretical percentiles. Even over the discontinuities at x=20 and x=80, the computed estimates quickly transition between the two theoretical values on either side.

We can also see that our quantitative claims were fairly accurate: the exact interquartile range for x < 20 is 0.5, and 0.27 for 20 < x < 40.

TimeHistogram also allows us to choose different bandwidths for the smoothing kernel, in the X and Y dimensions. This lets us control how smooth the estimated percentiles are. In other words, this controlls the bias-variance tradeoff. For example, here are the results with a tighter X bandwidth, producing less smoothed results (less bias, more variance):

<img src="https://raw.githubusercontent.com/mattiverson/TimeHistogram/master/figures/LowerBias.png" width="80%"></img>

And here are the results with a wider X bandwidth, producing more smooth results (more bias, less variance).

<img src="https://raw.githubusercontent.com/mattiverson/TimeHistogram/master/figures/LowerVariance.png" width="80%"></img>

The less smooth results adjust faster to the step changes, but don't track as closely to the exact values. The more smooth results track the exact values very well in the domains where the exact values are constant, or vary slowly, but they don't adjust as quickly to step changes, and they partly smooth over the true oscillation for 40 < x < 60.

You can tailor the choice of X bandwidth to your data, and the characteristic scale of the trend you are looking for. The Y bandwidth is also configurable, which can be useful in applications where the y-values in the data are known to contain Gaussian noise with a certain variance.
