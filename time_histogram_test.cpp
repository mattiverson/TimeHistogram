#include "time_histogram.h"

using U64 = uint64_t;

void TestCorrectness()
{
    constexpr U64 nData = 16;
    float xData[nData];
    float yData[nData];
    for (U64 i = 0; i < nData; ++i)
    {
        xData[i] = static_cast<float>(i) / 16.0f;
        yData[i] = 4.0f * xData[i] * (1.0f - xData[i]);
    }
    TimeHistogram::TimeHistogram hist(xData, yData, nData, 1.0f, 1.0f);
    float quantileData[3 * 1024];
    constexpr float quantiles[] = { 0.0f, 0.5f, 1.0f };
    hist.ComputeQuantiles(quantileData, quantiles, 3);
}
