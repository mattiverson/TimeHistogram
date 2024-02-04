#include <chrono>

#include "time_histogram.h"

using U64 = uint64_t;

void Profile()
{
    std::chrono::high_resolution_clock clock;
    auto startData = clock.now();

    constexpr U64 nData = 1048576;
    float* xData;
    float* yData;
    xData = reinterpret_cast<float*>(_mm_malloc(2 * nData * sizeof(float), 4096));
    yData = xData + nData;

    for (U64 i = 0; i < nData; ++i)
    {
        xData[i] = -123.45f + 0.00076567f * static_cast<float>(i);
        yData[i] = sinf(xData[i]);
    }

    auto stopData = clock.now();
    auto dataDuration = stopData - startData;
    U64 dataNanos = dataDuration.count();
    printf("Data generated in %llu nanos\n", dataNanos);

    TimeHistogram::TimeHistogram hist(xData, yData, nData, 1.0f, 1.0f);
    float quantileData[5 * 1024];
    constexpr float quantiles[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
    auto computeStart = clock.now();
    hist.ComputeQuantiles(quantileData, quantiles, 5);
    auto computeStop = clock.now();
    auto computeDuration = computeStop - computeStart;
    U64 computeNanos = computeDuration.count();
    U64 integralCount = 100ULL * 1024ULL * nData;
    printf("%llu integrals in %llu nanos\n", integralCount, computeNanos);
    double integralRate = static_cast<double>(integralCount) / static_cast<double>(computeNanos);
    printf("Rate: %f G/s\n", integralRate);
}
