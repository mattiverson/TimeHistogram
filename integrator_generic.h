#pragma once

#if !defined ALLOW_INTEGRATOR_IMPL
#error "Include integrator.h instead of integrator_generic.h"
#endif

#include <thread>
#include <atomic>

#include "integrator_common.h"

namespace Integrator
{

static float Min(const float a, const float b) { return (a < b) ? a : b; }

static float Max(const float a, const float b) { return (a > b) ? a : b; }

//
// Evaluate the C4 Wendland kernel with the given bandwidth, at a vector of 16 points.
//
static float WendlandEval(float x, const float bandwidth, const float rBandwidth)
{
    // take abs value, clamp to [0, 1], map nans to 1.0
    x = fabsf(x);
    x = Min(x, bandwidth);
    const float m = rBandwidth * x - 1.0f;
    const float m3 = m * m * m;
    const float m6 = m3 * m3;

    float y = (WEND_C2 * m + WEND_C1) * m + WEND_C0;
    y = y * m6;
    return y;
}

//
// Integrate the C4 Wendland kernel from -bandwidth to the given value.
//
static float WendlandIntegral(float x, const float bandwidth, const float rBandwidth)
{
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    x = fabsf(x);
    x = Min(x, bandwidth);

    const float m = rBandwidth * x - 1.0f;
    const float m3 = m * m * m;
    const float m6 = m3 * m3;
    const float m7 = m6 * m;

    constexpr float wi = (0.5f * WEND_INT);
    const float i = ((WEND_INT_C2 * m + WEND_INT_C1) * m + WEND_INT_C0) * m7 + wi;
    // This is the value of the integral times rBandwidth. Since we only use this integral to find
    // relative mass (e.g. 50th percentile), we can ignore this constant factor.
    return sign * i + wi;
}

//
// Find the min and max x and y coordinate of the given data set.
//
static void FindDataBounds(float& xMin,
                           float& xMax,
                           float& yMin,
                           float& yMax,
                           const float* const __restrict xData,
                           const float* const __restrict yData,
                           const U64 n)
{
    float _xMin = std::numeric_limits<float>::infinity();
    float _xMax = -std::numeric_limits<float>::infinity();
    float _yMin = std::numeric_limits<float>::infinity();
    float _yMax = -std::numeric_limits<float>::infinity();

    for (U64 i = 0; i < n; ++i)
    {
        _xMin = (xData[i] < _xMin) ? xData[i] : _xMin;
        _xMax = (xData[i] > _xMax) ? xData[i] : _xMax;
        _yMin = (yData[i] < _yMin) ? yData[i] : _yMin;
        _yMax = (yData[i] > _yMax) ? yData[i] : _yMax;
    }
    xMin = _xMin;
    xMax = _xMax;
    yMin = _yMin;
    yMax = _yMax;
}

//
// Compute the integral along the line x = x[i], for each x[i] in xGrid,
//  of the sum of the kernel function of each data point.
// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
//
static void LineIntegrals(float* const __restrict out, const LineIntegralsInputs& in)
{
    const float* const __restrict dataStart = in.xData;
    const float* const __restrict dataEnd = dataStart + in.nData;
    const float* const __restrict gridStart = in.xGrid;
    const float* const __restrict gridEnd = gridStart + in.nGrid;
    const float bandwidth = in.xBandwidth;
    const float rBandwidth = 1.0f / bandwidth;

    memset(out, 0, sizeof(float) * in.nGrid);

    static constexpr U64 DATA_BLOCK_SIZE = sizeof(float) * 1024;
    const float* __restrict pDataBlock = dataStart;
    const float* __restrict pDataBlockEnd =
        reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlock) + DATA_BLOCK_SIZE);
    pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
    do
    {
        const float* __restrict pGrid = gridStart;
        float* __restrict write = out;
        do
        {
            float integrals = 0.0f;
            float gridX = *pGrid;
            float prevIntegrals = *write;
            pGrid = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pGrid) + 4);

            const float* __restrict pData = pDataBlock;
            do
            {
                float dx = gridX - *pData;
                integrals += WendlandEval(dx, bandwidth, rBandwidth);
                ++pData;
            } while (pData < pDataBlockEnd);
            prevIntegrals += WEND_INT * integrals;
            *write = prevIntegrals;
            write = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(write) + 4);
        } while (pGrid < gridEnd);
        pDataBlock = pDataBlockEnd;
        pDataBlockEnd = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlockEnd) +
                                                       DATA_BLOCK_SIZE);
        pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
    } while (pDataBlock < dataEnd);
}


//
// Compute the integral along the line x = x[i], for each x[i] in xGrid,
//  for y <= y[i] in yGrid, of the sum of the kernel function of each data point.
// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
//
static void IntegrateToMassWorker(float* const __restrict out,
                                  const IntegrateToMassInputs& in,
                                  std::atomic<U64>& sharedGridIdx)
{
    U64 gridIdx;
    constexpr U64 kWidth = 16;
    while ((gridIdx = sharedGridIdx.fetch_add(kWidth, std::memory_order_relaxed)) < in.nGrid)
    {
        const float* const __restrict dataStart = in.xData;
        const float* const __restrict yDataStart = in.yData;
        const float* const __restrict dataEnd = dataStart + in.nData;
        const float xBandwidth = in.xBandwidth;
        const float rXBandwidth = 1.0f / xBandwidth;
        const float yBandwidth = in.yBandwidth;
        const float rYBandwidth = 1.0f / yBandwidth;
        float targetMass[kWidth];
        float gridX[kWidth];
        float gridYMin[kWidth];
        float gridYMax[kWidth];
        float lowerMass[kWidth];
        float upperMass[kWidth];
        for (U64 j = 0; j < kWidth; ++j)
        {
            targetMass[j] = in.targetMass[gridIdx + j];
            gridX[j] = in.xGrid[gridIdx + j];
            gridYMin[j] = in.lowerY[gridIdx + j];
            gridYMax[j] = in.upperY[gridIdx + j];
            lowerMass[j] = in.lowerMass[gridIdx + j];
            upperMass[j] = in.upperMass[gridIdx + j];
        }

        // Bisect 20 times, to converge to the quantile.
        // Could be much smarter than bisecting (secant method, maybe NR).
        for (U64 i = 0; i < 20; ++i)
        {
            const float* __restrict pData = dataStart;
            const float* __restrict pYData = yDataStart;
            float gridY[kWidth];
            float integrals[kWidth] = {};
            for (U64 j = 0; j < kWidth; ++j)
                gridY[j] = 0.5f * (gridYMin[j] + gridYMax[j]);

            do
            {
                for (U64 j = 0; j < kWidth; ++j)
                {
                    const float dx = gridX[j] - *pData;
                    const float dy = gridY[j] - *pYData;
                    integrals[j] += WendlandEval(dx, xBandwidth, rXBandwidth) *
                                    WendlandIntegral(dy, yBandwidth, rYBandwidth);
                }
                pData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pData) + 4);
                pYData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pYData) + 4);
            } while (pData < dataEnd);

            for (U64 j = 0; j < kWidth; ++j)
            {
                float mass = integrals[j];
                const bool isOverTarget = targetMass[j] <= mass;

                lowerMass[j] = isOverTarget ? lowerMass[j] : mass;
                gridYMin[j] = isOverTarget ? gridYMin[j] : gridY[j];
                upperMass[j] = isOverTarget ? mass : upperMass[j];
                gridYMax[j] = isOverTarget ? gridY[j] : gridYMax[j];
            }
        }
        for (U64 j = 0; j < kWidth; ++j)
            out[gridIdx + j] = gridYMin[j];
    }
}


//
// Compute the integral along the line x = x[i], for each x[i] in xGrid,
//  for y <= y[i] in yGrid, of the sum of the kernel function of each data point.
// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
//
static void IntegrateToMass(float* const __restrict out, const IntegrateToMassInputs& in)
{
    std::atomic<U64> sharedGridIdx = 0;
    constexpr U64 minThreads = 1;
    const U64 maxThreads = std::thread::hardware_concurrency();
    const U64 targetThreads = (in.nData + 1023) >> 10;
    U64 nThreads = (targetThreads < maxThreads) ? targetThreads : maxThreads;
    nThreads = (nThreads > minThreads) ? nThreads : minThreads;
    std::thread* threads = static_cast<std::thread*>(malloc(sizeof(std::thread) * nThreads - 1));
    for (U64 tIdx = 0; tIdx < (nThreads - 1); ++tIdx)
    {
        new (threads + tIdx)
            std::thread(IntegrateToMassWorker, out, std::ref(in), std::ref(sharedGridIdx));
    }
    IntegrateToMassWorker(out, in, sharedGridIdx);
    for (U64 tIdx = 0; tIdx < (nThreads - 1); ++tIdx)
        threads[tIdx].join();
}

} // namespace Integrator
