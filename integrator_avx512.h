#pragma once

#if !defined ALLOW_INTEGRATOR_IMPL
#error "Include Integrator.h instead of Integrator_AVX512.h"
#endif

#include <immintrin.h>
#include <thread>
#include <atomic>

#include "integrator_common.h"

namespace Integrator
{
//
// Evaluate the C4 Wendland kernel with the given bandwidth, at a vector of 16 points.
//
static __m512 WendlandEval(__m512 x, const __m512 bandwidth, const __m512 rBandwidth)
{
    // take abs value, clamp to [0, 1], map nans to 1.0
    const __m512 one = _mm512_set1_ps(1.0f);
    x = _mm512_abs_ps(x);
    x = _mm512_min_ps(x, bandwidth);
    const __m512 m = _mm512_fmsub_ps(rBandwidth, x, one);
    const __m512 m3 = _mm512_mul_ps(m, _mm512_mul_ps(m, m));
    const __m512 m6 = _mm512_mul_ps(m3, m3);

    const __m512 c0 = _mm512_set1_ps(WEND_C0);
    const __m512 c1 = _mm512_set1_ps(WEND_C1);
    const __m512 c2 = _mm512_set1_ps(WEND_C2);

    __m512 y = _mm512_fmadd_ps(_mm512_fmadd_ps(c2, m, c1), m, c0);
    y = _mm512_mul_ps(y, m6);
    return y;
}

//
// Integrate the C4 Wendland kernel from -bandwidth to the given value.
//
static __m512 WendlandIntegral(__m512 x, const __m512 bandwidth, const __m512 rBandwidth)
{
    // consider converting and/xor to epi32 so only AVX512F is needed
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 signBit = _mm512_and_ps(_mm512_set1_ps(SIGN_MASK), x);
    const __m512 sign = _mm512_xor_ps(signBit, one);
    x = _mm512_abs_ps(x);
    x = _mm512_min_ps(x, bandwidth);
    // (x/bandwidth) - 1
    const __m512 m = _mm512_fmsub_ps(rBandwidth, x, one);
    const __m512 m3 = _mm512_mul_ps(m, _mm512_mul_ps(m, m));
    const __m512 m6 = _mm512_mul_ps(m3, m3);
    const __m512 m7 = _mm512_mul_ps(m6, m);

    const __m512 c0 = _mm512_set1_ps(WEND_INT_C0);
    const __m512 c1 = _mm512_set1_ps(WEND_INT_C1);
    const __m512 c2 = _mm512_set1_ps(WEND_INT_C2);
    const __m512 wi = _mm512_set1_ps(0.5f * WEND_INT);

    __m512 i = _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_fmadd_ps(c2, m, c1), m, c0), m7, wi);
    // This is the value of the integral times rBandwidth. Since we only use this integral to find
    // relative mass (e.g. 50th percentile), we can ignore this constant factor.
    return _mm512_fmadd_ps(sign, i, wi);
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
    const U64 blockN = n & ~15ULL;
    float xMin_ = std::numeric_limits<float>::infinity();
    float xMax_ = -std::numeric_limits<float>::infinity();
    float yMin_ = std::numeric_limits<float>::infinity();
    float yMax_ = -std::numeric_limits<float>::infinity();

    __m512 xMinVec = _mm512_set1_ps(xMin_);
    __m512 xMaxVec = _mm512_set1_ps(xMax_);
    __m512 yMinVec = _mm512_set1_ps(yMin_);
    __m512 yMaxVec = _mm512_set1_ps(yMax_);

    for (U64 i = 0; i < blockN; i += 16)
    {
        const __m512 x = _mm512_loadu_ps(xData + i);
        const __m512 y = _mm512_loadu_ps(yData + i);
        xMinVec = _mm512_min_ps(x, xMinVec);
        xMaxVec = _mm512_max_ps(x, xMaxVec);
        yMinVec = _mm512_min_ps(y, yMinVec);
        yMaxVec = _mm512_max_ps(y, yMaxVec);
    }
    xMin_ = _mm512_reduce_min_ps(xMinVec);
    xMax_ = _mm512_reduce_max_ps(xMaxVec);
    yMin_ = _mm512_reduce_min_ps(yMinVec);
    yMax_ = _mm512_reduce_max_ps(yMaxVec);

    for (U64 i = blockN; i < n; ++i)
    {
        xMin_ = (xData[i] < xMin_) ? xData[i] : xMin_;
        xMax_ = (xData[i] > xMax_) ? xData[i] : xMax_;
        yMin_ = (yData[i] < yMin_) ? yData[i] : yMin_;
        yMax_ = (yData[i] > yMax_) ? yData[i] : yMax_;
    }
    xMin = xMin_;
    xMax = xMax_;
    yMin = yMin_;
    yMax = yMax_;
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
    const __m512 wendInt = _mm512_set1_ps(WEND_INT);
    const __m512 bandwidth = _mm512_set1_ps(in.xBandwidth);
    const __m512 rBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), bandwidth);

    memset(out, 0, sizeof(float) * in.nGrid);

    static constexpr U64 DATA_BLOCK_SIZE = sizeof(float) * 1024;
    const float* __restrict pDataBlock = dataStart;
    auto pDataBlockEnd =
        reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlock) + DATA_BLOCK_SIZE);
    pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
    do
    {
        const float* __restrict pGrid = gridStart;
        float* __restrict write = out;
        do
        {
            __m512 integrals = _mm512_setzero_ps();
            __m512 gridX = _mm512_loadu_ps(pGrid);
            const __m512 prevIntegrals = _mm512_loadu_ps(write);
            pGrid = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pGrid) + 64);

            const float* __restrict pData = pDataBlock;
            do
            {
                __m512 dx = _mm512_sub_ps(gridX, _mm512_set1_ps(*pData));
                integrals = _mm512_add_ps(integrals, WendlandEval(dx, bandwidth, rBandwidth));
                ++pData;
            } while (pData < pDataBlockEnd);
            integrals = _mm512_fmadd_ps(wendInt, integrals, prevIntegrals);
            _mm512_storeu_ps(write, integrals);
            write = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(write) + 64);
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
    constexpr U32 kWidth = 1;
    while ((gridIdx = sharedGridIdx.fetch_add(16 * kWidth, std::memory_order_relaxed)) < in.nGrid)
    {
        const float* const __restrict dataStart = in.xData;
        const float* const __restrict yDataStart = in.yData;
        const float* const __restrict dataEnd = dataStart + in.nData;
        const __m512 xBandwidth = _mm512_set1_ps(in.xBandwidth);
        const __m512 rXBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), xBandwidth);
        const __m512 yBandwidth = _mm512_set1_ps(in.yBandwidth);
        const __m512 rYBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), yBandwidth);
        const __m512 oneHalf = _mm512_set1_ps(0.5f);
        __m512 targetMass[kWidth];
        __m512 gridX[kWidth];
        __m512 gridYMin[kWidth];
        __m512 gridYMax[kWidth];
        __m512 lowerMass[kWidth];
        __m512 upperMass[kWidth];
        for (U64 j = 0; j < kWidth; ++j)
        {
            targetMass[j] = _mm512_loadu_ps(in.targetMass + gridIdx + 16 * j);
            gridX[j] = _mm512_loadu_ps(in.xGrid + gridIdx + 16 * j);
            gridYMin[j] = _mm512_loadu_ps(in.lowerY + gridIdx + 16 * j);
            gridYMax[j] = _mm512_loadu_ps(in.upperY + gridIdx + 16 * j);
            lowerMass[j] = _mm512_loadu_ps(in.lowerMass + gridIdx + 16 * j);
            upperMass[j] = _mm512_loadu_ps(in.upperMass + gridIdx + 16 * j);
        }

        // Bisect 20 times, to converge to the quantile.
        // Could be much smarter than bisecting (secant method, maybe NR).
        for (U64 i = 0; i < 20; ++i)
        {
            const float* __restrict pData = dataStart;
            const float* __restrict pYData = yDataStart;
            __m512 gridY[kWidth];
            __m512 integrals[kWidth];
            for (U64 j = 0; j < kWidth; ++j)
            {
                gridY[j] = _mm512_mul_ps(oneHalf, _mm512_add_ps(gridYMin[j], gridYMax[j]));
                integrals[j] = _mm512_setzero_ps();
            }

            do
            {
                for (U64 j = 0; j < kWidth; ++j)
                {
                    __m512 dx = _mm512_sub_ps(gridX[j], _mm512_set1_ps(*pData));
                    __m512 dy = _mm512_sub_ps(gridY[j], _mm512_set1_ps(*pYData));
                    integrals[j] = _mm512_fmadd_ps(WendlandEval(dx, xBandwidth, rXBandwidth),
                                                   WendlandIntegral(dy, yBandwidth, rYBandwidth),
                                                   integrals[j]);
                }
                pData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pData) + 4);
                pYData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pYData) + 4);
            } while (pData < dataEnd);

            for (U64 j = 0; j < kWidth; ++j)
            {
                __m512 mass = integrals[j];
                __mmask16 isOverTarget = _mm512_cmple_ps_mask(targetMass[j], mass);
                __mmask16 isUnderTarget = ~isOverTarget;

                lowerMass[j] = _mm512_mask_blend_ps(isOverTarget, mass, lowerMass[j]);
                gridYMin[j] = _mm512_mask_blend_ps(isOverTarget, gridY[j], gridYMin[j]);
                upperMass[j] = _mm512_mask_blend_ps(isUnderTarget, mass, upperMass[j]);
                gridYMax[j] = _mm512_mask_blend_ps(isUnderTarget, gridY[j], gridYMax[j]);
            }
        }

        for (U64 j = 0; j < kWidth; ++j)
            _mm512_storeu_ps(out + gridIdx + 16 * j, gridYMin[j]);
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
    auto threads = static_cast<std::thread*>(malloc(sizeof(std::thread) * nThreads - 1));
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
