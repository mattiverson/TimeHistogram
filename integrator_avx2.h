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
static __m256 WendlandEval(__m256 x, const __m256 bandwidth, const __m256 rBandwidth)
{
    // take abs value, clamp to [0, 1], map nans to 1.0
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 signMask = _mm256_set1_ps(SIGN_MASK);
    x = _mm256_andnot_ps(signMask, x);
    x = _mm256_min_ps(x, bandwidth);
    const __m256 m = _mm256_fmsub_ps(rBandwidth, x, one);
    const __m256 m3 = _mm256_mul_ps(m, _mm256_mul_ps(m, m));
    const __m256 m6 = _mm256_mul_ps(m3, m3);

    const __m256 c0 = _mm256_set1_ps(WEND_C0);
    const __m256 c1 = _mm256_set1_ps(WEND_C1);
    const __m256 c2 = _mm256_set1_ps(WEND_C2);

    __m256 y = _mm256_fmadd_ps(_mm256_fmadd_ps(c2, m, c1), m, c0);
    y = _mm256_mul_ps(y, m6);
    return y;
}

//
// Integrate the C4 Wendland kernel from -bandwidth to the given value.
//
static __m256 WendlandIntegral(__m256 x, const __m256 bandwidth, const __m256 rBandwidth)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 signBit = _mm256_and_ps(_mm256_set1_ps(SIGN_MASK), x);
    const __m256 sign = _mm256_xor_ps(signBit, one);
    x = _mm256_xor_ps(x, signBit);
    x = _mm256_min_ps(x, bandwidth);
    const __m256 m = _mm256_fmsub_ps(rBandwidth, x, one);
    const __m256 m3 = _mm256_mul_ps(m, _mm256_mul_ps(m, m));
    const __m256 m6 = _mm256_mul_ps(m3, m3);
    const __m256 m7 = _mm256_mul_ps(m6, m);

    const __m256 c0 = _mm256_set1_ps(WEND_INT_C0);
    const __m256 c1 = _mm256_set1_ps(WEND_INT_C1);
    const __m256 c2 = _mm256_set1_ps(WEND_INT_C2);
    const __m256 wi = _mm256_set1_ps(0.5f * WEND_INT);

    __m256 i = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(c2, m, c1), m, c0), m7, wi);
    // This is the value of the integral times rBandwidth. Since we only use this integral to find
    // relative mass (e.g. 50th percentile), we can ignore this constant factor.
    return _mm256_fmadd_ps(sign, i, wi);
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
    const U64 blockN = n & ~7ULL;
    float xMin_ = std::numeric_limits<float>::infinity();
    float xMax_ = -std::numeric_limits<float>::infinity();
    float yMin_ = std::numeric_limits<float>::infinity();
    float yMax_ = -std::numeric_limits<float>::infinity();

    __m256 xMinVec = _mm256_set1_ps(xMin_);
    __m256 xMaxVec = _mm256_set1_ps(xMax_);
    __m256 yMinVec = _mm256_set1_ps(yMin_);
    __m256 yMaxVec = _mm256_set1_ps(yMax_);

    for (U64 i = 0; i < blockN; i += 8)
    {
        const __m256 x = _mm256_loadu_ps(xData + i);
        const __m256 y = _mm256_loadu_ps(yData + i);
        xMinVec = _mm256_min_ps(x, xMinVec);
        xMaxVec = _mm256_max_ps(x, xMaxVec);
        yMinVec = _mm256_min_ps(y, yMinVec);
        yMaxVec = _mm256_max_ps(y, yMaxVec);
    }
    xMinVec = _mm256_min_ps(xMinVec, _mm256_permute2f128_ps(xMinVec, xMinVec, 1));
    xMinVec = _mm256_min_ps(xMinVec, _mm256_permute_ps(xMinVec, 177));
    xMinVec = _mm256_min_ps(xMinVec, _mm256_permute_ps(xMinVec, 78));
    xMin_ = _mm256_cvtss_f32(xMinVec);
    xMaxVec = _mm256_max_ps(xMaxVec, _mm256_permute2f128_ps(xMaxVec, xMaxVec, 1));
    xMaxVec = _mm256_max_ps(xMaxVec, _mm256_permute_ps(xMaxVec, 177));
    xMaxVec = _mm256_max_ps(xMaxVec, _mm256_permute_ps(xMaxVec, 78));
    xMax_ = _mm256_cvtss_f32(xMaxVec);
    yMinVec = _mm256_min_ps(yMinVec, _mm256_permute2f128_ps(yMinVec, yMinVec, 1));
    yMinVec = _mm256_min_ps(yMinVec, _mm256_permute_ps(yMinVec, 177));
    yMinVec = _mm256_min_ps(yMinVec, _mm256_permute_ps(yMinVec, 78));
    yMin_ = _mm256_cvtss_f32(yMinVec);
    yMaxVec = _mm256_max_ps(yMaxVec, _mm256_permute2f128_ps(yMaxVec, yMaxVec, 1));
    yMaxVec = _mm256_max_ps(yMaxVec, _mm256_permute_ps(yMaxVec, 177));
    yMaxVec = _mm256_max_ps(yMaxVec, _mm256_permute_ps(yMaxVec, 78));
    yMax_ = _mm256_cvtss_f32(yMaxVec);

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
    const __m256 wendInt = _mm256_set1_ps(WEND_INT);
    const __m256 bandwidth = _mm256_set1_ps(in.xBandwidth);
    const __m256 rBandwidth = _mm256_div_ps(_mm256_set1_ps(1.0f), bandwidth);

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
            __m256 integrals = _mm256_setzero_ps();
            __m256 gridX = _mm256_loadu_ps(pGrid);
            const __m256 prevIntegrals = _mm256_loadu_ps(write);
            pGrid = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pGrid) + 32);

            const float* __restrict pData = pDataBlock;
            do
            {
                __m256 dx = _mm256_sub_ps(gridX, _mm256_set1_ps(*pData));
                integrals = _mm256_add_ps(integrals, WendlandEval(dx, bandwidth, rBandwidth));
                ++pData;
            } while (pData < pDataBlockEnd);
            integrals = _mm256_fmadd_ps(wendInt, integrals, prevIntegrals);
            _mm256_storeu_ps(write, integrals);
            write = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(write) + 32);
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
    constexpr U32 kWidth = 2;
    while ((gridIdx = sharedGridIdx.fetch_add(8 * kWidth, std::memory_order_relaxed)) < in.nGrid)
    {
        const float* const __restrict dataStart = in.xData;
        const float* const __restrict yDataStart = in.yData;
        const float* const __restrict dataEnd = dataStart + in.nData;
        const __m256 xBandwidth = _mm256_set1_ps(in.xBandwidth);
        const __m256 rXBandwidth = _mm256_div_ps(_mm256_set1_ps(1.0f), xBandwidth);
        const __m256 yBandwidth = _mm256_set1_ps(in.yBandwidth);
        const __m256 rYBandwidth = _mm256_div_ps(_mm256_set1_ps(1.0f), yBandwidth);
        const __m256 oneHalf = _mm256_set1_ps(0.5f);
        __m256 targetMass[kWidth];
        __m256 gridX[kWidth];
        __m256 gridYMin[kWidth];
        __m256 gridYMax[kWidth];
        __m256 lowerMass[kWidth];
        __m256 upperMass[kWidth];
        for (U64 j = 0; j < kWidth; ++j)
        {
            targetMass[j] = _mm256_loadu_ps(in.targetMass + gridIdx + 8 * j);
            gridX[j] = _mm256_loadu_ps(in.xGrid + gridIdx + 8 * j);
            gridYMin[j] = _mm256_loadu_ps(in.lowerY + gridIdx + 8 * j);
            gridYMax[j] = _mm256_loadu_ps(in.upperY + gridIdx + 8 * j);
            lowerMass[j] = _mm256_loadu_ps(in.lowerMass + gridIdx + 8 * j);
            upperMass[j] = _mm256_loadu_ps(in.upperMass + gridIdx + 8 * j);
        }

        // Bisect 20 times, to converge to the quantile.
        // Could be much smarter than bisecting (secant method, maybe NR).
        for (U64 i = 0; i < 20; ++i)
        {
            const float* __restrict pData = dataStart;
            const float* __restrict pYData = yDataStart;
            __m256 gridY[kWidth];
            __m256 integrals[kWidth];
            for (U64 j = 0; j < kWidth; ++j)
            {
                gridY[j] = _mm256_mul_ps(oneHalf, _mm256_add_ps(gridYMin[j], gridYMax[j]));
                integrals[j] = _mm256_setzero_ps();
            }

            do
            {
                for (U64 j = 0; j < kWidth; ++j)
                {
                    __m256 dx = _mm256_sub_ps(gridX[j], _mm256_set1_ps(*pData));
                    __m256 dy = _mm256_sub_ps(gridY[j], _mm256_set1_ps(*pYData));
                    integrals[j] = _mm256_fmadd_ps(WendlandEval(dx, xBandwidth, rXBandwidth),
                                                   WendlandIntegral(dy, yBandwidth, rYBandwidth),
                                                   integrals[j]);
                }
                pData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pData) + 4);
                pYData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pYData) + 4);
            } while (pData < dataEnd);

            for (U64 j = 0; j < kWidth; ++j)
            {
                __m256 mass = integrals[j];
                __m256 isOverTarget = _mm256_cmp_ps(targetMass[j], mass, _CMP_LE_OS);

                lowerMass[j] = _mm256_blendv_ps(mass, lowerMass[j], isOverTarget);
                gridYMin[j] = _mm256_blendv_ps(gridY[j], gridYMin[j], isOverTarget);
                upperMass[j] = _mm256_blendv_ps(upperMass[j], mass, isOverTarget);
                gridYMax[j] = _mm256_blendv_ps(gridYMax[j], gridY[j], isOverTarget);
            }
        }
        for (U64 j = 0; j < kWidth; ++j)
            _mm256_storeu_ps(out + gridIdx + 8 * j, gridYMin[j]);
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
