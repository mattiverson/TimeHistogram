#if !defined INTEGRATOR_H
#error "Include Integrator.h instead of Integrator_AVX512.h"
#endif

#ifndef INTEGRATOR_AVX512_H
#define INTEGRATOR_AVX512_H

#include <immintrin.h>

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

  __m512 y = _mm512_fmadd_ps(_mm512_fmadd_ps(c2, x, c1), x, c0);
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
  // rBandwidth * (x-1)
  const __m512 m = _mm512_fmsub_ps(rBandwidth, x, rBandwidth);
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
static void FindDataBounds(float& xMin, float& xMax, float& yMin, float& yMax, const float* const __restrict xData, const float* const __restrict yData, const U64 n)
{
  const U64 blockN = n & ~15ULL;
  float _xMin = std::numeric_limits<float>::infinity();
  float _xMax = -std::numeric_limits<float>::infinity();
  float _yMin = std::numeric_limits<float>::infinity();
  float _yMax = -std::numeric_limits<float>::infinity();

  __m512 __xMin = _mm512_set1_ps(_xMin);
  __m512 __xMax = _mm512_set1_ps(_xMax);
  __m512 __yMin = _mm512_set1_ps(_yMin);
  __m512 __yMax = _mm512_set1_ps(_yMax);

  for (U64 i = 0; i < blockN; i += 16)
  {
    const __m512 x = _mm512_loadu_ps(xData + i);
    const __m512 y = _mm512_loadu_ps(yData + i);
    __xMin = _mm512_min_ps(x, __xMin);
    __xMax = _mm512_max_ps(x, __xMax);
    __yMin = _mm512_min_ps(y, __yMin);
    __yMax = _mm512_max_ps(y, __yMax);
  }
  _xMin = _mm512_reduce_min_ps(__xMin);
  _xMax = _mm512_reduce_max_ps(__xMax);
  _yMin = _mm512_reduce_min_ps(__yMin);
  _yMax = _mm512_reduce_max_ps(__yMax);

  for (U64 i = blockN; i < n; ++i)
  {
    _xMin = (xData[i] < xMin) ? xData[i] : xMin;
    _xMax = (xData[i] > xMax) ? xData[i] : xMax;
    _yMin = (yData[i] < yMin) ? yData[i] : yMin;
    _yMax = (yData[i] > yMax) ? yData[i] : yMax;
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
  const __m512 wendInt = _mm512_set1_ps(WEND_INT);
  const __m512 bandwidth = _mm512_set1_ps(in.xBandwidth);
  const __m512 rBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), bandwidth);

  memset(out, 0, sizeof(float) * in.nGrid);

  static constexpr U64 DATA_BLOCK_SIZE = sizeof(float) * 1024;
  const float* __restrict pDataBlock = dataStart;
  const float* __restrict pDataBlockEnd = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlock) + DATA_BLOCK_SIZE);
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
    pDataBlockEnd = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlockEnd) + DATA_BLOCK_SIZE);
    pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
  } while (pDataBlock < dataEnd);
}




//
// Compute the integral along the line x = x[i], for each x[i] in xGrid,
//  for y <= y[i] in yGrid, of the sum of the kernel function of each data point.
// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
//
static void IntegrateToMass(float* const __restrict out, const IntegrateToMassInputs& in)
{
  const float* const __restrict dataStart = in.xData;
  const float* const __restrict dataEnd = dataStart + in.nData;
  const float* const __restrict gridStart = in.xGrid;
  const __m512 wendInt = _mm512_set1_ps(WEND_INT);
  const __m512 xBandwidth = _mm512_set1_ps(in.xBandwidth);
  const __m512 rXBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), xBandwidth);
  const __m512 yBandwidth = _mm512_set1_ps(in.yBandwidth);
  const __m512 rYBandwidth = _mm512_div_ps(_mm512_set1_ps(1.0f), yBandwidth);
  const __m512 oneHalf = _mm512_set1_ps(0.5f);

  static constexpr U64 DATA_BLOCK_SIZE = sizeof(float) * 1024;

  for (U64 i = 0; i < 20; ++i)
  {
    memset(out, 0, sizeof(float) * in.nGrid);
    const float* __restrict pDataBlock = dataStart;
    const float* __restrict pYDataBlock = in.yData;
    const float* __restrict pDataBlockEnd = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlock) + DATA_BLOCK_SIZE);
    pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
    do
    {
      U64 iGrid = 0;
      do
      {
        __m512 gridX = _mm512_loadu_ps(gridStart + iGrid);
        const __m512 gridYMin = _mm512_loadu_ps(in.lowerY + iGrid);
        const __m512 gridYMax = _mm512_loadu_ps(in.upperY + iGrid);
        __m512 twoGridY = _mm512_add_ps(gridYMin, gridYMax);
        const __m512 prevIntegrals = _mm512_loadu_ps(out + iGrid);
        __m512 integrals = _mm512_setzero_ps();

        const float* __restrict pData = pDataBlock;
        const float* __restrict pYData = pYDataBlock;
        do
        {
          __m512 dx = _mm512_sub_ps(gridX, _mm512_set1_ps(*pData));
          __m512 dy = _mm512_fmsub_ps(oneHalf, twoGridY, _mm512_set1_ps(*pYData));
          integrals = _mm512_fmadd_ps(WendlandEval(dx, xBandwidth, rXBandwidth), WendlandIntegral(dy, yBandwidth, rYBandwidth), integrals);
          pData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pData) + 4);
          pYData = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pYData) + 4);
        } while (pData < pDataBlockEnd);
        integrals = _mm512_add_ps(integrals, prevIntegrals);
        _mm512_storeu_ps(out + iGrid, integrals);
        iGrid += 16;
      } while (iGrid < in.nGrid);
      pDataBlock = pDataBlockEnd;
      pDataBlockEnd = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pDataBlockEnd) + DATA_BLOCK_SIZE);
      pDataBlockEnd = (dataEnd < pDataBlockEnd) ? dataEnd : pDataBlockEnd;
      pYDataBlock = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pYDataBlock) + DATA_BLOCK_SIZE);
    } while (pDataBlock < dataEnd);

    U64 iGrid = 0;
    do
    {
      __m512 mass = _mm512_loadu_ps(out + iGrid);
      __m512 targetMass = _mm512_loadu_ps(in.targetMass + iGrid);
      __mmask16 isOverTarget = _mm512_cmple_ps_mask(targetMass, mass);
      __mmask16 isUnderTarget = ~isOverTarget;
      const __m512 gridYMin = _mm512_loadu_ps(in.lowerY + iGrid);
      const __m512 gridYMax = _mm512_loadu_ps(in.upperY + iGrid);
      __m512 gridY = _mm512_mul_ps(oneHalf, _mm512_add_ps(gridYMin, gridYMax));

      _mm512_mask_storeu_ps(in.upperMass + iGrid, isOverTarget, mass);
      _mm512_mask_storeu_ps(in.upperY + iGrid, isOverTarget, gridY);
      _mm512_mask_storeu_ps(in.lowerMass + iGrid, isUnderTarget, mass);
      _mm512_mask_storeu_ps(in.lowerY + iGrid, isUnderTarget, gridY);

      iGrid += 16;
    } while (iGrid < in.nGrid);
  }
  memcpy(out, in.lowerY, sizeof(float) * in.nGrid);
}

#endif // INTEGRATOR_AVX512_H