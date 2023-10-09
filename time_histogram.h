#pragma once
#include <cstdint>

#include "integrator.h"

namespace TimeHistogram {

using U32 = uint32_t;
using U64 = uint64_t;

class TimeHistogram
{
public:
  TimeHistogram(const float* x, const float* y, const U64 n);
  TimeHistogram() = delete;
  TimeHistogram(const TimeHistogram& other) = delete;
  TimeHistogram(TimeHistogram&& other) = delete;
  ~TimeHistogram();
  void ComputeQuantiles(
    float* const __restrict out,
    const float* const __restrict quantiles,
    const U64 nQuantiles);

  void SetTimeBandwidth(const float x)
  {
    xBandwidth = x;
  }

  void SetSpaceBandwidth(const float y)
  {
    yBandwidth = y;
  }

  void SetXGridRange(const float xMin, const float xMax)
  {
    const float stepSize = (xMax - xMin) / float(nGrid - 1);
    float step = 0.0f;
    for (U64 i = 0; i < nGrid; ++i)
    {
      xGrid[i] = xMin + step * stepSize;
      step += 1.0f;
    }
  }

  U64 GetNGrid()
  {
    return nGrid;
  }

private:

  void FillLineIntegralsInputs(LineIntegralsInputs& out)
  {
    out.xGrid = xGrid;
    out.nGrid = nGrid;
    out.xData = xData;
    out.nData = nData;
    out.xBandwidth = xBandwidth;
  }

  void FillIntegrateToMassInputs(IntegrateToMassInputs& out)
  {
    out.xGrid = xGrid;
    out.targetMass = targetMass;
    out.lowerY = lowerY;
    out.lowerMass = lowerMass;
    out.upperY = upperY;
    out.upperMass = upperMass;
    out.nGrid = nGrid;
    out.xData = xData;
    out.yData = yData;
    out.nData = nData;
    out.xBandwidth = xBandwidth;
    out.yBandwidth = yBandwidth;
    out.yMin = yMin;
    out.yMax = yMax;
  }

  const U64 nData;
  const float* const __restrict xData;
  const float* const __restrict yData;
  U64 nGrid;
  float* __restrict xGrid;
  float* __restrict lowerY;
  float* __restrict lowerMass;
  float* __restrict upperY;
  float* __restrict upperMass;
  float* __restrict targetMass;
  float* __restrict totalMass;
  float xBandwidth;
  float yBandwidth;
  float xMin;
  float xMax;
  float yMin;
  float yMax;
};

}