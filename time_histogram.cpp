#include "time_histogram.h"

namespace TimeHistogram {

TimeHistogram::TimeHistogram(const float* x, const float* y, const U64 n)
  : nData{n}
  , xData{x}
  , yData{y}
  , nGrid{1024}
  , xBandwidth{1.0f}
  , yBandwidth{1.0f}
{
  xGrid = reinterpret_cast<float*>(_mm_malloc(sizeof(float) * nGrid * 7, 64));
  lowerY = xGrid + nGrid;
  lowerMass = xGrid + 2*nGrid;
  upperY = xGrid + 3*nGrid;
  upperMass = xGrid + 4*nGrid;
  targetMass = xGrid + 5*nGrid;
  totalMass = xGrid + 6*nGrid;
  Integrator::FindDataBounds(xMin, xMax, yMin, yMax, x, y, n);
  SetXGridRange(xMin, xMax);
  Integrator::LineIntegralsInputs in;
  FillLineIntegralsInputs(in);
  Integrator::LineIntegrals(totalMass, in);
}

TimeHistogram::~TimeHistogram()
{
  _mm_free(xGrid);
}

void TimeHistogram::ComputeQuantiles(
  float* const __restrict out,
  const float* const __restrict quantiles,
  const U64 nQuantiles)
{
  const float loY = yMin - yBandwidth;
  const float hiY = yMax + yBandwidth;

  Integrator::IntegrateToMassInputs in;
  FillIntegrateToMassInputs(in);
  for (U64 i = 0; i < nQuantiles; ++i)
  {
    const float quantile = quantiles[i];
    memcpy(upperMass, totalMass, nGrid * sizeof(float));
    for (U64 j = 0; j < nGrid; ++j)
    {
      targetMass[j] = totalMass[j] * quantile;
      lowerY[j] = loY;
      lowerMass[j] = 0.0f;
      upperY[j] = hiY;
    }

    Integrator::IntegrateToMass(out + i*nGrid, in);
  }
}

}