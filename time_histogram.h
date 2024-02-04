#pragma once
#include <cstdint>

#include "integrator.h"

namespace TimeHistogram
{

using U32 = uint32_t;
using U64 = uint64_t;

class TimeHistogram
{
  public:
    TimeHistogram(const float* x,
                  const float* y,
                  const U64 n,
                  const float xBandwidth,
                  const float yBandwidth);
    TimeHistogram() = delete;
    TimeHistogram(const TimeHistogram& other) = delete;
    TimeHistogram(TimeHistogram&& other) = delete;
    ~TimeHistogram();
    void ComputeQuantiles(float* const __restrict out,
                          const float* const __restrict quantiles,
                          const U64 nQuantiles);

    void SetXBandwidth(const float x) { m_xBandwidth = x; }

    void SetYBandwidth(const float y) { m_yBandwidth = y; }

    void SetXGridRange(const float xMin, const float xMax)
    {
        const float stepSize = (xMax - xMin) / float(m_nGrid - 1);
        float step = 0.0f;
        for (U64 i = 0; i < m_nGrid; ++i)
        {
            m_xGrid[i] = xMin + step * stepSize;
            step += 1.0f;
        }
    }

    U64 GetNGrid() { return m_nGrid; }

    const float* GetXGrid() { return m_xGrid; }

  private:
    void FillLineIntegralsInputs(Integrator::LineIntegralsInputs& out)
    {
        out.xGrid = m_xGrid;
        out.nGrid = m_nGrid;
        out.xData = m_xData;
        out.nData = m_nData;
        out.xBandwidth = m_xBandwidth;
    }

    void FillIntegrateToMassInputs(Integrator::IntegrateToMassInputs& out)
    {
        out.xGrid = m_xGrid;
        out.targetMass = m_targetMass;
        out.lowerY = m_lowerY;
        out.lowerMass = m_lowerMass;
        out.upperY = m_upperY;
        out.upperMass = m_upperMass;
        out.nGrid = m_nGrid;
        out.xData = m_xData;
        out.yData = m_yData;
        out.nData = m_nData;
        out.xBandwidth = m_xBandwidth;
        out.yBandwidth = m_yBandwidth;
        out.yMin = m_yMin;
        out.yMax = m_yMax;
    }

    U64 m_nData;
    const float* __restrict m_xData;
    const float* __restrict m_yData;
    U64 m_nGrid;
    float* __restrict m_xGrid;
    float* __restrict m_lowerY;
    float* __restrict m_lowerMass;
    float* __restrict m_upperY;
    float* __restrict m_upperMass;
    float* __restrict m_targetMass;
    float* __restrict m_totalMass;
    float m_xBandwidth;
    float m_yBandwidth;
    float m_xMin;
    float m_xMax;
    float m_yMin;
    float m_yMax;
};

} // namespace TimeHistogram
