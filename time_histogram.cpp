#include "time_histogram.h"

namespace TimeHistogram
{
    TimeHistogram::TimeHistogram(const float* x, const float* y, const U64 n, const float xBandwidth_,
                                 const float yBandwidth_)
        : m_nData{n}
          , m_xData{x}
          , m_yData{y}
          , m_nGrid{1024}
          , m_xBandwidth{xBandwidth_}
          , m_yBandwidth{yBandwidth_}
    {
        m_xGrid = reinterpret_cast<float*>(_mm_malloc(sizeof(float) * m_nGrid * 7, 4096));
        m_lowerY = m_xGrid + m_nGrid;
        m_lowerMass = m_xGrid + 2 * m_nGrid;
        m_upperY = m_xGrid + 3 * m_nGrid;
        m_upperMass = m_xGrid + 4 * m_nGrid;
        m_targetMass = m_xGrid + 5 * m_nGrid;
        m_totalMass = m_xGrid + 6 * m_nGrid;
        Integrator::FindDataBounds(m_xMin, m_xMax, m_yMin, m_yMax, x, y, n);
        SetXGridRange(m_xMin, m_xMax);
        Integrator::LineIntegralsInputs in;
        FillLineIntegralsInputs(in);
        LineIntegrals(m_totalMass, in);
    }

    TimeHistogram::~TimeHistogram()
    {
        _mm_free(m_xGrid);
    }

    void TimeHistogram::ComputeQuantiles(
        float* const __restrict out,
        const float* const __restrict quantiles,
        const U64 nQuantiles)
    {
        const float loY = m_yMin - m_yBandwidth;
        const float hiY = m_yMax + m_yBandwidth;

        Integrator::IntegrateToMassInputs in;
        FillIntegrateToMassInputs(in);
        for (U64 i = 0; i < nQuantiles; ++i)
        {
            const float quantile = quantiles[i];
            memcpy(m_upperMass, m_totalMass, m_nGrid * sizeof(float));
            for (U64 j = 0; j < m_nGrid; ++j)
            {
                m_targetMass[j] = m_totalMass[j] * quantile;
                m_lowerY[j] = loY;
                m_lowerMass[j] = 0.0f;
                m_upperY[j] = hiY;
            }

            IntegrateToMass(out + i * m_nGrid, in);
        }
    }
}
