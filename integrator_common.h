#pragma once

#include <cstdint>
#include <cmath>

using U32 = uint32_t;
using U64 = uint64_t;

namespace Integrator
{

static constexpr float WEND_C0 = 56.0f / 3.0f;
static constexpr float WEND_C1 = 88.0f / 3.0f;
static constexpr float WEND_C2 = 35.0f / 3.0f;

static constexpr float WEND_INT_C0 = 8.0f / 3.0f;
static constexpr float WEND_INT_C1 = 11.0f / 3.0f;
static constexpr float WEND_INT_C2 = 35.0f / 27.0f;


static constexpr float WEND_INT = 16.0f / 27.0f;
static constexpr float R_WEND_INT = 27.0f / 16.0f;

static constexpr float SIGN_MASK = -0.0f;

// Reference only. The C4 Wendland kernel, 1/3 * (35x^2 + 18x + 3) * (x-1)^6.
static float WendlandEvalScalar(float x)
{
    const float m = abs(x) - 1.0f;
    const float m3 = m * m * m;
    return (WEND_C2 * x * x + WEND_C1 * x + WEND_C0) * m3 * m3;
}

// Reference only. Integral of the C4 Wendland kernel from -1 to x.
// 35/27 (x-1)^9 + 11/3 (x-1)^8 + 8/3 (x-1)^7 + 8/27.
static float WendlandIntegralEvalScalar(float x)
{
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    x = abs(x);
    const float m = abs(x) - 1.0f;
    const float m3 = m * m * m;
    const float m6 = m3 * m3;
    float integral =
        (WEND_INT_C2 * m * m + WEND_INT_C1 * m + WEND_INT_C0) * m6 * m + 0.5f * WEND_INT;
    integral = sign * integral + 0.5f * WEND_INT;
    return integral;
}

struct LineIntegralsInputs
{
    const float* __restrict xGrid;
    U64 nGrid;
    const float* __restrict xData;
    U64 nData;
    float xBandwidth;
};

struct IntegrateToMassInputs
{
    const float* __restrict xGrid;
    const float* __restrict targetMass;
    float* __restrict lowerY;
    float* __restrict lowerMass;
    float* __restrict upperY;
    float* __restrict upperMass;
    U64 nGrid;
    const float* __restrict xData;
    const float* __restrict yData;
    U64 nData;
    float xBandwidth;
    float yBandwidth;
    float yMin;
    float yMax;
};
} // namespace Integrator
