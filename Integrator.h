#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <cstdint>
#include <cmath>

using U32 = uint32_t;
using U64 = uint64_t;

static constexpr float WEND_C0 = 1.0f;
static constexpr float WEND_C1 = 6.0f;
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
  float I = (WEND_INT_C2 * m * m + WEND_INT_C1 * m + WEND_INT_C0) * m6 * m + 0.5 * WEND_INT;
  I = sign * I + 0.5 * WEND_INT;
  return I;
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

#if defined __AVX512F__ && defined __AVX512DQ__ && !defined FORCE_GENERIC && !defined FORCE_AVX2 || defined FORCE_AVX512

#include "Integrator_AVX512.h"

#elif defined(__AVX2__) && !defined FORCE_GENERIC || defined FORCE_AVX2
#include <immintrin.h>

// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
void LineIntegrals(float* const __restrict out, const LineIntegralsInputs& in)
{

}


// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
void IntegrateToMass(float* const __restrict out, const IntegrateToMassInputs& in)
{

}

#else // Generic Arch

// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
void LineIntegrals(float* const __restrict out, const LineIntegralsInputs& in)
{

}


// PRECONDITION: nGrid is divisible by 16, nGrid > 0, nData > 0.
void IntegrateToMass(float* const __restrict out, const IntegrateToMassInputs& in)
{

}

#endif

#endif
