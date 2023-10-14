#pragma once

#include "integrator_common.h"

#define ALLOW_INTEGRATOR_IMPL
#if defined __AVX512F__ && defined __AVX512DQ__ && !defined FORCE_GENERIC && !defined FORCE_AVX2 || defined FORCE_AVX512

#include "integrator_avx512.h"

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
#undef ALLOW_INTEGRATOR_IMPL
