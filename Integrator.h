#pragma once

#define ALLOW_INTEGRATOR_IMPL

#undef TIMEHISTOGRAM_USE_AVX512
#undef TIMEHISTOGRAM_USE_AVX2
#undef TIMEHISTOGRAM_USE_GENERIC

#if defined TIMEHISTOGRAM_FORCE_AVX512
#    define TIMEHISTOGRAM_USE_AVX512
#elif defined TIMEHISTOGRAM_FORCE_AVX2
#    define TIMEHISTOGRAM_USE_AVX2
#elif defined TIMEHISTOGRAM_FORCE_GENERIC
#    define TIMEHISTOGRAM_USE_GENERIC
#elif defined __AVX512F__ && defined __AVX512DQ__
#    define TIMEHISTOGRAM_USE_AVX512
#elif defined __AVX2__
#    define TIMEHISTOGRAM_USE_AVX2
#else
#    define TIMEHISTOGRAM_USE_GENERIC
#endif

#if defined TIMEHISTOGRAM_USE_AVX512
#    include "integrator_avx512.h"
#elif defined TIMEHISTOGRAM_USE_AVX2
#    include "integrator_avx2.h"
#else
#    include "integrator_generic.h"
#endif

#undef USE_AVX512
#undef USE_AVX2
#undef USE_GENERIC

#undef ALLOW_INTEGRATOR_IMPL
