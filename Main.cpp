#include "TimeHistogram.h"

#include <chrono>

void TestCorrectness()
{
  constexpr U64 nData = 16;
  float xData[nData];
  float yData[nData];
  for (U64 i = 0; i < nData; ++i)
  {
    xData[i] = float(i) / 16.0f;
    yData[i] = 4.0f * xData[i] * (1.0f - xData[i]);
  }
  TimeHistogram hist(xData, yData, nData);
  float quantileData[3 * 1024];
  const float quantiles[] = { 0.0f, 0.5f, 1.0f };
  hist.ComputeQuantiles(quantileData, quantiles, 3);
  printf("%.6f, %.6f, %.6f, %.6f\n", WendlandIntegralEvalScalar(-1.0f), WendlandIntegralEvalScalar(-0.0f), WendlandIntegralEvalScalar(0.0f), WendlandIntegralEvalScalar(1.0f));
}

void Profile()
{
  std::chrono::high_resolution_clock clock;
  auto startData = clock.now();

  constexpr U64 nData = 1048576;
  float* xData;
  float* yData;
  xData = reinterpret_cast<float*>(_mm_malloc(2 * nData * sizeof(float), 4096));
  yData = xData + nData;

  for (U64 i = 0; i < nData; ++i)
  {
    xData[i] = -123.45f + 0.00076567f * float(i);
    yData[i] = sinf(xData[i]);
  }

  auto stopData = clock.now();
  auto dataDuration = stopData - startData;
  U64 dataNanos = dataDuration.count();
  printf("Data generated in %llu nanos\n", dataNanos);

  TimeHistogram hist(xData, yData, nData);
  float quantileData[5 * 1024];
  const float quantiles[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
  auto computeStart = clock.now();
  hist.ComputeQuantiles(quantileData, quantiles, 5);
  auto computeStop = clock.now();
  auto computeDuration = computeStop - computeStart;
  U64 computeNanos = computeDuration.count();
  U64 integralCount = 20ULL * 1024ULL * nData;
  printf("%llu integrals in %llu nanos\n", integralCount, computeNanos);
  double integralRate = double(integralCount) / double(computeNanos);
  printf("Rate: %f G /s\n", integralRate);
}

int main()
{
  Profile();
  return 0;
}