#include "time_histogram.h"

#include <chrono>
#include <fstream>
#include <immintrin.h>

using U64 = uint64_t;

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
  TimeHistogram::TimeHistogram hist(xData, yData, nData, 1.0f, 1.0f);
  float quantileData[3 * 1024];
  const float quantiles[] = { 0.0f, 0.5f, 1.0f };
  hist.ComputeQuantiles(quantileData, quantiles, 3);
  //printf("%.6f, %.6f, %.6f, %.6f\n", Integrator::WendlandIntegralEvalScalar(-1.0f), Integrator::WendlandIntegralEvalScalar(-0.0f), WendlandIntegralEvalScalar(0.0f), WendlandIntegralEvalScalar(1.0f));
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

  TimeHistogram::TimeHistogram hist(xData, yData, nData, 1.0f, 1.0f);
  float quantileData[5 * 1024];
  const float quantiles[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
  auto computeStart = clock.now();
  hist.ComputeQuantiles(quantileData, quantiles, 5);
  auto computeStop = clock.now();
  auto computeDuration = computeStop - computeStart;
  U64 computeNanos = computeDuration.count();
  U64 integralCount = 100ULL * 1024ULL * nData;
  printf("%llu integrals in %llu nanos\n", integralCount, computeNanos);
  double integralRate = double(integralCount) / double(computeNanos);
  printf("Rate: %f G /s\n", integralRate);
}

void RunFromFile(int argc, char** argv)
{
  if (argc < 2)
  {
    printf("Missing input file name!");
    return;
  }
  float xBandwidth = 1.0f;
  float yBandwidth = 1.0f;
  char* fileName = argv[1];
  if (argc >= 3)
  {
    xBandwidth = std::atof(argv[2]);
  }
  if (argc >= 4)
  {
    yBandwidth = std::atof(argv[3]);
  }
  char readBuffer[256] = {};
  std::ifstream inputFile;
  inputFile.open(fileName);
  if (!inputFile)
  {
    printf("Couldn't open file %s", fileName);
    return;
  }
  // discard 1st line
  inputFile.getline(readBuffer, 256);
  U64 nData = 0;
  U64 maxSize = 1024;
  float* xData = reinterpret_cast<float*>(_mm_malloc(maxSize * sizeof(float), 4096));
  float* yData = reinterpret_cast<float*>(_mm_malloc(maxSize * sizeof(float), 4096));
  while (inputFile.getline(readBuffer, 256, ','))
  {
    const float x = std::atof(readBuffer);
    inputFile.getline(readBuffer, 256);
    const float y = std::atof(readBuffer);
    xData[nData] = x;
    yData[nData] = y;
    ++nData;
    if (nData == maxSize)
    {
      const U64 newMaxSize = maxSize << 1;
      float* const newXData = reinterpret_cast<float*>(_mm_malloc(newMaxSize * sizeof(float), 4096));
      float* const newYData = reinterpret_cast<float*>(_mm_malloc(newMaxSize * sizeof(float), 4096));
      memcpy(newXData, xData, maxSize * sizeof(float));
      memcpy(newYData, yData, maxSize * sizeof(float));
      _mm_free(xData);
      _mm_free(yData);
      xData = newXData;
      yData = newYData;
      maxSize = newMaxSize;
    }
  }

  TimeHistogram::TimeHistogram hist(xData, yData, nData, xBandwidth, yBandwidth);
  hist.SetXBandwidth(xBandwidth);
  hist.SetYBandwidth(yBandwidth);
  constexpr U64 nGrid = 1024;
  const float* xGrid = hist.GetXGrid();
  float quantileData[5 * nGrid];
  const float quantiles[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
  hist.ComputeQuantiles(quantileData, quantiles, 5);
  
  std::ofstream output;
  output.open("QuantileData.csv");
  output << "xGrid,p10,p25,p50,p75,p90\n";
  for (U64 i = 0; i < nGrid; ++i)
  {
    output << xGrid[i] << "," << quantileData[i] << "," << quantileData[nGrid + i] << "," << quantileData[2*nGrid + i] << "," << quantileData[3*nGrid + i] << "," << quantileData[4*nGrid + i] << "\n";
  }
  output.close();
}

int main(int argc, char** argv)
{
  //Profile();
  RunFromFile(argc, argv);
  return 0;
}