#include "time_histogram.h"

#include <fstream>

using U64 = uint64_t;

void Profile();

int main(int argc, char** argv)
{
    //Profile();
    //return 0;

    if (argc < 2)
    {
        printf("Missing input file name!");
        return 1;
    }
    float xBandwidth = 0.0f;
    float yBandwidth = 0.0f;
    char* fileName = argv[1];
    if (argc >= 3)
    {
        xBandwidth = static_cast<float>(std::atof(argv[2]));
    }
    if (argc >= 4)
    {
        yBandwidth = static_cast<float>(std::atof(argv[3]));
    }
    char readBuffer[256] = {};
    std::ifstream inputFile;
    inputFile.open(fileName);
    if (!inputFile)
    {
        printf("Couldn't open file %s", fileName);
        return 1;
    }
    // discard 1st line
    inputFile.getline(readBuffer, 256);
    U64 nData = 0;
    U64 maxSize = 1024;
    float* xData = reinterpret_cast<float*>(_mm_malloc(2 * maxSize * sizeof(float), 4096));
    float* yData = xData + maxSize;
    while (inputFile.getline(readBuffer, 256, ','))
    {
        const float x = static_cast<float>(std::atof(readBuffer));
        inputFile.getline(readBuffer, 256);
        const float y = static_cast<float>(std::atof(readBuffer));
        xData[nData] = x;
        yData[nData] = y;
        ++nData;
        if (nData == maxSize)
        {
            const U64 newMaxSize = maxSize << 1;
            float* newXData =
                reinterpret_cast<float*>(_mm_malloc(2 * newMaxSize * sizeof(float), 4096));
            float* newYData = newXData + newMaxSize;
            memcpy(newXData, xData, maxSize * sizeof(float));
            memcpy(newYData, yData, maxSize * sizeof(float));
            _mm_free(xData);
            xData = newXData;
            yData = newYData;
            maxSize = newMaxSize;
        }
    }

    TimeHistogram::TimeHistogram hist(xData, yData, nData, xBandwidth, yBandwidth);
    constexpr U64 nGrid = 1024;
    const float* xGrid = hist.GetXGrid();
    float quantileData[5 * nGrid];
    constexpr float quantiles[] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
    hist.ComputeQuantiles(quantileData, quantiles, 5);

    std::ofstream output;
    output.open("QuantileData.csv");
    output << "xGrid,p10,p25,p50,p75,p90\n";
    for (U64 i = 0; i < nGrid; ++i)
    {
        output << xGrid[i] << "," << quantileData[i] << "," << quantileData[nGrid + i] << ","
               << quantileData[2 * nGrid + i] << "," << quantileData[3 * nGrid + i] << ","
               << quantileData[4 * nGrid + i] << "\n";
    }
    output.close();
    return 0;
}
