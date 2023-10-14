//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//
//#include "time_histogram.h"
//
//namespace py = pybind11;
//
//using U64 = uint64_t;
//
//py::array_t<float> compute_quantiles(py::array_t<float, py::array::c_style | py::array::forcecast> data)
//{
//  U64 nData = data.size() / 2;
//  const float* xData = data.data();
//  const float* yData = xData + nData;
//  TimeHistogram::TimeHistogram hist(xData, yData, nData);
//  const float quantiles[5] = { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };
//  py::array_t<float> result = py::array_t<float>(py::ssize_t_cast(5 * 1024), nullptr);
//  float *resultData = result.mutable_data();
//  hist.ComputeQuantiles(resultData, quantiles, 5);
//  return result;
//}
//
//PYBIND11_MODULE(time_histogram, m)
//{
//  m.doc() = "pybind for time histogram";
//  m.def("compute_quantiles", &compute_quantiles, "Compute quantile timeseries for a dataset, with optional kernel bandwidths in the x and y dimensions.");
//}