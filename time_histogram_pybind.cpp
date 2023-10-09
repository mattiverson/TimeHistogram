#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(time_histogram, m)
{
  m.doc() = "pybind for time histogram";
  m.def("compute_quantiles", &compute_quantiles, "Compute quantile timeseries for a dataset, with optional kernel bandwidths in the x and y dimensions.");
}