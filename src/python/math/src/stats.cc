/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 17 Jun 20:46:46 2011 
 *
 * @brief Python bindings to statistical methods
 */

#include <boost/python.hpp>
#include "math/stats.h"

using namespace boost::python;

static const char* SCATTER_DOC1 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). The resulting matrix 'S' is resized if necessary to accomodate the results.";

static const char* SCATTER_DOC2 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant also returns the sample means in 'M'. The resulting arrays 'M' and 'S' are resized if necessary to accomodate the results.";

static const char* SCATTER_DOC3 = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row). This variant returns the sample means and the scatter matrix in a tuple. If you are looking for efficiency, prefer the variants that receive the output variable as one of the input parameters. This version will allocate the resulting arrays 'M' and 'S' internally every time it is called.";

template <typename T>
static tuple scatter(const blitz::Array<T,2>& A) {
  blitz::Array<T,1> M;
  blitz::Array<T,2> S;
  Torch::math::scatter<T>(A,M,S);
  return make_tuple(M, S);
}

void bind_math_stats() {
  def("scatter", (void (*)(const blitz::Array<float,2>&, blitz::Array<float,2>&))&Torch::math::scatter<float>, (arg("A"),arg("S")), SCATTER_DOC1);
  def("scatter", (void (*)(const blitz::Array<double,2>&, blitz::Array<double,2>&))&Torch::math::scatter<double>, (arg("A"),arg("S")), SCATTER_DOC1);
  def("scatter", (void (*)(const blitz::Array<float,2>&, blitz::Array<float,1>&, blitz::Array<float,2>&))&Torch::math::scatter<float>, (arg("A"),arg("S")), SCATTER_DOC2);
  def("scatter", (void (*)(const blitz::Array<double,2>&, blitz::Array<double,1>&, blitz::Array<double,2>&))&Torch::math::scatter<double>, (arg("A"),arg("S")), SCATTER_DOC2);
  def("scatter", &scatter<float>, (arg("A")), SCATTER_DOC3);
  def("scatter", &scatter<double>, (arg("A")), SCATTER_DOC3);
}
