/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 17 Jun 20:46:46 2011 
 *
 * @brief Python bindings to statistical methods
 */

#include <boost/python.hpp>
#include "math/stats.h"

using namespace boost::python;

static const char* SCATTER_DOC = "Computes the scatter matrix of a 2D array considering data is organized column-wise (each sample is a column, each feature is a row).";

void bind_math_stats() {
  def("scatter", &Torch::math::scatter<float>, (arg("A"),arg("S")), 
      SCATTER_DOC);
  def("scatter", &Torch::math::scatter<double>, (arg("A"),arg("S")), 
      SCATTER_DOC);
}
