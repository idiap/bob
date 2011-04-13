/**
 * @file src/python/math/src/norminv.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the inverse normal cumulative distribution into python
 */

#include <boost/python.hpp>

#include "math/norminv.h"

using namespace boost::python;

static const char* NORMSINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with zero mean and and unit variance.";
static const char* NORMINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with mean mu and standard deviation sigma.";

void bind_math_norminv()
{
  // Linear system solver
  def("normsinv", (double (*)(const double))&Torch::math::normsinv, (arg("p")), NORMSINV_DOC);
  def("norminv", (double (*)(const double, const double, const double))&Torch::math::norminv, (arg("p"), arg("mu"), arg("sigma")), NORMINV_DOC);
}

