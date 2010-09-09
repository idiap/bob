/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_machine_machine();
void bind_machine_ProbabilityDistribution();
void bind_machine_MultiVariateNormalDistribution();
void bind_machine_MultiVariateDiagonalGaussianDistribution();

BOOST_PYTHON_MODULE(libpytorch_machine) {
  scope().attr("__doc__") = "not available, nik lazy 44543543643";
  bind_machine_machine();
  bind_machine_ProbabilityDistribution();
  bind_machine_MultiVariateNormalDistribution();
  bind_machine_MultiVariateDiagonalGaussianDistribution();
}
