
/**
 * @file src/machine/machine.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "machine/MultiVariateDiagonalGaussianDistribution.h"

using namespace boost::python;

void bind_machine_MultiVariateDiagonalGaussianDistribution()
{
	class_<Torch::MultiVariateDiagonalGaussianDistribution, bases<Torch::MultiVariateNormalDistribution>, boost::noncopyable>("MultiVariateDiagonalGaussianDistribution", "")
	;
}
