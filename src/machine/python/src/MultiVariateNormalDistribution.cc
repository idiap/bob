/**
 * @file src/machine/machine.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "machine/MultiVariateNormalDistribution.h"

using namespace boost::python;

void bind_machine_MultiVariateNormalDistribution()
{
	class_<Torch::MultiVariateNormalDistribution, bases<Torch::ProbabilityDistribution>, boost::noncopyable>("MultiVariateNormalDistribution", "", no_init)
	;
}
