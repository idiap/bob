/**
 * @file src/machine/python/src/ProbabilityDistribution.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "machine/Machine.h"
#include "machine/ProbabilityDistribution.h"

using namespace boost::python;

void bind_machine_ProbabilityDistribution()
{
	class_<Torch::ProbabilityDistribution, bases<Torch::Machine>, boost::noncopyable>("ProbabilityDistribution", "", no_init)
	;
}
