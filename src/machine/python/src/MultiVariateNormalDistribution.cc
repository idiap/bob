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

static double score(Torch::Machine &self, const Torch::Tensor &tensor)
{
	self.forward(tensor);
	const Torch::DoubleTensor &res = self.getOutput();
	double score = *((double *) res.dataR());

	return score;
}


void bind_machine_MultiVariateNormalDistribution()
{
	class_<Torch::MultiVariateNormalDistribution, bases<Torch::ProbabilityDistribution>, boost::noncopyable>("MultiVariateNormalDistribution", "", no_init)
		.def("score", &score, (arg("self"), arg("tensor")), "score a tensor against the mixture")
	;
}
