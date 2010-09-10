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

static double score_set(Torch::Machine &self, Torch::DataSet &set)
{
	double tot = 0.0;
	
	const long elements = set.getNoExamples();
	for (long index = 0; index < elements; index++) {

		// get the example
		Torch::Tensor *tensor = set.getExample(index);
		Torch::DoubleTensor ten;
		ten.copy(tensor);

		// calc the score for that sample
		tot += score(self, ten);
	}

	// mean score
	return tot /= elements;
}

void bind_machine_MultiVariateNormalDistribution()
{
	class_<Torch::MultiVariateNormalDistribution, bases<Torch::ProbabilityDistribution>, boost::noncopyable>("MultiVariateNormalDistribution", "", no_init)
		.def("score", &score, (arg("self"), arg("tensor")), "score a tensor against the mixture")
		.def("score", &score_set, (arg("self"), arg("data set")), "score a dataset against the mixture")
	;
}
