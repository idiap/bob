/**
 * @file python/old/machine/src/MultiVariateNormalDistribution.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the Scanner to python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
