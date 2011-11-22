/**
 * @file python/old/machine/src/MultiVariateDiagonalGaussianDistribution.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

/**
 * @file src/python/machine/src/MultiVariateDiagonalGaussianDistribution.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include <exception>

#include "machine/MultiVariateDiagonalGaussianDistribution.h"

using namespace boost::python;

typedef Torch::MultiVariateDiagonalGaussianDistribution TGMM;

static boost::shared_ptr<TGMM> createWithString(const char *filename)
{
	boost::shared_ptr<TGMM> machine(new TGMM());

	Torch::File file;
	file.open(filename, "r");
	bool status;

	if (file.isOpened()) {

		status = machine->loadFile(file);
		file.close();

		return machine;
	} else {
		// TODO error message
		status = false;
		throw std::exception();
		
	}

	if (status)
		return machine;
	else
		return boost::shared_ptr<TGMM>();
}

void bind_machine_MultiVariateDiagonalGaussianDistribution()
{
	class_<TGMM, boost::shared_ptr<TGMM>, bases<Torch::MultiVariateNormalDistribution>, boost::noncopyable>("MultiVariateDiagonalGaussianDistribution", "")
		.def("__init__", make_constructor(createWithString))

	;
}
