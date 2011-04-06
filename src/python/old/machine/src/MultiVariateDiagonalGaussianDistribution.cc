
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
