/**
 * @file src/machine/machine.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include <cstdio>

#include "core/Object.h"
#include "core/File.h"
#include "core/Parameters.h"
#include "core/Tensor.h"

#include "machine/Machine.h"

using namespace boost::python;

static bool loadFileWithString(Torch::Machine &self, const char *filename)
{
	bool status = true;

	Torch::File file;
	file.open(filename, "r");

	if (file.isOpened()) {

		status = self.loadFile(file);
		
		file.close();
	} else {
		// TODO add missing warning message
		status = false;
	}


	return status;
}

void bind_machine_machine()
{
	class_<Torch::Machine, bases<Torch::Object>, boost::noncopyable>("Machine", "", no_init)
		.def("forward", &Torch::Machine::forward, (arg("self"), arg("tensor input")), "Process the input tensor")
		.def("loadFile", &loadFileWithString, (arg("self"), arg("file")), "loads a file")
		;
}
