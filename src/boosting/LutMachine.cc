#include "LutMachine.h"
#include "spCore.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

LutMachine::LutMachine() : Machine()
{
   	n_bins = 0;
	lut = NULL;
	min = 0;
	max = 0;
}

bool LutMachine::forward(const Tensor& input)
{
   	if(lut == NULL)
	{
	   	Torch::error("LutMachine::forward() no LUT available.");
		return false;
	}

   	if(m_core == NULL)
	{
	   	Torch::error("LutMachine::forward() no core available.");
		return false;
	}

	if(m_core->process(input) == false)
	{
	   	Torch::error("LutMachine::forward() core failed.");
		return false;
	}

	DoubleTensor *core_t_output = (DoubleTensor*) &m_core->getOutput(0);

	double feature = core_t_output->get(0);

	double lut_output_;

	if(feature < min) lut_output_ = lut[0];
	else if(feature > max) lut_output_ = lut[n_bins-1];
	else
	{
		int index = (int) floor(n_bins * (feature - min) / (max - min));
		lut_output_ = lut[index];
	}

	DoubleTensor* t_output = (DoubleTensor*) m_output;
	(*t_output)(0) = lut_output_;

	return true;
}

bool LutMachine::loadFile(File& file)
{
	return true;
}

bool LutMachine::saveFile(File& file) const
{
	return true;
}

void LutMachine::setParams(double min_, double max_, int n_bins_, double *lut_)
{
	Torch::print("   LutMachine::setParams()\n");

	min = min_;
	max = max_;
	n_bins = n_bins_;
	lut = lut_;
}

LutMachine::~LutMachine()
{
}

}

