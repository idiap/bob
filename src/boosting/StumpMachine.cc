#include "StumpMachine.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

StumpMachine::StumpMachine() : spCoreMachine()
{
	//feature_id = -1;
	threshold = 0.0;
	direction = 0;
}

bool StumpMachine::forward(const Tensor& input)
{
   	if(core == NULL)
	{
	   	Torch::error("StumpMachine::forward() no core available.");
		return false;
	}

	//DoubleTensor* t_input = (DoubleTensor*) input;

	if(core->process(input) == false)
	{
	   	Torch::error("StumpMachine::forward() core failed.");
		return false;
	}

	DoubleTensor *core_t_output = (DoubleTensor*) &core->getOutput(0);

	double feature = core_t_output->get(0);

	double stump_output_;

	if(direction == 1)
	{
		if(feature >= threshold) stump_output_ = 1.0;
		else stump_output_ = -1.0;
	}
	else
	{
		if(feature < threshold) stump_output_ = 1.0;
		else stump_output_ = -1.0;
	}

	DoubleTensor* t_output = (DoubleTensor*) m_output;
	(*t_output)(0) = stump_output_;
   	
	return true;
}

bool StumpMachine::loadFile(File& file)
{
	return true;
}

bool StumpMachine::saveFile(File& file) const
{
   	print("StumpMachine::saveFile()\n");
	print("   threshold = %g\n", threshold);
	print("   direction = %d\n", direction);
	core->saveFile(file);
	
	return true;
}

void StumpMachine::setParams(int direction_, float threshold_)
{
	Torch::print("   StumpMachine::setParams()\n");

   	direction = direction_;
	threshold = threshold_;
}

StumpMachine::~StumpMachine()
{
}

}

