#include "spCoreMachine.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

spCoreMachine::spCoreMachine() : Machine()
{
   	core = NULL;

	// Allocate the output
	m_output = new DoubleTensor(1);
}

void spCoreMachine::setCore(spCore *core_) 
{
	Torch::print("   spCoreMachine::setCore()\n");

	core = core_;
}


spCoreMachine::~spCoreMachine()
{
	delete m_output;
}

}

