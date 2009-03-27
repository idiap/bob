#ifndef _TORCH5SPRO_SPCORE_MACHINE_H_
#define _TORCH5SPRO_SPCORE_MACHINE_H_

#include "Machine.h"
#include "spCore.h"

namespace Torch {


	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::spCoreMachine:
	//      The output is a DoubleTensor!
	//
	//      NB: The ouput should be allocated and deallocated by each Machine implementation!
	//
	//	EACH MACHINE SHOULD REGISTER
	//		==> MachineManager::GetInstance().add(new XXXMachine) <==
	//	TO THE MACHINEMANAGER CLASS!!!
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class spCoreMachine : public Machine
	{

	public:

		/// Constructor
		spCoreMachine();

		/// Destructor
		virtual ~spCoreMachine();

		///////////////////////////////////////////////////////////
		// Access functions

		void setCore(spCore *core_);
		
		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		spCore *core;
	};
}

#endif
