#ifndef _TORCH5SPRO_SPCORE_MANAGER_H_
#define _TORCH5SPRO_SPCORE_MANAGER_H_

#include "Machine.h"
#include "spCore.h"
//#include "ipHaar.h"
#include "ipHaarLienhart.h"
#include "ipLBP4R.h"
#include "ipLBP8R.h"
//#include
namespace Torch {


//	//////////////////////////////////////////////////////////////////////////////////////
//	// Torch::spCoreMachine:
//	//      The output is a DoubleTensor!
//	//
//	//      NB: The ouput should be allocated and deallocated by each Machine implementation!
//	//
//	//	EACH MACHINE SHOULD REGISTER
//	//		==> MachineManager::GetInstance().add(new XXXMachine) <==
//	//	TO THE MACHINEMANAGER CLASS!!!
//	//
//	// TODO: doxygen header!
//	//////////////////////////////////////////////////////////////////////////////////////

	class spCoreManager : public Object
	{

	public:

		/// Constructor
		spCoreManager();

		/// Destructor
		virtual ~spCoreManager();

		///////////////////////////////////////////////////////////
		// Access functions

		spCore* getCore(int id);

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		//spCore *core;
	};
}

#endif
