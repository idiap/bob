#ifndef _TORCH5SPRO_MACHINES_H_
#define _TORCH5SPRO_MACHINES_H_

//////////////////////////////////////////////////////////////////////////////////
// Include all the Machines, to make sure they are all registered to the MachineManager
// NB: the test programs should include this header instead of each Machine's header

#include "LBPMachine.h"
#include "CascadeMachine.h"

namespace Torch
{
        ///////////////////////////////////////////////////////////////////////////

        // Load a generic machine from some file
        // Returns <0/NULL> if some error,
        //      otherwise you are responsible for deallocating the Machine
        Machine*                loadMachineFromFile(const char* filename);

        ///////////////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////////////

#endif
