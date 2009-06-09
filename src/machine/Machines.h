#ifndef _TORCH5SPRO_MACHINES_H_
#define _TORCH5SPRO_MACHINES_H_

//////////////////////////////////////////////////////////////////////////////////
// Include all the Machines, to make sure they are all registered to the MachineManager
// NB: the test programs should include this header instead of each Machine's header

#define CASCADE_MACHINE_ID 2
#define DIAG_GMM_MACHINE_ID 5
#define MLP_MACHINE_ID 6
#define STUMP_MACHINE_ID 20
#define INT_LUT_MACHINE_ID 21
#define REAL_LUT_MACHINE_ID 22

#include "CascadeMachine.h"
#include "StumpMachine.h"
#include "IntLutMachine.h"
#include "RealLutMachine.h"

//////////////////////////////////////////////////////////////////////////////////

#endif
