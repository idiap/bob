#ifndef _TORCH5SPRO_MACHINES_H_
#define _TORCH5SPRO_MACHINES_H_

//////////////////////////////////////////////////////////////////////////////////
// Include all the Machines, to make sure they are all registered to the MachineManager
// NB: the test programs should include this header instead of each Machine's header

// Gradient machines 1XXX
#define GRADIENT_MACHINE_ID 1000
#define LINEAR_GRADIENT_MACHINE_ID 1001
#define EXP_GRADIENT_MACHINE_ID 1002
#define LOG_GRADIENT_MACHINE_ID 1003
#define SIGMOID_GRADIENT_MACHINE_ID 1004
#define TANH_GRADIENT_MACHINE_ID 1005

#define MLP_GRADIENT_MACHINE_ID 1100

// Distribution machines 2XXX
#define PROBABILITY_DISTRIBUTION_MACHINE_ID 2000
#define MULTIVARIATE_GAUSSIAN_DISTRIBUTION_MACHINE_ID 2001
#define MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_ID 2002
#define MULTINOMIAL_DISTRIBUTION_MACHINE_ID 2003
#define HMM_DISTRIBUTION_MACHINE_ID 2004
#define BN_DISTRIBUTION_MACHINE_ID 2005

// Support Vector machines 3XXX

// Decision Tree machines 4XXX

// Normalisation machines 5XXX
#define MEANVAR_NORM_MACHINE_ID 5001

// Ensemble machines 6XXX ?
#define BAGGING_MACHINE_ID 1
#define CASCADE_MACHINE_ID 2
#define TREE_MACHINE_ID 3

// Stumps XXXX ?
#define STUMP_MACHINE_ID 20
#define INT_LUT_MACHINE_ID 21
#define REAL_LUT_MACHINE_ID 22


//
#include "CascadeMachine.h"
#include "StumpMachine.h"
#include "IntLutMachine.h"
#include "RealLutMachine.h"

#include "Exp.h"
#include "Log.h"
#include "Sigmoid.h"
#include "Linear.h"
#include "Tanh.h"
#include "MLP.h"

#include "MeanVarNorm.h"

//////////////////////////////////////////////////////////////////////////////////

#endif
