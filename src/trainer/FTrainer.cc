#include "FTrainer.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

FTrainer::FTrainer() : Trainer()
{
}

FTrainer::~FTrainer()
{
}
float FTrainer::forwardScan(const Tensor &example_,TensorRegion &tregion)
{
	print("Ftrainer::Should not be here\n");
	return 1.0;
}
}

