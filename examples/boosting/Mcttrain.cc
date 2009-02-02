#include <stdio.h>

#include "Tensor.h"
#include "FileBinDataSet.h"
#include "MctLbpTrainer.h"
#include "LBPMachine.h"
#include "ipLBP.h"

using namespace Torch;

int main()
{
	FileBinDataSet fb;
	fb.setData("bindata.list", Tensor::Short,Tensor::Short, 19, 19);

	MctLbpTrainer mct_trainer;
	mct_trainer.setData(&fb);
	mct_trainer.setnRounds(10,3);
	mct_trainer.setLbpParameters(LBPMachine::LBP8RAverageAddBit, 1);
	mct_trainer.train();

	return 0;
}
