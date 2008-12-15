#include <stdio.h>

#include "Tensor.h"
#include "FileBinDataSet.h"
#include "MctLbpTrainer.h"
#include "LBPMachine.h"
#include "ipLBP.h"
using namespace Torch;
int main()
{



    FileBinDataSet *FB = new FileBinDataSet();
   FB->setData("bindata.list",Tensor::Short,Tensor::Short,19,19);

    Tensor *T1 = new ShortTensor();
    T1= FB->getExample(3);
    print("dimension of T1 is %d\n",T1->nDimension());
    Tprint((ShortTensor*)T1);
    T1=FB->getTarget(3);
    Tprint((ShortTensor*)T1);
    print("%d\n",(*FB->short_example)(5,5,5));

    LBPMachine *lbp = new LBPMachine();
//    lbp->LBPType lbpt;
//    lbpt = LBP8RAverageAddBit;
    MctLbpTrainer *M1 = new MctLbpTrainer();
    M1->setData(FB);
    //M1->setnRounds(100,10);
    M1->setnRounds(10,3);
    M1->setLbpParameters(LBPMachine::LBP8RAverageAddBit,1);
    M1->train();

    delete FB;
    delete M1;
    delete lbp;

    return 0;


}
