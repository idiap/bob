#ifndef MCTLBPTRAINER_INC
#define MCTLBPTRAINER_INC

//#include "Trainer.h"
#include "Tensor.h"
#include "FileBinDataSet.h"
#include "Object.h"
#include "ipLBP.h"
#include "ipLBP8R.h"
#include "ipLBP4R.h"
#include <cassert>
#include <stdio.h>
#include <math.h>
#include "LBPMachine.h"
#include "CascadeMachine.h"

namespace Torch
{

    class LBPMachine;
    class CascadeMachine;
    class MctLbpTrainer : public Object
    {
    public:
///Constructor
        MctLbpTrainer();
       void setData(FileBinDataSet *data_);

///Destructor
        ~MctLbpTrainer();


        void train();

        void setnRounds(int n_rounds_,int n_features_);


       // void testdata(myDataSet *data_);

        void setLbpParameters(LBPMachine::LBPType lbptype,int rad_);


    protected:
        FileBinDataSet *data;
        ipLBP *ip_lbp;
        LBPMachine *LBPM;
        int rad;
        int width;
        int height;
        LBPMachine::LBPType lbp_type;



        // here after selecting the nfeatures, boosting is continued using these nfeatures for nrounds
        int n_rounds;
        int n_features;
        int c_features;

        int current_round;

        int n_examples;

        int sizeof_LUT;

        IntTensor *tempLUT;

        DoubleTensor *LUT; // will store final LUT
        IntTensor *b_pixellocation; // stores best x and y locations

        IntTensor *LBPvalue; // stores the computed LBP value
        DoubleTensor *weights;

        DoubleTensor *it_weights; // weight for each iteration
        IntTensor *it_b_pix;
        IntTensor *it_LUT; // each iteration binary LUT

        IntTensor *ShuffleData; // stores the example location that will be used in current iteration

        void randw(); // this is to shuffle the examples based on the weights
        void precompute_LBPfeatures();
        void findbestfeature();
        void findweight();
        void initializeWeights();
        void finalLUT();
        void saveModel();







    };


}
#endif
