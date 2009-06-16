#include "CmdLine.h"


#include "TensorList.h"
#include "MemoryDataSet.h"

#include "BoostingRoundLBPTrainer.h"
#include "LBPRoundTrainer.h"
#include "IntLutMachine.h"
#include "Image.h"

//#include "LBPStumpMachine.h"
#include "ipLBP8R.h"
#include "ipLBP4R.h"
#include "ipLBP.h"
//#include "ipHaarLienhart.h"
//#include "ipIntegral.h"
#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"
#include "spCore.h"
#include "spCores.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char* tensor_filename_target0;
    char* tensor_filename_target1;
    int max_examples;
    int max_features;
    int n_classifiers;
    int n_rounds;
    bool verbose;
    int width;
    int height;

    FileListCmdOption* p_tensor_files = new FileListCmdOption("positivePatterns", "+ve training patterns");
    p_tensor_files->isArgument(true);

    FileListCmdOption* n_tensor_files = new FileListCmdOption("negativePatterns", "-ve training patterns");
    n_tensor_files->isArgument(true);


    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tensor read program");

    cmd.addText("\nArguments:");
    cmd.addCmdOption(p_tensor_files);
    cmd.addCmdOption(n_tensor_files);


    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    cmd.addICmdOption("-wc", &n_classifiers, 5, "number of weak classifiers");
    cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");
    // cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
    // cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    // cmd.addICmdOption("-width", &width, 19, "image width");
    //  cmd.addICmdOption("-height", &height, 19, "image height");

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

    int n_examples;
    int n_features;


    ShortTensor *target1 = manage(new ShortTensor(1));
    target1->fill(1);
    ShortTensor *target0 = manage(new ShortTensor(1));
    target0->fill(0);
    DataSet *pDataSet;
    DataSet *nDataSet;
    TensorList *tensorList_p= manage(new TensorList()); //for positive
    TensorList *tensorList_n= manage(new TensorList()); //for validation


    if (tensorList_p->process(p_tensor_files,target1,Tensor::Double)==false)
    {
        print("Error in reading +ve training patterns - Tensor list\n");
        return 1;
    }
    pDataSet = tensorList_p->getOutput();


    if (tensorList_n->process(n_tensor_files,target0,Tensor::Double)==false)
    {
        print("Error in reading -ve patterns - Tensor list\n");
        return 1;
    }
    nDataSet = tensorList_n->getOutput();


    Tensor *st = pDataSet->getExample(0);
//print("dimension %d %d\n",st->nDimension(),st->size(0));
    width = st->size(1);
    height = st->size(0);
    n_examples = pDataSet->getNoExamples() + nDataSet->getNoExamples();
    int pexamples = pDataSet->getNoExamples();
    MemoryDataSet *m_dataset = manage(new MemoryDataSet(n_examples, Tensor::Double, true, Tensor::Short));
    Tensor *example;
    for (int i=0;i<pDataSet->getNoExamples();i++)
    {
        m_dataset->getExample(i)->resize(height, width);
        example = pDataSet->getExample(i);
        m_dataset->getExample(i)->copy(example);
        m_dataset->setTarget(i, target1);
    }
    for (int i=0;i<nDataSet->getNoExamples();i++)
    {
        m_dataset->getExample(i+pexamples)->resize(height, width);
        example = nDataSet->getExample(i);
        m_dataset->getExample(i+pexamples)->copy(example);
        m_dataset->setTarget(i+pexamples, target0);
    }

    print("Prepare Boosting ...\n");

    if (width>0 && height >0)
    {
        n_features =0;

        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
                n_features++;

    }

    print("Number of feature %d\n",n_features);


    spCore **weakfeatures = manage_array(new spCore* [n_features]);


    bool t;
    TensorSize *modelsize = manage(new TensorSize(height,width));
    TensorRegion *tregion = manage(new TensorRegion(0,0,height,width));
    // print("width and height %d %d\n", width, height);
    if (width>0 && height >0)
    {
        int i =0;

        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
            {
                weakfeatures[i] = manage(new ipLBP8R);//width, height);

                t= ((ipLBP *)weakfeatures[i])->setBOption("ToAverage", true);
                t= ((ipLBP*)weakfeatures[i])->setBOption("AddAvgBit", true);
                t= ((ipLBP *)weakfeatures[i])->setBOption("Uniform", false);
                t= ((ipLBP *)weakfeatures[i])->setBOption("RotInvariant", false);
                ((ipLBP  *)weakfeatures[i])->setModelSize(*modelsize);
                ((ipLBP  *)weakfeatures[i])->setRegion(*tregion);
                t=((ipLBP *)weakfeatures[i])->setXY(x_, y_);
                //  print("iteration number %d\n",i);
                i++;

            }

    }


    else
    {
        if (max_features > 0 && max_features < n_features) n_features = max_features;

        weakfeatures = manage_array(new spCore* [n_features]);
        for (int i = 0 ; i < n_features ; i++)
            weakfeatures[i] = manage(new spDeltaOne(i));
    }

    WeakLearner **lbp_trainers = manage_array(new WeakLearner*[n_rounds]);
    IntLutMachine **lbp_machines = manage_array(new IntLutMachine*[n_rounds]);
    for ( int i = 0; i < n_rounds; i++)
    {
        lbp_machines[i] = manage(new IntLutMachine);

        lbp_trainers[i] = manage(new LBPRoundTrainer(lbp_machines[i], n_features, weakfeatures));
    }

    BoostingRoundLBPTrainer boosting;
    boosting.setBOption("boosting_by_sampling", true);
    boosting.setIOption("number_of_rounds", n_rounds);
    boosting.setWeakLearners(n_classifiers, lbp_trainers);
    boosting.setData(m_dataset);
    boosting.train();


    //Saving the model.

    CascadeMachine* CM = manage(new CascadeMachine());
    CM->resize(1);
    CM->setSize(*modelsize);
    CM->resize(0,n_classifiers);

    for (int i=0;i<n_classifiers;i++)
    {
        t= CM->setMachine(0,i,lbp_machines[i]);
        t= CM->setWeight(0,i,1);


    }

    t=CM->setThreshold(0,0.0);

    File f1;
    t=f1.open("model.wsm","w");
    t=CM->saveFile(f1);
    f1.close();

    // OK
    print("OK.\n");

    return 0;
}

