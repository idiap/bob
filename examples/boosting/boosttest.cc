#include "CmdLine.h"

#include "TensorFile.h"
#include "MemoryDataSet.h"
#include "TensorList.h"
#include "BoostingTrainer.h"
#include "FileListCmdOption.h"
//#include "StumpTrainer.h"
#include "StumpMachine.h"
#include "IntLutMachine.h"
#include "Image.h"

#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"
#include "spCores.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options

    char* modelfilename;
    int max_examples;
    int max_features;
    int n_classifiers;
    bool verbose;
    int width;
    int height;

    FileListCmdOption* tensor_files = new FileListCmdOption("Patterns", "Pattern List");
    tensor_files->isArgument(true);

    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tests the performance of Cascade model");

    cmd.addText("\nArguments:");
    cmd.addCmdOption(tensor_files);
    cmd.addSCmdArg("model file name", &modelfilename, "Trained model file");


    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");


    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }



    ShortTensor *target1 = new ShortTensor(1);
    target1->fill(1);
 DataSet *mdataset;
    TensorList *tensorList = manage(new TensorList());
    if (tensorList->process(tensor_files,target1,Tensor::Double)==false)
    {
        print("Error in reading patterns - Tensor list\n");
        return 1;
    }
    mdataset = tensorList->getOutput();

    //




    // Tprint(mdataset.getExample(2));

    //
    print("Test Model ...\n");
    Tensor *st = mdataset->getExample(0);
Tprint((DoubleTensor*)st);
    int w = st->size(1);
    int h = st->size(0);

    File ofile;
    ofile.open("scores.out","w");
    int n_examples = mdataset->getNoExamples();
    int count =0;

    print("\n\n\n\n\n...................Loading the model\n");
    CascadeMachine* cascade = manage((CascadeMachine*)Torch::loadMachineFromFile(modelfilename));
    if (cascade == 0)
    {
        print("ERROR: loading model [%s]!\n", "model.wsm");
        return 1;
    }

    TensorRegion *tr = new TensorRegion(0,0,h,w);
    for (int i=0;i<n_examples;i++)
    {
        Tensor *st = mdataset->getExample(i);

        cascade->setRegion(*tr);
        if (cascade->forward(*st) == false)
        {

            delete cascade;
            return 1;
        }

        ofile.printf("%g\n",cascade->getConfidence());
        // print("CONFIDENCE = %f\n", cascade->getConfidence());
        count += cascade->isPattern() ? 1 : 0;
    }

    print("Number of Detection %d / %d \n",count,n_examples);
    //  print("Number of features counted %d   \n",cascade->countfeature);
    print("Performance %f \n",(count+0.0)/(n_examples+0.2));
    ofile.close();





    // OK
    print("OK.\n");

    return 0;
}

