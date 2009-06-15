#include "CmdLine.h"

#include "TensorFile.h"
#include "MemoryDataSet.h"
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc

//#include "BoostingTrainer.h"
=======
#include "TensorList.h"
#include "BoostingTrainer.h"
#include "FileListCmdOption.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
//#include "StumpTrainer.h"
#include "StumpMachine.h"
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
//#include "LutMachine.h"
//#include "LienhartMachine.h"
=======
#include "IntLutMachine.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
#include "Image.h"
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
#include "ipHaarLienhart.h"
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
#include "ipIntegral.h"

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
//#include "LBPStumpMachine.h"
//#include "ipLBP8R.h"
//#include "ipHaar.h"
//#include "ipIntegral.h"
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
#include "spCores.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    char* tensor_filename_target0;
    char* tensor_filename_target1;
=======

    char* modelfilename;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    int max_examples;
    int max_features;
    int n_classifiers;
    bool verbose;
    int width;
    int height;

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
    FileListCmdOption* tensor_files = new FileListCmdOption("Patterns", "Pattern List");
    tensor_files->isArgument(true);

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    cmd.info("Tensor read program");
=======
    cmd.info("Tests the performance of Cascade model");
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

    cmd.addText("\nArguments:");
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    cmd.addSCmdArg("tensor file for target 0", &tensor_filename_target0, "tensor file for target 0");
    cmd.addSCmdArg("tensor file for target 1", &tensor_filename_target1, "tensor file for target 1");
=======
    cmd.addCmdOption(tensor_files);
    cmd.addSCmdArg("model file name", &modelfilename, "Trained model file");

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
  //  cmd.addICmdOption("-wc", &n_classifiers, 10, "number of weak classifiers");
    cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
 //   cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    cmd.addICmdOption("-width", &width, -1, "image width");
    cmd.addICmdOption("-height", &height, -1, "image height");
=======

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    int n_examples;
    int n_features;

    //
    TensorFile tf0;
    CHECK_FATAL(tf0.openRead(tensor_filename_target0));

    //
    print("Reading tensor header file %s ...\n", tensor_filename_target0);
    const TensorFile::Header& header0 = tf0.getHeader();

    print("Tensor file:\n");
    print(" type:         [%s]\n", str_TensorTypeName[header0.m_type]);
    print(" n_tensors:    [%d]\n", header0.m_n_samples);
    print(" n_dimensions: [%d]\n", header0.m_n_dimensions);
    print(" size[0]:      [%d]\n", header0.m_size[0]);
    //print(" size[1]:      [%d]\n", header0.m_size[1]);
    //print(" size[2]:      [%d]\n", header0.m_size[2]);
    //print(" size[3]:      [%d]\n", header0.m_size[3]);

    //
    TensorFile tf1;
    CHECK_FATAL(tf1.openRead(tensor_filename_target1));

    //
    print("Reading tensor header file %s ...\n", tensor_filename_target1);
    const TensorFile::Header& header1 = tf1.getHeader();

    print("Tensor file:\n");
    print(" type:         [%s]\n", str_TensorTypeName[header1.m_type]);
    print(" n_tensors:    [%d]\n", header1.m_n_samples);
    print(" n_dimensions: [%d]\n", header1.m_n_dimensions);
    print(" size[0]:      [%d]\n", header1.m_size[0]);

    CHECK_FATAL(header0.m_type == header1.m_type);
    CHECK_FATAL(header0.m_n_dimensions == header1.m_n_dimensions);
    CHECK_FATAL(header0.m_size[0] == header1.m_size[0]);

    int n_examples_0 = header0.m_n_samples;
    int n_examples_1 = header1.m_n_samples;

    if (max_examples > 0)
    {
        if (max_examples < n_examples_0) n_examples_0 = max_examples;
        if (max_examples < n_examples_1) n_examples_1 = max_examples;
    }

    n_examples = n_examples_0 + n_examples_1;
    n_features = header0.m_size[0];
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    if (width > 0 && height > 0)
    {
        print("Width = %d\n", width);
        print("Height = %d\n", height);
        CHECK_FATAL(n_features == (width * height));
    }
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    Tensor *tensor = new FloatTensor(n_features);
    FloatTensor *unfoldtensor = new FloatTensor;
    ShortTensor *target0 = new ShortTensor(1);
    target0->fill(0);
    ShortTensor *target1 = new ShortTensor(1);
=======
    ShortTensor *target1 = manage(new ShortTensor(1));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    target1->fill(1);
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc

    //
    print("Building a MemoryDataSet ...\n");
    MemoryDataSet mdataset(n_examples, Tensor::Double, true, Tensor::Short);
    CHECK_FATAL(mdataset.getNoExamples() == n_examples);

    //
    print("Filling the MemoryDataSet ...\n");

    for (int i = 0 ; i < n_examples_0 ; i++)
=======
    DataSet *mdataset;
    TensorList *tensorList = manage(new TensorList());
    if (tensorList->process(tensor_files,target1,Tensor::Double)==false)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    {
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
        // memory allocation for the current tensor example
        if (width > 0 && height > 0) mdataset.getExample(i)->resize(width, height);
        else mdataset.getExample(i)->resize(n_features);

        // load a tensor from the file (assuming same type and size)
        tf0.load(*tensor);

        // copy the tensor read from the file into the current tensor example (thus supporting type conversion)
        if (width > 0 && height > 0)
        {
            unfoldtensor->unfold(tensor, 0, width, width);
            mdataset.getExample(i)->copy(unfoldtensor);
        }
        else mdataset.getExample(i)->copy(tensor);

        //
        mdataset.setTarget(i, target0);
=======
        print("Error in reading patterns - Tensor list\n");
        return 1;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    }
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
    mdataset = tensorList->getOutput();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    tf0.close();

    for (int i = n_examples_0 ; i < n_examples ; i++)
    {
        // memory allocation for the current tensor example
        if (width > 0 && height > 0) mdataset.getExample(i)->resize(width, height);
        else mdataset.getExample(i)->resize(n_features);

        // load a tensor from the file (assuming same type and size)
        tf1.load(*tensor);

        // copy the tensor read from the file into the current tensor example (thus supporting type conversion)
        if (width > 0 && height > 0)
        {
            unfoldtensor->unfold(tensor, 0, width, width);
            mdataset.getExample(i)->copy(unfoldtensor);
        }
        else mdataset.getExample(i)->copy(tensor);

        //
        mdataset.setTarget(i, target1);
    }
    tf1.close();
=======
    //
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    delete unfoldtensor;
    delete tensor;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    //
    if (verbose)
    {
        print("Reading the MemoryDataSet ...\n");
        for (int i = 0 ; i < mdataset.getNoExamples() ; i++)
        {
            mdataset.getExample(i)->sprint("%d",i);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
            mdataset.getTarget(i)->print("target");
        }
    }
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

    // Tprint(mdataset.getExample(2));

    //
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    print("Test Boosting ...\n");

    /*
    int w = 19;
    int h = 19;
=======
    print("Test Model ...\n");
    Tensor *st = mdataset->getExample(0);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    CHECK_FATAL(n_features == w*h);
=======
    int w = st->size(1);
    int h = st->size(0);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    ipLBP8R* ip_lbp = new ipLBP8R();
    ip_lbp->setR(2);
    */

    //
=======
    File ofile;
    ofile.open("scores.out","w");
    int n_examples = mdataset->getNoExamples();
    int count =0;
    ipIntegral *ipI = manage(new ipIntegral());
    DoubleTensor *temptensor = manage(new DoubleTensor());
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
    for (int e=0;e<n_examples;e++)
    {
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
        temptensor = (DoubleTensor*)mdataset->getExample(e);
        //   if(e==0)
        //           Tprint(temptensor);
        bool t = ipI->process(*temptensor);
        temptensor = (DoubleTensor*) &ipI->getOutput(0);
        //    if(e==0)
        //   Tprint(temptensor);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
        mdataset->getExample(e)->copy(temptensor);
    }
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    print("\n\n\n\n\n...................Loading the model\n");
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile("model.wsm");
=======
    CascadeMachine* cascade = manage((CascadeMachine*)Torch::loadMachineFromFile(modelfilename));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    if (cascade == 0)
    {
        print("ERROR: loading model [%s]!\n", "model.wsm");
        return 1;
    }

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
ipIntegral *ipI = new ipIntegral();
    DoubleTensor *st = new DoubleTensor();

    for (int e=0;e<n_examples;e++)
=======
    TensorRegion *tr = manage(new TensorRegion(0,0,h,w));
    for (int i=0;i<n_examples;i++)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    {
<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
=======
        Tensor *st = mdataset->getExample(i);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
        st = (DoubleTensor*)mdataset.getExample(e);
=======
        cascade->setRegion(*tr);
        if (cascade->forward(*st) == false)
        {
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
        bool t = ipI->process(*st);
        st = (DoubleTensor*) &ipI->getOutput(0);
=======
            delete cascade;
            return 1;
        }
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
        mdataset.getExample(e)->copy(st);
    }
TensorRegion *tr = new TensorRegion(0,0,height,width);
for(int i=0;i<n_examples;i++)
{
    Tensor *st = mdataset.getExample(i);
//print("size of tensor  %d\n",st->nDimension());
//Tprint(st);
    cascade->setRegion(*tr);
    if (cascade->forward(*st) == false)
    {
//			print("ERROR: failed to run the cascade on the image [%d/%d]!\n",
//				j + 1, n_samples);
        delete cascade;
        return 1;
=======
        ofile.printf("%g\n",cascade->getConfidence());
        // print("CONFIDENCE = %f\n", cascade->getConfidence());
        count += cascade->isPattern() ? 1 : 0;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc
    }

<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    print("CONFIDENCE = %f\n", cascade->getConfidence());
}
=======
    print("Number of Detection %d / %d \n",count,n_examples);
    //  print("Number of features counted %d   \n",cascade->countfeature);
    print("Performance %f \n",(count+0.0)/(n_examples+0.2));
    ofile.close();

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc


<<<<<<< HEAD:examples/boosting/boosttestintegral.cc
    //
    delete target1;
    delete target0;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boosttestintegral.cc


    // OK
    print("OK.\n");

    return 0;
}

