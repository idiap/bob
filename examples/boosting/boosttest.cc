#include "CmdLine.h"

#include "TensorFile.h"
#include "MemoryDataSet.h"

#include "BoostingTrainer.h"
//#include "StumpTrainer.h"
#include "StumpMachine.h"
#include "LBPMachine.h"
#include "Image.h"

//#include "LBPStumpMachine.h"
//#include "ipLBP8R.h"
#include "ipHaar.h"
//#include "ipIntegral.h"
#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"

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
    bool verbose;
    int width;
    int height;

    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tensor read program");

    cmd.addText("\nArguments:");
    cmd.addSCmdArg("tensor file for target 0", &tensor_filename_target0, "tensor file for target 0");
    cmd.addSCmdArg("tensor file for target 1", &tensor_filename_target1, "tensor file for target 1");

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    //  cmd.addICmdOption("-wc", &n_classifiers, 10, "number of weak classifiers");
    cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
    // cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    cmd.addICmdOption("-width", &width, -1, "image width");
    cmd.addICmdOption("-height", &height, -1, "image height");

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

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

    if (width > 0 && height > 0)
    {
        print("Width = %d\n", width);
        print("Height = %d\n", height);
        CHECK_FATAL(n_features == (width * height));
    }

    Tensor *tensor = new FloatTensor(n_features);
    FloatTensor *unfoldtensor = new FloatTensor;
    ShortTensor *target0 = new ShortTensor(1);
    target0->fill(0);
    ShortTensor *target1 = new ShortTensor(1);
    target1->fill(1);

    //
    print("Building a MemoryDataSet ...\n");
    MemoryDataSet mdataset(n_examples, Tensor::Double, true, Tensor::Short);
    CHECK_FATAL(mdataset.getNoExamples() == n_examples);

    //
    print("Filling the MemoryDataSet ...\n");

    for (int i = 0 ; i < n_examples_0 ; i++)
    {
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
    }

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

    delete unfoldtensor;
    delete tensor;

    //
    if (verbose)
    {
        print("Reading the MemoryDataSet ...\n");
        for (int i = 0 ; i < mdataset.getNoExamples() ; i++)
        {
            mdataset.getExample(i)->sprint("%d",i);

            mdataset.getTarget(i)->print("target");
        }
    }

    // Tprint(mdataset.getExample(2));

    //
    print("Test Boosting ...\n");

    /*
    int w = 19;
    int h = 19;

    CHECK_FATAL(n_features == w*h);

    ipLBP8R* ip_lbp = new ipLBP8R();
    ip_lbp->setR(2);
    */

    //



    print("\n\n\n\n\n...................Loading the model\n");
    CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile("model.wsm");
    if (cascade == 0)
    {
        print("ERROR: loading model [%s]!\n", "model.wsm");
        return 1;
    }

    TensorRegion *tr = new TensorRegion(0,0,height,width);
    for (int i=0;i<n_examples;i++)
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
        }

        print("CONFIDENCE = %f\n", cascade->getConfidence());
    }


    //
    delete target1;
    delete target0;


    // OK
    print("OK.\n");

    return 0;
}

