#include "CmdLine.h"

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
#include "TensorFile.h"
=======
#include "TensorList.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
#include "MemoryDataSet.h"

#include "BoostingTrainer.h"
#include "StumpTrainer.h"
#include "StumpMachine.h"
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
#include "Image.h"

//#include "LBPStumpMachine.h"
//#include "ipLBP8R.h"
=======
#include "WeakLearner.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
#include "ipHaarLienhart.h"
#include "ipIntegral.h"
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
#include "spDeltaOne.h"
#include "CascadeMachine.h"
#include "Machines.h"
<<<<<<< HEAD:examples/boosting/boostingintegral.cc

=======
#include "spCore.h"
#include "spCores.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
using namespace Torch;


int main(int argc, char* argv[])
{
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char* tensor_filename_target0;
    char* tensor_filename_target1;
    int max_examples;
    int max_features;
    int n_classifiers;
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
    // int n_rounds;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    bool verbose;
    int width;
    int height;

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
    FileListCmdOption* p_tensor_files = new FileListCmdOption("positivePatterns", "+ve training patterns");
    p_tensor_files->isArgument(true);

    FileListCmdOption* n_tensor_files = new FileListCmdOption("negativePatterns", "-ve training patterns");
    n_tensor_files->isArgument(true);


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tensor read program");

    cmd.addText("\nArguments:");
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    cmd.addSCmdArg("tensor file for target 0", &tensor_filename_target0, "tensor file for target 0");
    cmd.addSCmdArg("tensor file for target 1", &tensor_filename_target1, "tensor file for target 1");
=======
    cmd.addCmdOption(p_tensor_files);
    cmd.addCmdOption(n_tensor_files);

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    cmd.addICmdOption("-wc", &n_classifiers, 10, "number of weak classifiers");
    cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
    cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    cmd.addICmdOption("-width", &width, -1, "image width");
    cmd.addICmdOption("-height", &height, -1, "image height");
=======
    cmd.addICmdOption("-wc", &n_classifiers, 5, "number of weak classifiers");
    // cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

    int n_examples;
    int n_features;

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
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
=======
    ShortTensor *target1 = manage(new ShortTensor(1));
    target1->fill(1);
    ShortTensor *target0 = manage(new ShortTensor(1));
    target0->fill(0);
    DataSet *pDataSet;
    DataSet *nDataSet;
    TensorList *tensorList_p= manage(new TensorList()); //for positive
    TensorList *tensorList_n= manage(new TensorList()); //for validation
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    n_examples = n_examples_0 + n_examples_1;
    n_features = header0.m_size[0];
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    if (width > 0 && height > 0)
=======
    if (tensorList_p->process(p_tensor_files,target1,Tensor::Double)==false)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        print("Width = %d\n", width);
        print("Height = %d\n", height);
        CHECK_FATAL(n_features == (width * height));
=======
        print("Error in reading +ve training patterns - Tensor list\n");
        return 1;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
    pDataSet = tensorList_p->getOutput();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
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
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    //
    print("Filling the MemoryDataSet ...\n");

    for (int i = 0 ; i < n_examples_0 ; i++)
=======
    if (tensorList_n->process(n_tensor_files,target0,Tensor::Double)==false)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        // memory allocation for the current tensor example
        if (width > 0 && height > 0) mdataset.getExample(i)->resize(height, width);
        else mdataset.getExample(i)->resize(n_features);

        // load a tensor from the file (assuming same type and size)
        tf0.load(*tensor);

        // copy the tensor read from the file into the current tensor example (thus supporting type conversion)
        if (width > 0 && height > 0)
        {
            unfoldtensor->unfold(tensor, 0, height, width);
            mdataset.getExample(i)->copy(unfoldtensor);
        }
        else mdataset.getExample(i)->copy(tensor);

        //
        mdataset.setTarget(i, target0);
=======
        print("Error in reading -ve patterns - Tensor list\n");
        return 1;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
    nDataSet = tensorList_n->getOutput();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    tf0.close();
=======
Tensor *st = pDataSet->getExample(0);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    for (int i = n_examples_0 ; i < n_examples ; i++)
    {
        // memory allocation for the current tensor example
        if (width > 0 && height > 0) mdataset.getExample(i)->resize(height, width);
        else mdataset.getExample(i)->resize(n_features);

        // load a tensor from the file (assuming same type and size)
        tf1.load(*tensor);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        // copy the tensor read from the file into the current tensor example (thus supporting type conversion)
        if (width > 0 && height > 0)
        {
            unfoldtensor->unfold(tensor, 0, height, width);
            mdataset.getExample(i)->copy(unfoldtensor);
        }
        else mdataset.getExample(i)->copy(tensor);
=======
    width = st->size(1);
    height = st->size(0);
    n_examples = pDataSet->getNoExamples() + nDataSet->getNoExamples();
    int pexamples = pDataSet->getNoExamples();
    MemoryDataSet *m_dataset = manage(new MemoryDataSet(n_examples, Tensor::Double, true, Tensor::Short));
    Tensor *example;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        //
        mdataset.setTarget(i, target1);
=======
    for (int i=0;i<pDataSet->getNoExamples();i++)
    {
        m_dataset->getExample(i)->resize(height, width);
        example = pDataSet->getExample(i);
        m_dataset->getExample(i)->copy(example);
        m_dataset->setTarget(i, target1);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    tf1.close();

    delete unfoldtensor;
    delete tensor;

    //
    if (verbose)
=======
    for (int i=0;i<nDataSet->getNoExamples();i++)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        print("Reading the MemoryDataSet ...\n");
        for (int i = 0 ; i < mdataset.getNoExamples() ; i++)
        {
            mdataset.getExample(i)->sprint("%d",i);

            mdataset.getTarget(i)->print("target");
        }
=======
        m_dataset->getExample(i+pexamples)->resize(height, width);
        example = nDataSet->getExample(i);
        m_dataset->getExample(i+pexamples)->copy(example);
        m_dataset->setTarget(i+pexamples, target0);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    // Tprint(mdataset.getExample(2));
=======
    print("Prepare Boosting ...\n");

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    ipIntegral *ipI = new ipIntegral();
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    DoubleTensor *st;// = new DoubleTensor();
=======
    DoubleTensor *temptensor = new DoubleTensor();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

    for (int e=0;e<n_examples;e++)
    {

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        st = (DoubleTensor*)mdataset.getExample(e);

        bool t = ipI->process(*st);
        st = (DoubleTensor*) &ipI->getOutput(0);
=======
        temptensor = (DoubleTensor*)m_dataset->getExample(e);
        if(e==0)
                Tprint(temptensor);
        bool t = ipI->process(*temptensor);
        temptensor = (DoubleTensor*) &ipI->getOutput(0);
        if(e==0)
        Tprint(temptensor);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        mdataset.getExample(e)->copy(st);
=======
        m_dataset->getExample(e)->copy(temptensor);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    print("Prepare Boosting ...\n");

    /*
    int w = 19;
    int h = 19;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    CHECK_FATAL(n_features == w*h);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    ipLBP8R* ip_lbp = new ipLBP8R();
    ip_lbp->setR(2);
    */
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//spCore **weakfeatures =NULL;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    if (width>0 && height >0)
    {
        n_features =0;
        //Edge feature 1

        for (int w_=2;w_<width;w_=w_+2)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;


        //Edge feature 2
        for (int w_=1;w_<width;w_++)
            for (int h_ =2;h_<height;h_=h_+2)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;


        //Line feature 1

        for (int w_=3;w_<width;w_=w_+3)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;


        //Line feature 2
        for (int w_=1;w_<width;w_++)
            for (int h_ =3;h_<height;h_=h_+3)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;

        //Line feature 3
        for (int w_=1;w_<width;w_++)
            for (int h_ =4;h_<height;h_=h_+4)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;

        //Line feature 4
        for (int w_=4;w_<width;w_=w_+4)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;

        for (int w_=3;w_<width;w_=w_+3)
            for (int h_ =3;h_<height;h_=h_+3)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;

        for (int w_=2;w_<width;w_=w_+2)
            for (int h_ =2;h_<height;h_=h_+2)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        n_features++;

    }

    print("Number of feature %d\n",n_features);


    spCore **weakfeatures =NULL;
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    weakfeatures = new spCore* [n_features];
=======
    weakfeatures = manage_array(new spCore* [n_features]);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc


    bool t;
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    TensorSize *modelsize = new TensorSize(height,width);
    TensorRegion *tregion = new TensorRegion(0,0,height,width);
=======
    TensorSize *modelsize = manage(new TensorSize(height,width));
    TensorRegion *tregion = manage(new TensorRegion(0,0,height,width));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    if (width>0 && height >0)
    {
        int i =0;
        //Edge feature 1

        for (int w_=2;w_<width;w_=w_+2)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_);
//                          t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_,x_+(w_/2), h_,(w_/2));
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        //  t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //   t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_,x_+(w_/2), h_,(w_/2));
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }



        //Edge feature 2
        for (int w_=1;w_<width;w_++)
            for (int h_ =2;h_<height;h_=h_+2)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+(h_/2), w_,(h_/2));
//                         t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,  y_+(h_/2),x_,(h_/2),w_);
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
=======
                        //  t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        // t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+(h_/2), w_,(h_/2));
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,  y_+(h_/2),x_,(h_/2),w_);
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
                        i++;
                    }

        //  n_features++;


        //Line feature 1

        for (int w_=3;w_<width;w_=w_+3)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, x_+(w_/3), y_, (w_/3),h_);
=======
                        //    t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //   t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, x_+(w_/3), y_, (w_/3),h_);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                         t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, y_,x_+(w_/3), h_, (w_/3));
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, y_,x_+(w_/3), h_, (w_/3));
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }

        // n_features++;



        //Line feature 2
        for (int w_=1;w_<width;w_++)
            for (int h_ =3;h_<height;h_=h_+3)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, x_, y_+h_/3, w_,h_/3);
=======
                        //       t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //      t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3, x_, y_+h_/3, w_,h_/3);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                         t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3,  y_+h_/3, x_,h_/3,w_);
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-3,  y_+h_/3, x_,h_/3,w_);
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }


        // n_features++;

        //Line feature 3
        for (int w_=1;w_<width;w_++)
            for (int h_ =4;h_<height;h_=h_+4)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+h_/4, w_,h_/2);
=======
                        //    t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //   t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+h_/4, w_,h_/2);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc


<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                       t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_+h_/4,x_ ,h_/2,w_);
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_+h_/4,x_ ,h_/2,w_);
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }

        // n_features++;

        //Line feature 4
        for (int w_=4;w_<width;w_=w_+4)
            for (int h_ =1;h_<height;h_++)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/4), y_, (w_/2),h_);
=======
                        //    t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //   t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/4), y_, (w_/2),h_);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                          t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/4), h_, (w_/2));
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/4), h_, (w_/2));
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }


        //n_features++;

        for (int w_=3;w_<width;w_=w_+3)
            for (int h_ =3;h_<height;h_=h_+3)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(2);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/3), y_+h_/3, (w_/3),h_/3);
=======
                        //             t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //                 t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/3), y_+h_/3, (w_/3),h_/3);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc


<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_+(h_/3), x_+w_/3, (h_/3),w_/3);
                         ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, y_+(h_/3), x_+w_/3, (h_/3),w_/3);
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }

        // n_features++;

        for (int w_=2;w_<width;w_=w_+2)
            for (int h_ =2;h_<height;h_=h_+2)
                for (int x_=0;x_<width-w_;x_++)
                    for (int y_=0;y_<height-h_;y_++)
                        //  n_features++;
                    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        weakfeatures[i] = new ipHaarLienhart();//width, height);
=======
                        weakfeatures[i] = manage(new ipHaarLienhart());//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setNoRec(3);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_/2);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(2,-2, x_, y_+h_/2, (w_/2),h_/2);
=======
                        //      t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
                        //      t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_/2);
                        //        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(2,-2, x_, y_+h_/2, (w_/2),h_/2);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/2), (h_/2),w_/2);
//                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(2,-2, y_+h_/2,x_, (h_/2),w_/2);
=======
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/2), (h_/2),w_/2);
                        t= ((ipHaarLienhart *)weakfeatures[i])->setRec(2,-2, y_+h_/2,x_, (h_/2),w_/2);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        ((ipHaarLienhart *)weakfeatures[i])->setModelSize(*modelsize);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
                         ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
=======
                        ((ipHaarLienhart *)weakfeatures[i])->setRegion(*tregion);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
                        i++;

                    }


    }


<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    else
    {
        if (max_features > 0 && max_features < n_features) n_features = max_features;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        weakfeatures = new spCore* [n_features];
        for (int i = 0 ; i < n_features ; i++)
            weakfeatures[i] = new spDeltaOne(i);
    }
//
//	/*
//	Tensor *t1 = mdataset.getExample(0);
////	t1->print("t1");
//	for(int i = 0 ; i < n_features ; i++)
//	{
//		weakfeatures[i]->process(*t1);
//
//		weakfeatures[i]->getOutput(0).sprint("f(%d)", i);
//	}
//	*/
//
=======
    ///////////////////////////////////////////////////
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    BoostingTrainer boosting;
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    WeakLearner **stump_trainers = new WeakLearner*[n_classifiers];
    StumpMachine **stump_machines = new StumpMachine*[n_classifiers];
//	//LBPStumpMachine *m = new LBPStumpMachine();
//	//m->setLBP(ip_lbp, w, h);
//
=======
    WeakLearner **stump_trainers = manage_array(new WeakLearner*[n_classifiers]);
    StumpMachine **stump_machines = manage_array(new StumpMachine*[n_classifiers]);


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    for ( int i = 0; i < n_classifiers; i++)
    {
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        stump_machines[i] = new StumpMachine();
=======
        stump_machines[i] = manage(new StumpMachine());
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
        stump_trainers[i] = new StumpTrainer(stump_machines[i], n_features, weakfeatures);
=======
        stump_trainers[i] = manage(new StumpTrainer(stump_machines[i], n_features, weakfeatures));
       stump_trainers[i]->setBOption("verbose",true);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }
    boosting.setBOption("boosting_by_sampling",true);
    boosting.setWeakLearners(n_classifiers, stump_trainers);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    boosting.setData(&mdataset);
=======
    boosting.setData(m_dataset);
    boosting.setBOption("verbose",true);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    boosting.train();
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======



//check the model if it is trained properly

//for (int e=0;e<n_examples;e++)
//    {
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
//
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
//        Tensor *ttensor = (Tensor*)m_dataset->getExample(e);
//        double s;
//        s=0;
//        for(int i=0;i<n_classifiers;i++)
//        {
//                stump_machines[i]->forward(*ttensor);
//                DoubleTensor *t_output = (DoubleTensor *) &stump_machines[i]->getOutput();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
//
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
//	// Saving ...
//	print("Saving model ...\n");
//	File file;
=======
//            s +=  stump_trainers[i]->getWeight()*(*t_output)(0);
//        }
//        print("Score %f\n",s);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
//
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
////	file.open("boost.model","w");
////	for( int i = 0; i < n_classifiers; i++)
////	{
////		print("WeakClassifier (%d):\n", i);
////		print("weight = %g\n", stump_trainers[i]->getWeight());
////		stump_machines[i]->saveFile(file);
////		print("\n");
////	}
////	file.close();
/////......................................................................

    CascadeMachine* CM = new CascadeMachine();
=======
//    }









    //Saving the model.


    CascadeMachine* CM = manage(new CascadeMachine());
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    CM->resize(1);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
    CM->setSize(*modelsize);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    CM->resize(0,n_classifiers);
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    //   bool t;
    TensorSize *ts = new TensorSize(height,width);
    CM->setSize(*ts);
=======

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    for (int i=0;i<n_classifiers;i++)
    {
        t= CM->setMachine(0,i,stump_machines[i]);
        t= CM->setWeight(0,i,stump_trainers[i]->getWeight());
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    }

    t=CM->setThreshold(0,0.0);


<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======
//Check Cascade

//for (int e=0;e<n_examples;e++)
//    {
//
//        Tensor *ttensor = (Tensor*)m_dataset->getExample(e);
//      //  double s;
//         CM->setRegion(*tregion);
//        if (CM->forward(*ttensor) == false)
//        {
//
//            delete CM;
//            return 1;
//        }
//
//        print("Score %f\n",CM->getConfidence());
//
//    }


printf(".....................\n");
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    //  t=CM->setModelSize(width,height);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    File f1;
    t=f1.open("model.wsm","w");
    t=CM->saveFile(f1);
    f1.close();
<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    delete CM;
//        // Load the cascade machine
//        print("\n\n\n\n\n...................Loading the model\n");
//	CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile("model.wsm");
//	if (cascade == 0)
//	{
//		print("ERROR: loading model [%s]!\n", "model.wsm");
//		return 1;
//	}
//
//Tensor *st = mdataset.getExample(3);
////print("size of tensor  %d\n",st->nDimension());
////Tprint(st);
//if (cascade->forward(*st) == false)
//		{
////			print("ERROR: failed to run the cascade on the image [%d/%d]!\n",
////				j + 1, n_samples);
//			delete cascade;
//			return 1;
//		}
//
//print("CONFIDENCE = %f\n", cascade->getConfidence());
//	//print("Cascade model size : %d %d\n",cascade->getInputSize().size[0],cascade->getInputSize().size[1]);
/////.....................................................................
    //
//    for ( int i = 0; i < n_classifiers; i++)
//    {
//        delete stump_machines[i];
//        delete stump_trainers[i];
//    }
//    delete [] stump_machines;
//    delete [] stump_trainers;
//	//delete ip_lbp;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    for (int i = 0 ; i < n_features ; i++) delete weakfeatures[i];
    delete []weakfeatures;
=======
 CascadeMachine* cascade = manage((CascadeMachine*)Torch::loadMachineFromFile("model.wsm"));
    if (cascade == 0)
    {
        print("ERROR: loading model [%s]!\n", "model.wsm");
        return 1;
    }
for (int i=0;i<n_examples;i++)
    {
        Tensor *st = (Tensor*)m_dataset->getExample(i);

        cascade->setRegion(*tregion);
        if (cascade->forward(*st) == false)
        {

            delete cascade;
            return 1;
        }
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
    //
    delete target1;
    delete target0;
=======
     //   ofile.printf("%g\n",cascade->getConfidence());
        print("CONFIDENCE = %f\n", cascade->getConfidence());
      //  count += cascade->isPattern() ? 1 : 0;
    }
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc


    // OK
    print("OK.\n");

<<<<<<< HEAD:examples/boosting/boostingintegral.cc
=======


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingintegral.cc
    return 0;
}

