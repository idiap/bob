#include "CmdLine.h"

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
#include "TensorFile.h"
=======

#include "TensorList.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
#include "MemoryDataSet.h"

#include "BoostingRoundLBPTrainer.h"
#include "LBPRoundTrainer.h"
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
#include "LBPMachine.h"
=======
#include "IntLutMachine.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
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
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
#include "spCores.h"
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

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

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
    FileListCmdOption* p_tensor_files = new FileListCmdOption("positivePatterns", "+ve training patterns");
    p_tensor_files->isArgument(true);

    FileListCmdOption* n_tensor_files = new FileListCmdOption("negativePatterns", "-ve training patterns");
    n_tensor_files->isArgument(true);


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tensor read program");

    cmd.addText("\nArguments:");
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    cmd.addSCmdArg("tensor file for target 0", &tensor_filename_target0, "tensor file for target 0");
    cmd.addSCmdArg("tensor file for target 1", &tensor_filename_target1, "tensor file for target 1");
=======
    cmd.addCmdOption(p_tensor_files);
    cmd.addCmdOption(n_tensor_files);

>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    cmd.addICmdOption("-wc", &n_classifiers, 5, "number of weak classifiers");
    cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
    cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    cmd.addICmdOption("-width", &width, -1, "image width");
    cmd.addICmdOption("-height", &height, -1, "image height");
=======
    // cmd.addICmdOption("-maxE", &max_examples, -1, "maximum number of examples to load");
    // cmd.addICmdOption("-maxF", &max_features, -1, "maximum number of features to process");
    // cmd.addICmdOption("-width", &width, 19, "image width");
    //  cmd.addICmdOption("-height", &height, 19, "image height");
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

    int n_examples;
    int n_features;

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
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
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    Tensor *tensor = new FloatTensor(n_features);
    FloatTensor *unfoldtensor = new FloatTensor;
    ShortTensor *target0 = new ShortTensor(1);
    target0->fill(0);
    ShortTensor *target1 = new ShortTensor(1);
=======
    ShortTensor *target1 = manage(new ShortTensor(1));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    target1->fill(1);
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
    ShortTensor *target0 = manage(new ShortTensor(1));
    target0->fill(0);
    DataSet *pDataSet;
    DataSet *nDataSet;
    TensorList *tensorList_p= manage(new TensorList()); //for positive
    TensorList *tensorList_n= manage(new TensorList()); //for validation
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    //
    print("Building a MemoryDataSet ...\n");
    MemoryDataSet mdataset(n_examples, Tensor::Double, true, Tensor::Short);
    CHECK_FATAL(mdataset.getNoExamples() == n_examples);

    //
    print("Filling the MemoryDataSet ...\n");
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    for (int i = 0 ; i < n_examples_0 ; i++)
=======
    if (tensorList_p->process(p_tensor_files,target1,Tensor::Double)==false)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    {
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
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
        print("Error in reading +ve training patterns - Tensor list\n");
        return 1;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
    pDataSet = tensorList_p->getOutput();
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    tf0.close();
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    for (int i = n_examples_0 ; i < n_examples ; i++)
=======
    if (tensorList_n->process(n_tensor_files,target0,Tensor::Double)==false)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    {
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
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
=======
        print("Error in reading -ve patterns - Tensor list\n");
        return 1;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    tf1.close();

    delete unfoldtensor;
    delete tensor;

    //
    if (verbose)
=======
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
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    {
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        print("Reading the MemoryDataSet ...\n");
        for (int i = 0 ; i < mdataset.getNoExamples() ; i++)
        {
            mdataset.getExample(i)->sprint("%d",i);

            mdataset.getTarget(i)->print("target");
        }
=======
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
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc

    // Tprint(mdataset.getExample(2));
//    ipIntegral *ipI = new ipIntegral();
//    DoubleTensor *st = new DoubleTensor();
//
//    for (int e=0;e<n_examples;e++)
//    {
//
//        st = (DoubleTensor*)mdataset.getExample(e);
//
//        bool t = ipI->process(*st);
//        st = (DoubleTensor*) &ipI->getOutput(0);
//
//        mdataset.getExample(e)->copy(st);
//    }
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    print("Prepare Boosting ...\n");

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    /*
    int w = 19;
    int h = 19;

    CHECK_FATAL(n_features == w*h);

    ipLBP8R* ip_lbp = new ipLBP8R();
    ip_lbp->setR(2);
    */

//spCore **weakfeatures =NULL;
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    if (width>0 && height >0)
    {
        n_features =0;
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        //Edge feature 1

      //  for (int w_=1;w_<width;w_=w_+2)
          //  for (int h_ =1;h_<height;h_++)
                for (int x_=1;x_<width-1;x_++)
                    for (int y_=1;y_<height-1;y_++)
                        n_features++;

=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
                n_features++;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    }

    print("Number of feature %d\n",n_features);


<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    spCore **weakfeatures = new spCore* [n_features];
=======
    spCore **weakfeatures = manage_array(new spCore* [n_features]);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc


    bool t;
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    TensorSize *modelsize = new TensorSize(height,width);
    TensorRegion *tregion = new TensorRegion(0,0,height,width);
   // print("width and height %d %d\n", width, height);
=======
    TensorSize *modelsize = manage(new TensorSize(height,width));
    TensorRegion *tregion = manage(new TensorRegion(0,0,height,width));
    // print("width and height %d %d\n", width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    if (width>0 && height >0)
    {
        int i =0;
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        //Edge feature 1

        //for (int w_=2;w_<width;w_=w_+2)
         //   for (int h_ =1;h_<height;h_++)
                for (int x_=1;x_<width-1;x_++)
                    for (int y_=1;y_<height-1;y_++)
                    {
                        weakfeatures[i] = new ipLBP8R;//width, height);

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
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
            {
                weakfeatures[i] = manage(new ipLBP8R);//width, height);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
                t= ((ipLBP *)weakfeatures[i])->setBOption("ToAverage", true);
                t= ((ipLBP*)weakfeatures[i])->setBOption("AddAvgBit", true);
                t= ((ipLBP *)weakfeatures[i])->setBOption("Uniform", false);
                t= ((ipLBP *)weakfeatures[i])->setBOption("RotInvariant", false);
                ((ipLBP  *)weakfeatures[i])->setModelSize(*modelsize);
                ((ipLBP  *)weakfeatures[i])->setRegion(*tregion);
                t=((ipLBP *)weakfeatures[i])->setXY(x_, y_);
                //  print("iteration number %d\n",i);
                i++;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======
            }
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    }


    else
    {
        if (max_features > 0 && max_features < n_features) n_features = max_features;

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        weakfeatures = new spCore* [n_features];
=======
        weakfeatures = manage_array(new spCore* [n_features]);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
        for (int i = 0 ; i < n_features ; i++)
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
            weakfeatures[i] = new spDeltaOne(i);
=======
            weakfeatures[i] = manage(new spDeltaOne(i));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
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
    BoostingRoundLBPTrainer boosting;
    WeakLearner **lbp_trainers = new WeakLearner*[n_rounds];
    LBPMachine **lbp_machines = new LBPMachine*[n_rounds];
//	//LBPStumpMachine *m = new LBPStumpMachine();
//	//m->setLBP(ip_lbp, w, h);
//
=======

    WeakLearner **lbp_trainers = manage_array(new WeakLearner*[n_rounds]);
    IntLutMachine **lbp_machines = manage_array(new IntLutMachine*[n_rounds]);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    for ( int i = 0; i < n_rounds; i++)
    {
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        lbp_machines[i] = new LBPMachine;
=======
        lbp_machines[i] = manage(new IntLutMachine);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        lbp_trainers[i] = new LBPRoundTrainer(lbp_machines[i], n_features, weakfeatures);
=======
        lbp_trainers[i] = manage(new LBPRoundTrainer(lbp_machines[i], n_features, weakfeatures));
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
=======

    BoostingRoundLBPTrainer boosting;
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    boosting.setBOption("boosting_by_sampling", true);
    boosting.setIOption("number_of_rounds", n_rounds);
    boosting.setWeakLearners(n_classifiers, lbp_trainers);
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    boosting.setData(&mdataset);
=======
    boosting.setData(m_dataset);
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    boosting.train();
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
//
//
//	// Saving ...
//	print("Saving model ...\n");
//	File file;
//
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


    //Saving the model.

    CascadeMachine* CM = manage(new CascadeMachine());
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    CM->resize(1);
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    CM->resize(0,n_rounds);
    //   bool t;
    for (int i=0;i<n_rounds;i++)
=======
    CM->setSize(*modelsize);
    CM->resize(0,n_classifiers);

    for (int i=0;i<n_classifiers;i++)
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    {
        t= CM->setMachine(0,i,lbp_machines[i]);
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
        t= CM->setWeight(0,i,lbp_trainers[i]->getWeight());
        print("Weight %f\n",lbp_trainers[i]->getWeight());
=======
        t= CM->setWeight(0,i,1);


>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    }

    t=CM->setThreshold(0,0.0);
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
//t=CM->setModelSize(width,height);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
    //  t=CM->setModelSize(width,height);
=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc
    File f1;
    t=f1.open("model.wsm","w");
    t=CM->saveFile(f1);
    f1.close();
<<<<<<< HEAD:examples/boosting/boostingLBPRound.cc
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

 //   for (int i = 0 ; i < n_features ; i++) delete weakfeatures[i];
  //  delete []weakfeatures;

    //
    delete target1;
    delete target0;

=======
>>>>>>> 75faad1c7f2eda5814add549cc0d84528aa9f1bb:examples/boosting/boostingLBPRound.cc

    // OK
    print("OK.\n");

    return 0;
}

