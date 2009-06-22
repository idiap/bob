#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char* tensor_filename_target0;
    char* tensor_filename_target1;
    char* modelfilename;
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
    cmd.addSCmdArg("model file name", &modelfilename, "Saving Trained model file");

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    cmd.addICmdOption("-wc", &n_classifiers, 5, "number of weak classifiers");
    cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");

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

    if (verbose)
        print("Prepare Boosting ...\n");

    if (width>0 && height >0)
    {
        n_features =0;

        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
                n_features++;

    }


    if (verbose)
        print("Number of feature %d\n",n_features);


    spCore **weakfeatures = manage_array(new spCore* [n_features]);


    bool t;
    TensorSize modelsize(height,width);
    TensorRegion tregion (0,0,height,width);

    if (width>0 && height >0)
    {
        int i =0;

        for (int x_=1;x_<width-1;x_++)
            for (int y_=1;y_<height-1;y_++)
            {
                ipLBP8R* ip_lbp = manage(new ipLBP8R);//width, height);
                ip_lbp->setBOption("ToAverage", true);
                ip_lbp->setBOption("AddAvgBit", true);
                ip_lbp->setBOption("Uniform", false);
                ip_lbp->setBOption("RotInvariant", false);
                ip_lbp->setModelSize(modelsize);
                ip_lbp->setRegion(tregion);
                ip_lbp->setXY(x_, y_);

                weakfeatures[i] = ip_lbp;
                i++;
            }
    }


    WeakLearner **lbp_trainers = manage_array(new WeakLearner*[n_rounds]);
    IntLutMachine **lbp_machines = manage_array(new IntLutMachine*[n_rounds]);
    for ( int i = 0; i < n_rounds; i++)
    {
        lbp_machines[i] = manage(new IntLutMachine);

        lbp_trainers[i] = manage(new LBPRoundTrainer(lbp_machines[i], n_features, weakfeatures));
        if (verbose)
            lbp_trainers[i]->setBOption("verbose",true);
    }

    BoostingRoundLBPTrainer boosting;
    boosting.setBOption("boosting_by_sampling", true);
    boosting.setIOption("number_of_rounds", n_rounds);
    boosting.setWeakLearners(n_classifiers, lbp_trainers);
    boosting.setData(m_dataset);
    if (verbose)
        boosting.setBOption("verbose",true);
    boosting.train();


    //Saving the model.

    CascadeMachine* CM = manage(new CascadeMachine());
    CM->resize(1);
    CM->setSize(modelsize);
    CM->resize(0,n_classifiers);

    for (int i=0;i<n_classifiers;i++)
    {
        CM->setMachine(0,i,lbp_machines[i]);
       CM->setWeight(0,i,1);

    }

    t=CM->setThreshold(0,0.0);

    File f1;
    f1.open(modelfilename,"w");
    CM->saveFile(f1);
    f1.close();

    // OK
    print("OK.\n");

    return 0;
}

