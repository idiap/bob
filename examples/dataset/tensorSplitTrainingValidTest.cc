#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char* tensor_filename;
    bool verbose;
    float trp,vap,tep; // percentage split
    int randseed;
    char* basename;
    char filename[2048];
    randseed = 0; //default value

    Tensor::Type mtype;
    int m_n_samples;
    FileListCmdOption* tensor_files = new FileListCmdOption("positivePatterns", "+ve training patterns");
    tensor_files->isArgument(true);

    CmdLine cmd;
    cmd.setBOption("write log", false);
    cmd.info("Splits Tensor files to training, valid and test dataset ");

    cmd.addText("\nArguments:");
    cmd.addCmdOption(tensor_files);
    cmd.addSCmdArg("File BaseName",&basename , "File BaseName");
    cmd.addFCmdArg("Training %", &trp, "Training %");
    cmd.addFCmdArg("Valid %", &vap, "Valid %");
    cmd.addFCmdArg("Testing %", &tep, "Testing %");
    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");


    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

    MTimer mtime;
    mtime.reset();
    ShortTensor *target1 = manage(new ShortTensor(1));
    target1->fill(1);

    TensorList *tensorList_p= manage(new TensorList()); //for positive
    if (tensorList_p->process(tensor_files,target1,Tensor::Short)==false)
    {
        print("Error in reading +ve training pattern Tensor list\n");
        return 1;
    }
    DataSet *pDataSet = tensorList_p->getOutput();
    mtime.stop();
    print("Time taken to load %d %d\n",mtime.minutes,mtime.seconds);

//Now u have all the tensors loaded in memory
//Use a constant seed to use the Random function


    int total_examples = pDataSet->getNoExamples();
    int n_training_examples = int(total_examples*trp);
    int n_valid_examples = int(total_examples*vap);
    int n_testing_examples = int(total_examples*tep);
    print("Total Number of examples : %d\n",total_examples);
    print("Training Examples : %d\n",n_training_examples);
    print("Valid Examples : %d\n",n_valid_examples);
    print("Testing Examples : %d\n",n_testing_examples);


//Create a filename to save


    bool *track = manage_array(new bool[total_examples]);


    for (int i=0;i<total_examples;i++)
        track[i]=0;

    Tensor *example = pDataSet->getExample(0);

//print("Example Ndimension %d, %d %d\n",example->nDimension(),example->size(0),example->size(1));
    THRandom_manualSeed(randseed);
    sprintf(filename,"%s_train.tensor",basename);
    TensorFile *onetensor_file = new TensorFile;
    CHECK_FATAL(onetensor_file->openWrite(filename,
                                          Tensor::Short, example->nDimension(),
                                          example->size(0), example->size(1), 0, 0));

//double z = THRandom_uniform(0, 1);

    int count=0;
    do
    {
        int z = int(THRandom_uniform(0,total_examples));
        if (track[z]==0)
        {
            ShortTensor *example = (ShortTensor *)pDataSet->getExample(z);
            onetensor_file->save(*example);
            track[z]=1;
            count++;
        }

    }
    while (count<n_training_examples);

    onetensor_file->close();

//..............................................
// For Valid dataset
    sprintf(filename,"%s_valid.tensor",basename);
    CHECK_FATAL(onetensor_file->openWrite(filename,
                                          Tensor::Short, example->nDimension(),
                                          example->size(0), example->size(1), 0, 0));

    count=0;
    do
    {
        int z = int(THRandom_uniform(0,total_examples));
        if (track[z]==0)
        {
            ShortTensor *example = (ShortTensor *)pDataSet->getExample(z);
            onetensor_file->save(*example);
            track[z]=1;
            count++;
        }

    }
    while (count<n_valid_examples);
    onetensor_file->close();
//..............................................

// for test dataset
if(tep>0)
{
    sprintf(filename,"%s_test.tensor",basename);
    CHECK_FATAL(onetensor_file->openWrite(filename,
                                          Tensor::Short, example->nDimension(),
                                          example->size(0), example->size(1), 0, 0));
    for (int i=0;i<total_examples;i++)
    {
        if (track[i]==0)
        {
            ShortTensor *example = (ShortTensor *)pDataSet->getExample(i);
            onetensor_file->save(*example);
        }
    }
    onetensor_file->close();

}
    return 0;
}
