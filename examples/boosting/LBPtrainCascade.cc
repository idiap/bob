#include "CmdLine.h"

#include "TensorList.h"
#include "MemoryDataSet.h"
#include "ImageScanDataSet.h"
#include "FileListCmdOption.h"
#include "BoostingRoundLBPTrainer.h"
#include "WeakLearner.h"
#include "FTrainer.h"
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
#include "CascadeTrainer.h"
#include "spCore.h"
#include "spCores.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char *tensor_filename;
    //char *p_valid_tensor_filename; //validation file
    char *parameter_filename;

    FileListCmdOption* p_tensor_files = new FileListCmdOption("positivePatterns", "+ve training patterns");
    p_tensor_files->isArgument(true);

    FileListCmdOption* v_tensor_files = new FileListCmdOption("validPatterns", "+ve validation patterns");
    v_tensor_files->isArgument(true);
    // Set source images
    FileListCmdOption* image_files = new FileListCmdOption("images", "source images");
    image_files->isArgument(true);

    // Set source .wnd
    FileListCmdOption* wnd_files = new FileListCmdOption("wnd", "source subwindows");
    wnd_files->isArgument(true);

    // Directory where to load images from
    char*	dir_images;

    // Image extension
    char*	ext_images;

    // Directory to load .wnd files from
    char*	dir_wnds;

    // Desired output size

    //These values can come under parameter file
    int	width, height;

    int n_stages;
    int *n_weakClassifier;
    int *n_rounds;
    double *detection_rate;
    bool verbose;

//..............
    int n_features;





    // Build the command line object
    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Cascade Trainer");

    cmd.addText("\nArguments:");
    //  cmd.addSCmdArg("tensor file for target 0", &tensor_filename_target0, "tensor file for target 0");
    // cmd.addSCmdArg("tensor file for target 1", &p_tensor_filename, "tensor file for target 1 (+ve training patterns)");
    cmd.addCmdOption(p_tensor_files);
    cmd.addCmdOption(v_tensor_files);
    cmd.addSCmdArg("parameter file", &parameter_filename, "parameter file name");
    cmd.addCmdOption(image_files);
    cmd.addCmdOption(wnd_files);
    cmd.addSCmdArg("image directory", &dir_images, "directory where to load images from");
    cmd.addSCmdArg("image extension", &ext_images, "image extension");
    cmd.addSCmdArg(".wnd directory", &dir_wnds, "directory where to load the .wnd files from");
    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");


    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

////////////////////////////////////////////////////////////////////
    //  Read the paramete file here
////////////////////////////////////////////////////////////////
    bool b;
    File fp1;
    b = fp1.open(parameter_filename,"r");
    fp1.scanf("%d",&width);
    fp1.scanf("%d",&height);
    fp1.scanf("%d",&n_stages);
    n_weakClassifier = manage_array(new int[n_stages]);
    n_rounds = manage_array(new int[n_stages]);
    detection_rate = manage_array(new double[n_stages]);

    for (int i=0;i<n_stages;i++)
    {
        int temp_val;
        double temp_d;
        fp1.scanf("%d",&temp_val);
        n_weakClassifier[i] = temp_val;
        fp1.scanf("%d",&temp_val);
        n_rounds[i] = temp_val;
        fp1.scanf("%lf",&temp_d);
        detection_rate[i] = temp_d;
    }
    fp1.close();

    //print the parameters
    print(".................................\n");
    print("Width %d, height %d\n",width,height);
    print("Number of stages %d\n",n_stages);

    for (int i=0;i<n_stages;i++)
    {
        print("nWeakClassifiers %d, nRounds %d, detectionRare %f\n",n_weakClassifier[i],n_rounds[i], detection_rate[i]);
    }
    print(".................................\n");




//Read the +ve and valida patterns
    ShortTensor *target1 = manage(new ShortTensor(1));
    target1->fill(1);
    DataSet *pDataSet;
    DataSet *vDataSet;
    TensorList *tensorList_p= manage(new TensorList()); //for positive
    TensorList *tensorList_v= manage(new TensorList()); //for validation

    if (tensorList_p->process(p_tensor_files,target1)==false)
    {
        print("Error in reading +ve training pattern Tensor list\n");
        return 1;
    }
    pDataSet = tensorList_p->getOutput();


    if (tensorList_v->process(v_tensor_files,target1)==false)
    {
        print("Error in reading +ve valid pattern Tensor list\n");
        return 1;
    }
    vDataSet = tensorList_v->getOutput();





//////////////////////////////////////////////////////////////////////////////////////////////
// Now set the Imagescan dataset
////////////////////////////////////////////////

// Check arguments
    if (image_files->n_files < 1)
    {
        print("Error: no image file provided!\n");
        return 1;
    }
    if (image_files->n_files != wnd_files->n_files)
    {
        print("Error: mismatch between the number of image files and wnd files!\n");
        return 1;
    }

    // Create the main ImageScanDataSet object
    ImageScanDataSet iscandataset(	image_files->n_files,
                                   dir_images, image_files->file_names, ext_images,
                                   dir_wnds, wnd_files->file_names,
                                   width, height, 1);
////////////////////////////////////////////////////////////////////////////////////////
    long n_imgexample = iscandataset.getNoExamples();
    DoubleTensor accept_target(1);

    accept_target.fill(1.0);

    for (long i = 0; i < n_imgexample; i ++)
    {

        // if (((DoubleTensor*)iscandataset.getTarget(i))->get(0) < 0.0);
        iscandataset.setTarget(i, &accept_target);
        //CHECK_FATAL(((DoubleTensor*)iscandataset.getTarget(i))->get(0) > 0.0);
    }

    print("Number of examples in iscandataset is %ld\n",iscandataset.getNoExamples());



    print("Prepare Cascade Boosting ...\n");



    ////////////////////////////////////////////////
    // get the number of features
    //////////////////////////////////////////////
    if (width>0 && height >0)
    {
        n_features =0;
        //Edge feature 1

        //  for (int w_=1;w_<width;w_=w_+2)
        //  for (int h_ =1;h_<height;h_++)
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
        //Edge feature 1

        //for (int w_=2;w_<width;w_=w_+2)
        //   for (int h_ =1;h_<height;h_++)
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


//////////////////////////////////////////////////////////////////
    FTrainer **boosting = manage_array(new FTrainer*[n_stages]);
    WeakLearner ***lbp_trainers = manage_array(new WeakLearner**[n_stages]);
    IntLutMachine ***lbp_machines = manage_array(new IntLutMachine**[n_stages]);
    print(".............1\n");
    for (int i=0;i<n_stages;i++)
    {
        boosting[i] = manage(new BoostingRoundLBPTrainer);

        lbp_trainers[i] = manage_array(new WeakLearner*[n_rounds[i]]);

        lbp_machines[i]  = manage_array(new IntLutMachine*[n_rounds[i]]);

        for (int j=0;j<n_rounds[i];j++)
        {
            lbp_machines[i][j] = manage(new IntLutMachine);
            lbp_trainers[i][j] = manage(new LBPRoundTrainer(lbp_machines[i][j], n_features, weakfeatures));
        }

        ((BoostingRoundLBPTrainer*)boosting[i])->setBOption("boosting_by_sampling", true);
        ((BoostingRoundLBPTrainer*)boosting[i])->setIOption("number_of_rounds", n_rounds[i]);
        ((BoostingRoundLBPTrainer*)boosting[i])->setWeakLearners(n_weakClassifier[i], lbp_trainers[i]);
    }

    print(".............2\n");
    CascadeTrainer *CT = manage(new CascadeTrainer());
    //

    t=CT->setTrainers(boosting, n_stages, detection_rate);
    t=CT->setData(pDataSet,vDataSet,&iscandataset);
    t=CT->train();


    CascadeMachine* CM = manage(new CascadeMachine());
    CM->resize(n_stages);
    CM->setSize(*modelsize);
    for (int k=0;k<n_stages;k++)
    {
        CM->resize(k,n_weakClassifier[k]);
        //   bool t;
        for (int i=0;i<n_weakClassifier[k];i++)
        {
            t= CM->setMachine(k,i,lbp_machines[k][i]);
            t= CM->setWeight(k,i,1.0);


        }

        t=CM->setThreshold(k,CT->threshold[k]);

    }

    File f1;
    t=f1.open("model.wsm","w");
    t=CM->saveFile(f1);
    f1.close();
    print("OK.\n");

    return 0;
}

