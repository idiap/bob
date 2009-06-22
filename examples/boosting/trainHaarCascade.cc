#include "torch5spro.h"

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
	int* n_weakClassifier;
	char* modelfilename;
	//  int *n_rounds;
	double* detection_rate;
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
	cmd.addSCmdArg("model file name", &modelfilename, "Saving Trained model file");
	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print values");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	////////////////////////////////////////////////////////////////////
	//  Read the parameters file here
	////////////////////////////////////////////////////////////////
	bool b;
	File fp1;
	b = fp1.open(parameter_filename,"r");
	fp1.scanf("%d",&width);
	fp1.scanf("%d",&height);
	fp1.scanf("%d",&n_stages);
	n_weakClassifier = manage_array(new int[n_stages]);
	//n_rounds = manage_array(new int[n_stages]);
	detection_rate = manage_array(new double[n_stages]);

	for (int i=0;i<n_stages;i++)
	{
		int temp_val;
		double temp_d;
		fp1.scanf("%d",&temp_val);
		n_weakClassifier[i] = temp_val;
		//   fp1.scanf("%d",&temp_val);
		//   n_rounds[i] = temp_val;
		fp1.scanf("%lf",&temp_d);
		detection_rate[i] = temp_d;
	}
	fp1.close();

	//print the parameters
	if (verbose)
	{
		print(".................................\n");
		print("Width %d, height %d\n",width,height);
		print("Number of stages %d\n",n_stages);

		for (int i=0;i<n_stages;i++)
		{
			print("nWeakClassifiers %d, detectionRate %f\n",n_weakClassifier[i], detection_rate[i]);
		}
		print(".................................\n");
	}

	//Read the +ve and valida patterns
	ShortTensor *target1 = manage(new ShortTensor(1));
	target1->fill(1);

	TensorList *tensorList_p= manage(new TensorList()); //for positive
	if (tensorList_p->process(p_tensor_files,target1,Tensor::Int)==false)
	{
		print("Error in reading +ve training pattern Tensor list\n");
		return 1;
	}
	DataSet *pDataSet = tensorList_p->getOutput();

	TensorList *tensorList_v= manage(new TensorList()); //for validation
	if (tensorList_v->process(v_tensor_files,target1,Tensor::Int)==false)
	{
		print("Error in reading +ve valid pattern Tensor list\n");
		return 1;
	}
	DataSet *vDataSet = tensorList_v->getOutput();

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
		iscandataset.setTarget(i, &accept_target);
		//CHECK_FATAL(((DoubleTensor*)iscandataset.getTarget(i))->get(0) > 0.0);
	}

	if (verbose)
	{
		print("Number of examples in iscandataset is %ld\n",iscandataset.getNoExamples());
		print("Prepare Cascade Boosting ...\n");
	}

	////////////////////////////////////////////////
	// get the number of features
	//////////////////////////////////////////////
	if (width>0 && height >0)
	{
		n_features =0;
		//Edge feature 1

		for (int w_=10;w_<width;w_=w_+2)
			for (int h_ =10;h_<height;h_++)
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

	if (verbose)
		print("Number of feature %d\n",n_features);


	spCore **weakfeatures = manage_array(new spCore* [n_features]);

	TensorSize modelsize(height,width);
	TensorRegion tregion(0,0,height,width);

	if (width>0 && height >0)
	{
		int i =0;

		//Edge feature 1
		for (int w_=10;w_<width;w_=w_+2)
			for (int h_ =10;h_<height;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
					{
						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
						iph->setNoRec(2);
						iph->setRec(0,1, y_, x_, h_, w_);
						iph->setRec(1,-2, y_,x_+(w_/2), h_,(w_/2));
						iph->setModelSize(modelsize);
						iph->setRegion(tregion);

						weakfeatures[i ++] = iph;
					}
 
 		//Edge feature 2
 		for (int w_=1;w_<width;w_++)
 			for (int h_ =2;h_<height;h_=h_+2)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-2,  y_+(h_/2),x_,(h_/2),w_);
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		//Line feature 1
 		for (int w_=3;w_<width;w_=w_+3)
 			for (int h_ =1;h_<height;h_++)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-3, y_,x_+(w_/3), h_, (w_/3));
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		//Line feature 2
 		for (int w_=1;w_<width;w_++)
 			for (int h_ =3;h_<height;h_=h_+3)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-3,  y_+h_/3, x_,h_/3,w_);
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		//Line feature 3
 		for (int w_=1;w_<width;w_++)
 			for (int h_ =4;h_<height;h_=h_+4)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-2, y_+h_/4,x_ ,h_/2,w_);
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		//Line feature 4
 		for (int w_=4;w_<width;w_=w_+4)
 			for (int h_ =1;h_<height;h_++)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-2,y_, x_+(w_/4), h_, (w_/2));
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		for (int w_=3;w_<width;w_=w_+3)
 			for (int h_ =3;h_<height;h_=h_+3)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(2);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-2, y_+(h_/3), x_+w_/3, (h_/3),w_/3);
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}
 
 		//n_features++;
 
 		for (int w_=2;w_<width;w_=w_+2)
 			for (int h_ =2;h_<height;h_=h_+2)
 				for (int x_=0;x_<width-w_;x_++)
 					for (int y_=0;y_<height-h_;y_++)
 						
 					{
 						ipHaarLienhart* iph = manage(new ipHaarLienhart());//width, height);
 						iph->setNoRec(3);
 						iph->setRec(0,1, y_, x_, h_, w_);
 						iph->setRec(1,-2,y_, x_+(w_/2), (h_/2),w_/2);
 						iph->setRec(2,-2, y_+h_/2,x_, (h_/2),w_/2);
 						iph->setModelSize(modelsize);
 						iph->setRegion(tregion);
 
 						weakfeatures[i ++] = iph;
 					}


	}

	// print("Number of feature %d\n",n_features);

	//////////////////////////////////////////////////////////////////
	FTrainer **boosting = manage_array(new FTrainer*[n_stages]);
	WeakLearner ***stump_trainers = manage_array(new WeakLearner**[n_stages]);
	StumpMachine ***stump_machines = manage_array(new StumpMachine**[n_stages]);

	for (int i=0;i<n_stages;i++)
	{
		stump_trainers[i] = manage_array(new WeakLearner*[n_weakClassifier[i]]);
		stump_machines[i]  = manage_array(new StumpMachine*[n_weakClassifier[i]]);

		for (int j=0;j<n_weakClassifier[i];j++)
		{
			stump_machines[i][j] = manage(new StumpMachine);
			stump_trainers[i][j] = manage(new StumpTrainer(stump_machines[i][j], n_features, weakfeatures));
			if (verbose)
				((StumpTrainer*)stump_trainers[i][j])->setBOption("verbose",true);
			// ((StumpTrainer*)stump_trainers[i][j])->
		}

		BoostingTrainer* trainer = manage(new BoostingTrainer);
		trainer->setBOption("boosting_by_sampling", true);
		trainer->setWeakLearners(n_weakClassifier[i], stump_trainers[i]);
		if (verbose)
			trainer->setBOption("verbose",true);
		boosting[i] = trainer;
	}

	CascadeTrainer *CT = manage(new CascadeTrainer());
	CT->setTrainers(boosting, n_stages, detection_rate);
	CT->setData(pDataSet,vDataSet,&iscandataset);
	CT->setPreprocessor(manage(new ipIntegral));

	if (verbose)
		CT->setBOption("verbose",true);
	CT->train();

	double *threshold = CT->getStageThreshold();
	if (verbose);
		print("Saving the model\n");

	CascadeMachine* CM = manage(new CascadeMachine());
	CM->resize(n_stages);
	CM->setSize(modelsize);
	for (int k=0;k<n_stages;k++)
	{
		CM->resize(k,n_weakClassifier[k]);
		for (int i=0;i<n_weakClassifier[k];i++)
		{
			CM->setMachine(k,i,stump_machines[k][i]);
			CM->setWeight(k,i, ((StumpTrainer*)stump_trainers[k][i])->getWeight());
		}

		CM->setThreshold(k,threshold[k]);
	}

	File f1;
	f1.open(modelfilename,"w");
	CM->saveFile(f1);
	f1.close();
	print("OK.\n");

	return 0;
}

