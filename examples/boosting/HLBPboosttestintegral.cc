#include "torch5spro.h"

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



	ShortTensor *target1 = manage(new ShortTensor(1));
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

	int w = st->size(1);
	int h = st->size(0);

	File ofile;
	ofile.open("scores.out","w");
	int n_examples = mdataset->getNoExamples();
	int count =0;
	ipIntegral *ipI = manage(new ipIntegral());
	DoubleTensor *temptensor = manage(new DoubleTensor());

	//--------------------------------

	///////////////////////////// ADDED //////////////////////////////////////////////////////////////
	// CREATING THE IP for LBP and HLBP.
	ipLBP4R *ip_lbp = new ipLBP4R(1); // ipLBP4R, R = 1.
	CHECK_FATAL(ip_lbp->setBOption("ToAverage", false) == true);
	ipLBPBitmap *ip_hlbp = new ipLBPBitmap(ip_lbp);
	int nLabels = ip_lbp->getMaxLabel();
	//--------------------------------

	for (int e=0;e<n_examples;e++)
	{

		DoubleTensor* example = (DoubleTensor*)mdataset->getExample(e);
		ip_hlbp->process(*example);
		ipI->process(ip_hlbp->getOutput(0));
		example->resize(height, width, nLabels);
		example->copy(&ipI->getOutput(0));

		//print("1.\n");
		//----------------------------------------
		//temptensor = (DoubleTensor*)mdataset->getExample(e);
		//print("2.\n");
		//bool t = ip_hlbp->process(*temptensor);
		//print("3.\n");
		// 1. Calculate ipLBPBitmap.
		//temptensor->resize(h,w,nLabels);
		//print("4.\n");
		//temptensor = (DoubleTensor*) &ip_hlbp->getOutput(0);
		//print("5.\n");
		// 2. Calculate II of ipLBPBitmap output.
		//t = ipI->process(*temptensor);
		//print("6.\n");
		//temptensor = (DoubleTensor*) &ipI->getOutput(0);
		//print("7.\n");
		//mdataset->getExample(e)->copy(temptensor);
		//print("8.\n");
		//----------------------------------------
		//temptensor = (DoubleTensor*)mdataset->getExample(e);
		//bool t = ipI->process(*temptensor);
		//temptensor = (DoubleTensor*) &ipI->getOutput(0);
		//mdataset->getExample(e)->copy(temptensor);
	}
	print("\n\n\n\n\n...................Loading the model\n");
	CascadeMachine* cascade = manage((CascadeMachine*)Torch::loadMachineFromFile(modelfilename));
	if (cascade == 0)
	{
		print("ERROR: loading model [%s]!\n", "model.wsm");
		return 1;
	}

	TensorRegion *tr = manage(new TensorRegion(0,0,h,w));// What does this do ?? Does it affect u_z ?
	for (int i=0;i<n_examples;i++)
	{
		Tensor *st = mdataset->getExample(i);

		cascade->setRegion(*tr);// What does this do ? Does it affect u_z ?
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

