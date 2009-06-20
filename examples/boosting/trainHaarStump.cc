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
	int max_examples;
	int max_features;
	int n_classifiers;
	// int n_rounds;
	bool verbose;

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
	// cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	ShortTensor *target1 = manage(new ShortTensor(1));
	target1->fill(1);
	ShortTensor *target0 = manage(new ShortTensor(1));
	target0->fill(0);

	TensorList *tensorList_p= manage(new TensorList()); //for positive
	if (tensorList_p->process(p_tensor_files,target1,Tensor::Int)==false)
	{
		print("Error in reading +ve training patterns - Tensor list\n");
		return 1;
	}
	DataSet *pDataSet = tensorList_p->getOutput();

	TensorList *tensorList_n= manage(new TensorList()); //for validation
	if (tensorList_n->process(n_tensor_files,target0,Tensor::Int)==false)
	{
		print("Error in reading -ve patterns - Tensor list\n");
		return 1;
	}
	DataSet *nDataSet = tensorList_n->getOutput();

	Tensor *st = pDataSet->getExample(0);
	int width = st->size(1);
	int height = st->size(0);

	int n_examples = pDataSet->getNoExamples() + nDataSet->getNoExamples();
	int pexamples = pDataSet->getNoExamples();

	MemoryDataSet *m_dataset = manage(new MemoryDataSet(n_examples, Tensor::Int, true, Tensor::Short));
	for (int i=0;i<pDataSet->getNoExamples();i++)
	{
		m_dataset->getExample(i)->resize(height, width);
		m_dataset->getExample(i)->copy(pDataSet->getExample(i));
		m_dataset->setTarget(i, target1);
	}
	for (int i=0;i<nDataSet->getNoExamples();i++)
	{
		m_dataset->getExample(i+pexamples)->resize(height, width);
		m_dataset->getExample(i+pexamples)->copy(nDataSet->getExample(i));
		m_dataset->setTarget(i+pexamples, target0);
	}

	if (verbose)
		print("Prepare Boosting ...\n");

	ipIntegral *ipI = manage(new ipIntegral());
	for (int e=0;e<n_examples;e++)
	{
		DoubleTensor* example = (DoubleTensor*)m_dataset->getExample(e);
		ipI->process(*example);
		example->copy(&ipI->getOutput(0));
	}

	int n_features = 0;
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

	if (verbose)
		print("Number of feature %d\n",n_features);

	spCore **weakfeatures = manage_array(new spCore* [n_features]);

	TensorSize modelsize(height,width);
	TensorRegion tregion(0,0,height,width);
	if (width>0 && height >0)
	{
		int i =0;
		//Edge feature 1

		for (int w_=2;w_<width;w_=w_+2)
			for (int h_ =1;h_<height;h_++)
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

		//  n_features++;

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

		// n_features++;

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

		// n_features++;

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

		// n_features++;

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

		// n_features++;

		for (int w_=2;w_<width;w_=w_+2)
			for (int h_ =2;h_<height;h_=h_+2)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						//  n_features++;
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


	///////////////////////////////////////////////////
	BoostingTrainer boosting;
	WeakLearner **stump_trainers = manage_array(new WeakLearner*[n_classifiers]);
	StumpMachine **stump_machines = manage_array(new StumpMachine*[n_classifiers]);

	for ( int i = 0; i < n_classifiers; i++)
	{
		stump_machines[i] = manage(new StumpMachine());

		stump_trainers[i] = manage(new StumpTrainer(stump_machines[i], n_features, weakfeatures));
		if (verbose)
			stump_trainers[i]->setBOption("verbose",true);
	}
	boosting.setBOption("boosting_by_sampling",true);
	boosting.setWeakLearners(n_classifiers, stump_trainers);
	boosting.setData(m_dataset);
	if (verbose)
		boosting.setBOption("verbose",true);
	boosting.train();


	//Saving the model.
	if (verbose)
		print("Saving the model\n");

	CascadeMachine* CM = manage(new CascadeMachine());
	CM->resize(1);
	CM->setSize(modelsize);
	CM->resize(0,n_classifiers);

	for (int i=0;i<n_classifiers;i++)
	{
		CM->setMachine(0,i,stump_machines[i]);
		CM->setWeight(0,i,stump_trainers[i]->getWeight());
	}

	CM->setThreshold(0,0.0);


	File f1;
	f1.open(modelfilename,"w");
	CM->saveFile(f1);
	f1.close();

//    CascadeMachine* cascade = manage((CascadeMachine*)Torch::loadMachineFromFile("model.wsm"));
//    if (cascade == 0)
//    {
//        print("ERROR: loading model [%s]!\n", "model.wsm");
//        return 1;
//    }
//    for (int i=0;i<n_examples;i++)
//    {
//        Tensor *st = (Tensor*)m_dataset->getExample(i);
//
//        cascade->setRegion(*tregion);
//        if (cascade->forward(*st) == false)
//        {
//
//            delete cascade;
//            return 1;
//        }
//
//        //   ofile.printf("%g\n",cascade->getConfidence());
//        print("CONFIDENCE = %f\n", cascade->getConfidence());
//        //  count += cascade->isPattern() ? 1 : 0;
//    }
//

	// OK
	print("OK.\n");

	return 0;
}

