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
	int max_examples;
	int max_features;
	int n_classifiers;
	// int n_rounds;
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


	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
	cmd.addICmdOption("-wc", &n_classifiers, 5, "number of weak classifiers");
	// cmd.addICmdOption("-nR", &n_rounds,50,"number of rounds");


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


	if (tensorList_p->process(p_tensor_files,target1,Tensor::Int)==false)
	{
		print("Error in reading +ve training patterns - Tensor list\n");
		return 1;
	}
	pDataSet = tensorList_p->getOutput();


	if (tensorList_n->process(n_tensor_files,target0,Tensor::Int)==false)
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
	MemoryDataSet *m_dataset = manage(new MemoryDataSet(n_examples, Tensor::Int, true, Tensor::Short));
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

	print("Prepare Boosting ...\n");

	ipIntegral *ipI = new ipIntegral();

	///////////////////////////// ADDED //////////////////////////////////////////////////////////////
	// CREATING THE IP for LBP and HLBP.
	ipLBP4R *ip_lbp = new ipLBP4R(1); // ipLBP4R, R = 1.
	CHECK_FATAL(ip_lbp->setBOption("ToAverage", false) == true);
	ipLBPBitmap *ip_hlbp = new ipLBPBitmap(ip_lbp);
	int nLabels = ip_lbp->getMaxLabel();
	int R = ip_lbp->getR();
	//--------------



	//DoubleTensor *temptensor;

	for (int e=0;e<n_examples;e++)
	{
		DoubleTensor* example = (DoubleTensor*)m_dataset->getExample(e);

		ip_hlbp->process(*example);
		ipI->process(ip_hlbp->getOutput(0));

		example->resize(height, width, nLabels);
		example->copy(&ipI->getOutput(0));
		if (e==0)
			Tprint(example);

		//bool t = ipI->process(*temptensor);
		//print("ok2");
		// 1. Calculate ipLBPBitmap.
		//example->resize(height,width,nLabels);
		//temptensor = (DoubleTensor*) &ip_hlbp->getOutput(0);
		//if(e==0)
		//Tprint(temptensor);
		//Tprint((DoubleTensor*)&ip_hlbp->getOutput(0));
		// 2. Calculate II of ipLBPBitmap output.
		//t = ipI->process(*temptensor);
		//temptensor = (DoubleTensor*) &ipI->getOutput(0);
		//if(e==0)
		//Tprint(temptensor);

		//m_dataset->getExample(e)->copy(temptensor);


		//temptensor = (DoubleTensor*)m_dataset->getExample(e);
		//if(e==0)
		//       Tprint(temptensor);
		//print("ok1");
		//bool t = ip_hlbp->process(*example);
		//bool t = ipI->process(*temptensor);
		//print("ok2");
		// 1. Calculate ipLBPBitmap.
		//example->resize(height,width,nLabels);
		//temptensor = (DoubleTensor*) &ip_hlbp->getOutput(0);
		//if(e==0)
		//Tprint(temptensor);
		//Tprint((DoubleTensor*)&ip_hlbp->getOutput(0));
		// 2. Calculate II of ipLBPBitmap output.
		//t = ipI->process(*temptensor);
		//temptensor = (DoubleTensor*) &ipI->getOutput(0);
		//if(e==0)
		//Tprint(temptensor);

		//m_dataset->getExample(e)->copy(temptensor);
	}

	// w_ and h_ start from 6, to reduce no. of features, and meaningful counts.


	if (width>0 && height >0)
	{
		n_features =0;
		//Edge feature 1

		for (int w_=6;w_<width-6;w_=w_+2)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;


		//Edge feature 2
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+2)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;


		//Line feature 1

		for (int w_=6;w_<width-6;w_=w_+3)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;


		//Line feature 2
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+3)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;

		//Line feature 3
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+4)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;

		//Line feature 4
		for (int w_=6;w_<width-6;w_=w_+4)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;

		for (int w_=6;w_<width-6;w_=w_+3)
			for (int h_=6;h_<height-6;h_=h_+3)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;

		for (int w_=6;w_<width-6;w_=w_+2)
			for (int h_=6;h_<height-6;h_=h_+2)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							n_features++;

	}

	print("Number of feature %d\n",n_features);


	spCore **weakfeatures =NULL;
	weakfeatures = manage_array(new spCore* [n_features]);


	bool t;
	TensorSize *modelsize = manage(new TensorSize(height,width));
	TensorRegion *tregion = manage(new TensorRegion(0,0,height,width));
	if (width>0 && height >0)
	{
		int i =0;
		//Edge feature 1

		for (int w_=6;w_<width-6;w_=w_+2)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);

							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//  t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//   t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, y_,x_+(w_/2), h_,(w_/2));
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}



		//Edge feature 2
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+2)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//  t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							// t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+(h_/2), w_,(h_/2));
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2,  y_+(h_/2),x_,(h_/2),w_);
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;
						}

		//  n_features++;


		//Line feature 1

		for (int w_=6;w_<width-6;w_=w_+3)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//    t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//   t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-3, x_+(w_/3), y_, (w_/3),h_);

							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-3, y_,x_+(w_/3), h_, (w_/3));
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}

		// n_features++;



		//Line feature 2
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+3)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//       t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//      t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-3, x_, y_+h_/3, w_,h_/3);

							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-3,  y_+h_/3, x_,h_/3,w_);
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}


		// n_features++;

		//Line feature 3
		for (int w_=6;w_<width-6;w_++)
			for (int h_=6;h_<height-6;h_=h_+4)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//    t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//   t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_, y_+h_/4, w_,h_/2);


							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, y_+h_/4,x_ ,h_/2,w_);
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}

		// n_features++;

		//Line feature 4
		for (int w_=6;w_<width-6;w_=w_+4)
			for (int h_=6;h_<height-6;h_++)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//    t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//   t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/4), y_, (w_/2),h_);

							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/4), h_, (w_/2));
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}


		//n_features++;

		for (int w_=6;w_<width-6;w_=w_+3)
			for (int h_=6;h_<height-6;h_=h_+3)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(2);
							//             t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//                 t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/3), y_+h_/3, (w_/3),h_/3);


							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, y_+(h_/3), x_+w_/3, (h_/3),w_/3);
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

						}

		// n_features++;

		for (int w_=6;w_<width-6;w_=w_+2)
			for (int h_=6;h_<height-6;h_=h_+2)
				for (int x_=0;x_<width-w_;x_++)
					for (int y_=0;y_<height-h_;y_++)
						for (int labelNo = 0; labelNo < nLabels; labelNo++)
							//  n_features++;
						{
							weakfeatures[i] = manage(new ipHLBPLienhart());//width, height);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setNoRec(3);
							//      t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, x_, y_, w_, h_);
							//      t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2, x_+(w_/2), y_, (w_/2),h_/2);
							//        t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(2,-2, x_, y_+h_/2, (w_/2),h_/2);

							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(0,1, y_, x_, h_, w_);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(1,-2,y_, x_+(w_/2), (h_/2),w_/2);
							t= ((ipHLBPLienhart *)weakfeatures[i])->setRec(2,-2, y_+h_/2,x_, (h_/2),w_/2);
							((ipHLBPLienhart *)weakfeatures[i])->setModelSize(*modelsize);
							((ipHLBPLienhart *)weakfeatures[i])->setRegion(TensorRegion(0,0,labelNo,height,width,1));
							i++;

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
		stump_trainers[i]->setBOption("verbose",true);
	}
	boosting.setBOption("boosting_by_sampling",true);
	boosting.setWeakLearners(n_classifiers, stump_trainers);
	boosting.setData(m_dataset);
	boosting.setBOption("verbose",true);
	boosting.train();



//check the model if it is trained properly

//for (int e=0;e<n_examples;e++)
//    {
//
//        Tensor *ttensor = (Tensor*)m_dataset->getExample(e);
//        double s;
//        s=0;
//        for(int i=0;i<n_classifiers;i++)
//        {
//                stump_machines[i]->forward(*ttensor);
//                DoubleTensor *t_output = (DoubleTensor *) &stump_machines[i]->getOutput();
//
//            s +=  stump_trainers[i]->getWeight()*(*t_output)(0);
//        }
//        print("Score %f\n",s);
//
//    }









	//Saving the model.


	CascadeMachine* CM = manage(new CascadeMachine());
	CM->resize(1);
	CM->setSize(*modelsize);
	CM->resize(0,n_classifiers);

	for (int i=0;i<n_classifiers;i++)
	{
		t= CM->setMachine(0,i,stump_machines[i]);
		t= CM->setWeight(0,i,stump_trainers[i]->getWeight());


	}

	t=CM->setThreshold(0,0.0);


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

	File f1;
	t=f1.open("model.wsm","w");
	t=CM->saveFile(f1);
	f1.close();

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

		//   ofile.printf("%g\n",cascade->getConfidence());
		print("CONFIDENCE = %f\n", cascade->getConfidence());
		//  count += cascade->isPattern() ? 1 : 0;
	}


	// OK
	print("OK.\n");



	return 0;
}

