#include "torch5spro.h"

using namespace Torch;

void targetEncoding(int, FloatTensor *, bool one_hot_encoding = false);
bool loadDataSet(MemoryDataSet *, FloatTensor *, int *, FileList *);
bool testDataSet(MemoryDataSet *, GradientMachine *, bool);

int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* list_tensor_filename_train;
        char* list_tensor_filename_test;
	bool verbose;
	bool one_hot_encoding;
	bool stochastic;
	int nhu;
	int max_iter;
	float end_accuracy;
	int early_stopping;
	float learning_rate;
	float learning_rate_decay;
	float weight_decay;
	float momentum;
	long long seed;
	bool norm;
	int criterion_type;


	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Train a MLP from a list of tensor files");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("list of tensor files for training", &list_tensor_filename_train, "list of tensor files to load for training");
	cmd.addSCmdArg("list of tensor filesi for testing", &list_tensor_filename_test, "list of tensor files to load for testing");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-norm", &norm, false, "norm");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addBCmdOption("-one_hot_encoding", &one_hot_encoding, false, "one hot encoding");
	cmd.addBCmdOption("-stochastic", &stochastic, false, "perform stochastic gradient descent");
	cmd.addICmdOption("-criterion", &criterion_type, 0, "type of criterion (0=MSE, 1=NLL-2class, 2=MVSE)");
	cmd.addICmdOption("-nhu", &nhu, 3, "number of hidden units");
	cmd.addICmdOption("-max_iter", &max_iter, 10, "maximum number of iterations");
	cmd.addFCmdOption("-end_accuracy", &end_accuracy, 0.0001, "end accuracy");
	cmd.addICmdOption("-early_stopping", &early_stopping, 0, "maximum number of iterations for early stopping");
	cmd.addFCmdOption("-learning_rate", &learning_rate, 0.01, "learning rate");
	cmd.addFCmdOption("-learning_rate_decay", &learning_rate_decay, 0.0, "learning rate decay");
	cmd.addFCmdOption("-weight_decay", &weight_decay, 0.0, "weight decay");
	cmd.addFCmdOption("-momentum", &momentum, 0.0, "inertia momentum");
	cmd.addLLCmdOption("-seed", &seed, -1, "seed for random generator");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	//
	FileList *file_list_train = new FileList(list_tensor_filename_train);
	int n_targets = file_list_train->n_files;
	if(n_targets <= 1)
	{
		warning("Not enough targets (at least 2).");

		delete file_list_train;

		return 1;
	}

	FileList *file_list_test = new FileList(list_tensor_filename_test);
	if(n_targets != file_list_test->n_files)
	{
		warning("Different number of targets between the training and the test set :-(");

		delete file_list_test;
		delete file_list_train;

		return 1;
	}

        FloatTensor targets[n_targets];

	targetEncoding(n_targets, targets, one_hot_encoding);

	//
	int n_inputs;
	MemoryDataSet mdataset_train;
	MemoryDataSet mdataset_test;

	if(loadDataSet(&mdataset_train, targets, &n_inputs, file_list_train) == false)
	{
		warning("Impossible to load the training dataset.");

		delete file_list_train;

		return 1;
	}

	//
	delete file_list_train;

	if(loadDataSet(&mdataset_test, targets, &n_inputs, file_list_test) == false)
	{
		warning("Impossible to load the testing dataset.");

		delete file_list_test;

		return 1;
	}

	//
	delete file_list_test;

	// Normalization
	MeanVarNorm *mv_norm = NULL;

	if(norm)
	{
		print("Computing normalisation parameters ...\n");
		mv_norm = new MeanVarNorm(n_inputs, &mdataset_train);

		long n_examples_train = mdataset_train.getNoExamples();
		for (long i=0 ; i<n_examples_train ; i++)
		{
			Tensor *x = mdataset_train.getExample(i);
			mv_norm->forward(*x);
			x->copy(&mv_norm->getOutput());
		}

		long n_examples_test = mdataset_test.getNoExamples();
		for (long i=0 ; i<n_examples_test ; i++)
		{
			Tensor *x = mdataset_test.getExample(i);
			mv_norm->forward(*x);
			x->copy(&mv_norm->getOutput());
		}
	}

	//
	// init random generator
   	if(verbose) print("\nInitializing the random generator ...\n");
	if(seed == -1) 
	{
		seed = THRandom_seed();
		print("Random seed = %ld\n", seed);
	}
	else 
	{
		THRandom_manualSeed(seed);
		print("Manual seed = %ld\n", seed);
	}

	int n_outputs;

	if(one_hot_encoding) n_outputs = n_targets;
	else n_outputs = 1;

	//
	print("Building the MLP %d x %d x %d ...\n", n_inputs, nhu, n_outputs);
	GradientMachine *mlp = new MLP(n_inputs, nhu, n_outputs);
	print("Gradient Machine:\n");
	print(" n_inputs: %d\n", mlp->getNinputs());
	print(" n_outputs: %d\n", mlp->getNoutputs());

	//
	print("Calculating performance on the train set ...\n");
	testDataSet(&mdataset_train, mlp, one_hot_encoding);

	print("Calculating performance on the test set ...\n");
	testDataSet(&mdataset_test, mlp, one_hot_encoding);

	//
	Criterion *criterion = NULL;

	if(one_hot_encoding) criterion = new MSECriterion(n_outputs);
	else
	{
		if(criterion_type == 0) criterion = new MSECriterion(n_outputs);
		else if(criterion_type == 1) criterion = new TwoClassNLLCriterion();
		else if(criterion_type == 2) criterion = new MVSECriterion(n_outputs);
		else criterion = new MSECriterion(n_outputs);
	}

	//
	Trainer *trainer;

	if(stochastic)
	{
		trainer = new StochasticGradientTrainer();
		((StochasticGradientTrainer *) trainer)->setCriterion(criterion);
	}
	else
	{
		trainer = new GradientTrainer();
		((GradientTrainer *) trainer)->setCriterion(criterion);
	}

	trainer->setData(&mdataset_train);
	trainer->setMachine(mlp);

	trainer->setIOption("max iter", max_iter);
	trainer->setFOption("end accuracy", end_accuracy);
	trainer->setIOption("early stopping", early_stopping);
	trainer->setFOption("learning rate", learning_rate);
	trainer->setFOption("learning rate decay", learning_rate_decay);
	trainer->setFOption("weight decay", weight_decay);
	trainer->setFOption("momentum", momentum);

	trainer->train();

	//
	print("Calculating performance on the train set ...\n");
	testDataSet(&mdataset_train, mlp, one_hot_encoding);

	print("Calculating performance on the test set ...\n");
	testDataSet(&mdataset_test, mlp, one_hot_encoding);

	//
	print("Saving model file ...\n");
	File ofile;
	ofile.open("test.mlp", "w");
	mlp->saveFile(ofile);
	ofile.close();


	//
	print("Loading model file ...\n");
	delete mlp; mlp = NULL;
	mlp = new MLP();
	File ifile;
	ifile.open("test.mlp", "r");
	mlp->loadFile(ifile);
	ifile.close();

	mlp->prepare();

	//
	print("Calculating performance on the train set ...\n");
	testDataSet(&mdataset_train, mlp, one_hot_encoding);

	print("Calculating performance on the test set ...\n");
	testDataSet(&mdataset_test, mlp, one_hot_encoding);

	//
	delete trainer;
	delete criterion;
	delete mlp;
	if(norm)
		delete mv_norm;

        // OK
	return 0;
}

bool testDataSet(MemoryDataSet *mdataset, GradientMachine *mlp, bool one_hot_encoding)
{
	int true_accept = 0;
	int true_reject = 0;
	int false_reject = 0;
	int false_accept = 0;
	int nP = 0;
	int nN = 0;

	double mean_P_output = 0.0;
	double mean_N_output = 0.0;

	long n_examples = mdataset->getNoExamples();

	//
	for (long i=0 ; i<n_examples ; i++)
	{
		Tensor *x = mdataset->getExample(i);

		mlp->forward(*x);

		DoubleTensor *o = (DoubleTensor *) &mlp->getOutput();
		FloatTensor *t = (FloatTensor *) mdataset->getTarget(i);

		//o->print("output");
		//t->print("target");

		// here a target decoding class will be interesting to compute
		// all possible errors: targetDecoding(n_targets, targets, one_hot_encoding);
		if(one_hot_encoding)
		{
			// we should compute the confusion matrix !!!

		   	// find the argmax of the mlp output
			int argmax_o = 0;
			float max_o = o->get(0);
			for(int j = 1 ; j < mlp->getNoutputs() ; j++)
			{
			   	float z = o->get(j);
				if(z > max_o)
				{
					argmax_o = j;
					max_o = z;
				}
			}
			
		   	// find the argmax of the target
			int argmax_t = 0;
			float max_t = t->get(0);
			for(int j = 1 ; j < mlp->getNoutputs() ; j++)
			{
			   	float z = t->get(j);
				if(z > max_t)
				{
					argmax_t = j;
					max_t = z;
				}
			}

			if(argmax_o == argmax_t)
			{
				nP++;
				mean_P_output += o->get(argmax_o);
			}
			else
			{
				nN++;
				mean_N_output += o->get(argmax_o);
			}
		}
		else
		{
			if(t->get(0) > 0)
			{
				if(o->get(0) > 0) true_accept++;
				else false_reject++;
				nP++;
				mean_P_output += o->get(0);
			}

			if(t->get(0) < 0)
			{
				if(o->get(0) < 0) true_reject++;
				else false_accept++;
				nN++;
				mean_N_output += o->get(0);
			}
		}
	}

	if(one_hot_encoding)
	{
		print("\n");
		print("Number of correct classifications = %d / %d\n", nP, n_examples);
		mean_P_output /= (double) nP;
		print("   mean output of positives = %g\n", mean_P_output);

		print("\n");
		print("Number of incorrect classifications = %d / %d\n", nN, n_examples);
		mean_N_output /= (double) nN;
		print("   mean output of negatives = %g\n", mean_N_output);

		double err = ((double) nN / n_examples);
		print("   Error = %g\n", err*100.0);
	}
	else
	{
		print("\n");
		print("Number of positives = %d\n", nP);
		mean_P_output /= (double) nP;
		print("   mean output of positives = %g\n", mean_P_output);
		print("   true accept (true positive) = %d\n", true_accept);
		print("   false reject (false negative) = %d\n", false_reject);

		print("\n");
		print("Number of negatives = %d\n", nN);
		mean_N_output /= (double) nN;
		print("   mean output of negatives = %g\n", mean_N_output);
		print("   true reject (true negative) = %d\n", true_reject);
		print("   false accept (false positive) = %d\n", false_accept);

		double hter = (((double) false_accept / nN) + ((double) false_reject / nP)) / 2.0;
		print("   HTER = %g\n", hter*100.0);
	}

	return true;
}

void targetEncoding(int n_targets, FloatTensor *targets, bool one_hot_encoding)
{
	float P_target_value = +0.9;
	float N_target_value = -0.9;

	if(one_hot_encoding)
	{
		for (int i = 0; i < n_targets; i ++)
		{
			targets[i].resize(n_targets);
			targets[i].fill(N_target_value);
			targets[i].set(i, P_target_value);
		}
	}
	else
	{
		targets[0].resize(1);
		targets[0].fill(P_target_value);
		targets[1].resize(1);
		targets[1].fill(N_target_value);
	}

	for (int i = 0; i < n_targets; i ++) targets[i].sprint("Target for class %d", i);
}

bool loadDataSet(MemoryDataSet *mdataset, FloatTensor *targets, int *n_inputs, FileList *file_list)
{
   	print("Loading memory dataset ...\n");

	int n_examples = 0;
	int input_size = 0;

	print("Scanning %d files ...\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++)
	{
		print("Tensor file %s:\n", file_list->file_names[i]);

		TensorFile tf;

		if(tf.openRead(file_list->file_names[i]) == false) return false;

		const TensorFile::Header& header = tf.getHeader();

		print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
		print(" n_tensors:    [%d]\n", header.m_n_samples);
		print(" n_dimensions: [%d]\n", header.m_n_dimensions);
		print(" size[0]:      [%d]\n", header.m_size[0]);
		print(" size[1]:      [%d]\n", header.m_size[1]);
		print(" size[2]:      [%d]\n", header.m_size[2]);
		print(" size[3]:      [%d]\n", header.m_size[3]);

		tf.close();

		if(header.m_type != Tensor::Float)
		{
			warning("Unsupported tensor type (Float only).");

			return 1;
		}

		if(header.m_n_dimensions != 1)
		{
			warning("Unsupported dimensions (1 only).");

			return 1;
		}

		if(input_size == 0) input_size = header.m_size[0];
		else if(header.m_size[0] != input_size)
		{
			warning("Inconsistant input size (%d).", input_size);

			return 1;
		}

		n_examples += header.m_n_samples;
	}

	mdataset->reset(n_examples, Tensor::Double, true, Tensor::Float);

	print("Loading ...\n");
	long p = 0;
	for(int c = 0 ; c < file_list->n_files ; c++)
	{
		TensorFile tf;

		if(tf.openRead(file_list->file_names[c]) == false) return false;

		const TensorFile::Header& header = tf.getHeader();

		for(int j = 0 ; j < header.m_n_samples ; j++)
		{
			FloatTensor *tensor = NULL;
			tensor = (FloatTensor *)tf.load();

			Tensor* example = mdataset->getExample(p);
			example->resize(input_size);

			// to support type conversion
			example->copy(tensor);

			//tf.load(*example);

			delete tensor;

			mdataset->setTarget(p, &targets[c]);

			p++;
		}

		tf.close();
	}

	*n_inputs = input_size;

	print("Number of examples of size %d loaded: %d\n", input_size, n_examples);

	return true;
}
