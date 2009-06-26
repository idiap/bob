#include "torch5spro.h"

using namespace Torch;

bool loadDataSet(MemoryDataSet *, int *, FileList *);
bool testDataSet(MemoryDataSet *, ProbabilityDistribution *);

int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* list_tensor_filename_train;
        char* list_tensor_filename_test;
	bool verbose;
	int ng;
	int max_iter;
	float end_accuracy;
	float map;
	float flooring;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Train a GMM from a list of tensor files");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("list of tensor files for training", &list_tensor_filename_train, "list of tensor files to load for training");
	cmd.addSCmdArg("list of tensor filesi for testing", &list_tensor_filename_test, "list of tensor files to load for testing");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addICmdOption("-ng", &ng, 3, "number of gaussians");
	cmd.addICmdOption("-max_iter", &max_iter, 10, "maximum number of iterations");
	cmd.addFCmdOption("-end_accuracy", &end_accuracy, 0.0001, "end accuracy");
	cmd.addFCmdOption("-map", &map, 0.5, "adaptation factor");
	cmd.addFCmdOption("-floor", &flooring, 0.001, "flooring factor");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	//
	FileList *file_list_train = new FileList(list_tensor_filename_train);
	FileList *file_list_test = new FileList(list_tensor_filename_test);

	//
	int n_inputs;
	MemoryDataSet mdataset_train;
	MemoryDataSet mdataset_test;

	if(loadDataSet(&mdataset_train, &n_inputs, file_list_train) == false)
	{
		warning("Impossible to load the training dataset.");

		delete file_list_train;

		return 1;
	}

	//
	delete file_list_train;

	if(loadDataSet(&mdataset_test, &n_inputs, file_list_test) == false)
	{
		warning("Impossible to load the testing dataset.");

		delete file_list_test;

		return 1;
	}

	//
	delete file_list_test;

	// Normalization
	print("Computing normalisation parameters ...\n");
	MeanVarNorm *norm = new MeanVarNorm(n_inputs, &mdataset_train);

	long n_examples_train = mdataset_train.getNoExamples();
	for (long i=0 ; i<n_examples_train ; i++)
	{
		Tensor *x = mdataset_train.getExample(i);
		norm->forward(*x);
		x->copy(&norm->getOutput());
	}

	long n_examples_test = mdataset_test.getNoExamples();
	for (long i=0 ; i<n_examples_test ; i++)
	{
		Tensor *x = mdataset_test.getExample(i);
		norm->forward(*x);
		x->copy(&norm->getOutput());
	}

	//
	//unsigned long seed = THRandom_seed();
	//print("Seed = %ld\n", seed);
	THRandom_manualSeed(950305);

	//
	print("Building the KMeans %d x %d ...\n", n_inputs, ng);
	MultiVariateMeansDistribution *kmean = new MultiVariateMeansDistribution(n_inputs, ng);
	//kmean->shuffle();
	kmean->setMeans(&mdataset_train);
	kmean->prepare();
	if(verbose) kmean->print();

	print("Building the GMM %d x %d ...\n", n_inputs, ng);
	MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution(n_inputs, ng);
	gmm->setBOption("log mode", true);
	gmm->setBOption("variance update", true);
	gmm->prepare();
	//gmm->shuffle();
	if(verbose) gmm->print();

	//
	EMTrainer *trainer = new EMTrainer();

	trainer->setData(&mdataset_train);
	trainer->setIOption("max iter", max_iter);
	trainer->setFOption("end accuracy", end_accuracy);

	print("Training KMeans ...\n");
	trainer->setMachine(kmean);
	trainer->train();
	if(verbose) kmean->print();

	print("Training GMM ...\n");
	//gmm->shuffle();
	gmm->setMeans(kmean->getMeans());
	gmm->setVariances(norm->m_stdv, flooring);
	if(verbose) gmm->print();
	trainer->setMachine(gmm);
	trainer->train();
	if(verbose) gmm->print();

	//
	print("Calculating performance on the train set ...\n");
	testDataSet(&mdataset_train, gmm);

	print("Calculating performance on the test set ...\n");
	testDataSet(&mdataset_test, gmm);

	//
	print("Building the MAP-GMM ...\n");
	MultiVariateMAPDiagonalGaussianDistribution *mapgmm = new MultiVariateMAPDiagonalGaussianDistribution(gmm);
	mapgmm->setBOption("log mode", true);
	mapgmm->setFOption("map factor", map);
	mapgmm->prepare();
	if(verbose) mapgmm->print();

	//
	print("Calculating performance on the train set with MAP GMM ...\n");
	testDataSet(&mdataset_train, mapgmm);

	print("Calculating performance on the test set with MAP GMM ...\n");
	testDataSet(&mdataset_test, mapgmm);

	print("Training MAP-GMM ...\n");
	mapgmm->setVarianceFlooring(norm->m_stdv, flooring);
	trainer->setMachine(mapgmm);
	trainer->train();
	if(verbose) mapgmm->print();

	//
	print("Calculating performance on the train set with MAP GMM after training ...\n");
	testDataSet(&mdataset_train, mapgmm);

	print("Calculating performance on the test set with MAP GMM after training ...\n");
	testDataSet(&mdataset_test, mapgmm);

	delete mapgmm;

	//
	print("Saving model file ...\n");
	File ofile;
	ofile.open("test.gmm", "w");
	gmm->saveFile(ofile);
	ofile.close();

	//
	print("Loading model file ...\n");
	delete gmm; gmm= NULL;
	gmm = new MultiVariateDiagonalGaussianDistribution();
	File ifile;
	ifile.open("test.gmm", "r");
	gmm->loadFile(ifile);
	ifile.close();
	gmm->setBOption("log mode", true);
	gmm->prepare();

	//
	print("Calculating performance on the train set ...\n");
	testDataSet(&mdataset_train, gmm);

	print("Calculating performance on the test set ...\n");
	testDataSet(&mdataset_test, gmm);

	//
	delete trainer;
	delete gmm;
	delete norm;

        // OK
	return 0;
}

bool testDataSet(MemoryDataSet *mdataset, ProbabilityDistribution *gmm)
{
	double mean_nll = 0.0;

	long n_examples = mdataset->getNoExamples();

	for (long i=0 ; i<n_examples ; i++)
	{
		Tensor *x = mdataset->getExample(i);

		gmm->forward(*x);

		DoubleTensor *o = (DoubleTensor *) &gmm->getOutput();

		mean_nll += o->get(0);
	}

	print("\n");
	print("Number of samples = %d\n", n_examples);
	mean_nll /= (double) n_examples;
	print("   mean nll = %g\n", mean_nll);

	return true;
}

bool loadDataSet(MemoryDataSet *mdataset, int *n_inputs, FileList *file_list)
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

	mdataset->reset(n_examples, Tensor::Double);

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

			p++;
		}

		tf.close();
	}

	*n_inputs = input_size;

	print("Number of examples of size %d loaded: %d\n", input_size, n_examples);

	return true;
}
