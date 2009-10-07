#include "torch5spro.h"

using namespace Torch;

const char *help = "\
progname: gmm.cc\n\
code2html: This program perform the training and the test of the Gaussian Mixture Model (GMM).\n\
version: Torch5spro\n\
date: June 2009\n\
author: Sebastien Marcel (marcel@idiap.ch) 2009\n";

using namespace Torch;

int checkFiles(int n_files, char **file_names, bool verbose = false);
bool loadDataSet(MemoryDataSet *, int *, FileList *, bool verbose = false);
bool loadDataSet(MemoryDataSet *, int *, char *, bool verbose = false);
bool testDataSet(MemoryDataSet *, ProbabilityDistribution *);

char *strFilename(char *filename, char dirsep = '/');
char *strBasename(char *filename, char extsep = '.');
   
int main(int argc, char **argv)
{
	//
	char *list_tensor_filename;
	char *input_model_filename;
	char *output_model_filename;
	char *norm_model_filename;
	char *score_filename;
	char *model_string;

	//
	int n_gaussians;

	//
	float end_accuracy;
	float flooring;
	float prior_weights;
	float map;
	bool variance_adapt;
	bool weight_adapt;
	int max_iter_kmeans;
	int max_iter_gmm;
	bool kmeans_random_init;
	long long seed;
	bool norm;

	//
	bool verbose;


	//
	// The command-line
	CmdLine cmd;

	// don't save the file "cmd.log"
	cmd.setBOption("write log", false);

	// Put the help line at the beginning
	cmd.info(help);

	// Train mode
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");

	cmd.addText("\nModel Options:");
	cmd.addICmdOption("-n_gaussians", &n_gaussians, 10, "number of Gaussians");

	cmd.addText("\nLearning Options:");
	cmd.addBCmdOption("-randk", &kmeans_random_init, false, "initialize randomly the Kmean otherwise initialize from data");
	cmd.addBCmdOption("-norm", &norm, false, "normalize the datas");
	cmd.addSCmdOption("-normfile", &norm_model_filename, "", "normalization model filename");
	cmd.addLLCmdOption("-seed", &seed, -1, "seed for random generator");
	cmd.addICmdOption("-iterk", &max_iter_kmeans, 25, "max number of iterations of KMeans");
	cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	cmd.addFCmdOption("-flooring", &flooring, 0.001, "variance flooring threshold");
	cmd.addFCmdOption("-e", &end_accuracy, 0.00001, "end of accuracy");
	cmd.addFCmdOption("-prior", &prior_weights, 0.001, "initial weight of Gaussians prior learning");

	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-save", &output_model_filename, "model-train.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Retrain mode
	cmd.addMasterSwitch("--retrain");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");

	cmd.addText("\nLearning Options:");
	cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	cmd.addFCmdOption("-flooring", &flooring, 0.001, "variance flooring threshold");
	cmd.addFCmdOption("-e", &end_accuracy, 0.00001, "end of accuracy");
	cmd.addFCmdOption("-prior", &prior_weights, 0.001, "initial weight of Gaussians prior learning");

	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-save", &output_model_filename, "model-retrain.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Adaptation mode
	cmd.addMasterSwitch("--adapt");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");

	cmd.addText("\nLearning Options:");
	cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	cmd.addFCmdOption("-flooring", &flooring, 0.001, "variance flooring threshold");
	cmd.addFCmdOption("-e", &end_accuracy, 0.00001, "end of accuracy");
	cmd.addFCmdOption("-map", &map, 0.5, "adaptation factor");
	cmd.addBCmdOption("-vadapt", &variance_adapt, false, "adapt variances");
	cmd.addBCmdOption("-wadapt", &weight_adapt, false, "adapt weights");
	cmd.addFCmdOption("-prior", &prior_weights, 0.001, "initial weight of Gaussians prior learning");

	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-save", &output_model_filename, "model-adapt.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Test mode
	cmd.addMasterSwitch("--test");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");
	cmd.addSCmdArg("scores output file", &score_filename, "output score filename");

	cmd.addText("\nMisc Options:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Merge mode
	cmd.addMasterSwitch("--merge");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("input models", &model_string, "input model file in double-quotes: \"model1 model2 ...\"");
	
	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-save", &output_model_filename, "model-merge.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Read mode
	cmd.addMasterSwitch("--read");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");

	cmd.addText("\nMisc Options:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Read mode
	cmd.addMasterSwitch("--copy");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model in", &input_model_filename, "input model filename");
	cmd.addSCmdArg("model out", &output_model_filename, "output model filename");

	cmd.addText("\nMisc Options:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Read the command line
	int mode = cmd.read(argc, argv);

	// set the working directory
	//cmd.setWorkingDirectory(dir_name);

	//==================================================================== 
	//=================== Training Mode  =================================
	//==================================================================== 
	if(mode == 0)
	{
	   	if(verbose) print("\nTraining mode\n");

		int n_inputs;
		MemoryDataSet mdataset_train;

		// load and check the list of files
		FileList *file_list_train = new FileList(list_tensor_filename);

		// load in memory
   		if(verbose) print("\nLoading memory dataset ...\n");
		if(loadDataSet(&mdataset_train, &n_inputs, file_list_train, verbose) == false)
		{
			warning("Impossible to load the training dataset.");

			delete file_list_train;

			return 1;
		}
		delete file_list_train;

		// normalize the data if needed
   		if(verbose) print("\nComputing normalization parameters from data ...\n");
		MeanVarNorm *mv_norm = new MeanVarNorm(n_inputs, &mdataset_train);
		if(norm)
		{	
   			if(verbose) print("\nNormalizing the dataset ...\n");

			long n_examples_train = mdataset_train.getNoExamples();
			for (long i=0 ; i<n_examples_train ; i++)
			{
				Tensor *x = mdataset_train.getExample(i);
				mv_norm->forward(*x);
				x->copy(&mv_norm->getOutput());
			}

			if(strcmp(norm_model_filename, ""))
			{
				print("Saving normalization file ...\n");
				File ofile;
				ofile.open(norm_model_filename, "w");
				mv_norm->saveFile(ofile);
				ofile.close();
			}
		}
		
		MeanVarNorm *vflooring = new MeanVarNorm(n_inputs, &mdataset_train);

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

		//
		MultiVariateMeansDistribution *kmean = NULL;
		if(max_iter_kmeans > 0)
		{
			print("Building the Kmeans %d x %d ...\n", n_inputs, n_gaussians);
			kmean = new MultiVariateMeansDistribution(n_inputs, n_gaussians);

			if(kmeans_random_init)
			{
   				if(verbose) print("\nInitializing the Kmeans randomly ...\n");
				kmean->shuffle();
			}
			else
			{
   				if(verbose) print("\nInitializing the Kmeans from data ...\n");
				kmean->setMeans(&mdataset_train);
				kmean->setFOption("min weights", prior_weights);
			}
			kmean->prepare();
			if(verbose) kmean->print();
		}

		//
		print("Building the GMM %d x %d ...\n", n_inputs, n_gaussians);
		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution(n_inputs, n_gaussians);
		gmm->setBOption("log mode", true);
		gmm->setBOption("variance update", true);
		gmm->setFOption("min weights", prior_weights);
		gmm->prepare();

		//
   		if(verbose) print("\nBuilding EM trainer ...\n");
		EMTrainer *trainer = new EMTrainer();

		trainer->setData(&mdataset_train);
		trainer->setFOption("end accuracy", end_accuracy);

		if(max_iter_kmeans > 0)
		{
			print("Training KMeans ...\n");
			trainer->setMachine(kmean);
			trainer->setIOption("max iter", max_iter_kmeans);

			trainer->train();
			
			if(verbose) kmean->print();

   			if(verbose) print("\nInitializing the GMM from Kmeans ...\n");
			gmm->setMeans(kmean->getMeans());
			gmm->setVariances(kmean->getVariances());
			gmm->setVarianceFlooring(vflooring->m_stdv, flooring);
			if(verbose) gmm->print();
		}
		else 
		{
   			if(verbose) print("\nInitializing the GMM randomly ...\n");
			gmm->shuffle();
			gmm->setVarianceFlooring(vflooring->m_stdv, flooring);
			if(verbose) gmm->print();
		}

		//
		print("Training GMM ...\n");
		trainer->setMachine(gmm);
		trainer->setIOption("max iter", max_iter_gmm);
		
		trainer->train();
		
		if(verbose) gmm->print();

		//
   		print("\nTesting the GMM ...\n");
		testDataSet(&mdataset_train, gmm);
		  
		//
		print("Saving model file ...\n");
		File ofile;
		ofile.open(output_model_filename, "w");
		if(norm)
		{
   			if(verbose) print("\nUn-normalizing the parameters of the GMM ...\n");

			double *means_ = gmm->m_parameters->getDarray("means");
			double *variances_ = gmm->m_parameters->getDarray("variances");

			for(int j = 0 ; j < n_gaussians ; j++)
			{
				for(int k = 0 ; k < n_inputs ; k++)
				{
				   	double z = means_[k];
					means_[k] = z * mv_norm->m_stdv[k] + mv_norm->m_mean[k];

					double zz = mv_norm->m_stdv[k];
					variances_[k] *= zz*zz;
				}

				means_ += n_inputs;
				variances_ += n_inputs;
			}
			if(verbose) gmm->print();
		}
		gmm->saveFile(ofile);
		ofile.close();

		//
		delete trainer;
		delete gmm;
		if(kmean != NULL) delete kmean;
		delete mv_norm;
		delete vflooring;
	}

	//==================================================================== 
	//=================== Retraining Mode  ===============================
	//==================================================================== 
	if(mode == 1)
	{
	   	if(verbose) print("Retraining mode\n");

		int n_inputs;
		MemoryDataSet mdataset_retrain;

		//
		FileList *file_list_retrain = new FileList(list_tensor_filename);

   		if(verbose) print("\nLoading memory dataset ...\n");
		if(loadDataSet(&mdataset_retrain, &n_inputs, file_list_retrain, verbose) == false)
		{
			warning("Impossible to load the training dataset.");

			delete file_list_retrain;

			return 1;
		}
		delete file_list_retrain;

		// normalize the data if needed
   		if(verbose) print("\nComputing normalization parameters from loaded data ...\n");
		MeanVarNorm *vflooring = new MeanVarNorm(n_inputs, &mdataset_retrain);

		//
		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution();

		File ifile;
		ifile.open(input_model_filename, "r");
		gmm->loadFile(ifile);
		ifile.close();
		if(verbose) gmm->print();
	
		gmm->setBOption("log mode", true);
		gmm->setBOption("variance update", true);
		gmm->setFOption("min weights", prior_weights);
		gmm->prepare();

		//
   		if(verbose) print("\nBuilding EM trainer ...\n");
		EMTrainer *trainer = new EMTrainer();

		trainer->setData(&mdataset_retrain);
		trainer->setFOption("end accuracy", end_accuracy);

		gmm->setVarianceFlooring(vflooring->m_stdv, flooring);
		if(verbose) gmm->print();

		//
		print("Training GMM ...\n");
		trainer->setMachine(gmm);
		trainer->setIOption("max iter", max_iter_gmm);
		
		trainer->train();
		
		if(verbose) gmm->print();

		//
   		print("\nTesting the GMM ...\n");
		testDataSet(&mdataset_retrain, gmm);
		  
		//
		print("Saving model file ...\n");
		File ofile;
		ofile.open(output_model_filename, "w");
		gmm->saveFile(ofile);
		ofile.close();

		//
		delete trainer;
		delete gmm;
		delete vflooring;
	}

	//==================================================================== 
	//=================== Adaptation Mode  ===============================
	//==================================================================== 
	if(mode == 2)
	{
	   	if(verbose) print("Adaptation mode\n");

		int n_inputs;
		MemoryDataSet mdataset_adapt;

		//
		FileList *file_list_adapt = new FileList(list_tensor_filename);

   		if(verbose) print("\nLoading memory dataset ...\n");
		if(loadDataSet(&mdataset_adapt, &n_inputs, file_list_adapt, verbose) == false)
		{
			warning("Impossible to load the training dataset.");

			delete file_list_adapt;

			return 1;
		}
		delete file_list_adapt;

		// normalize the data if needed
   		if(verbose) print("\nComputing normalization parameters from loaded data ...\n");
		MeanVarNorm *vflooring = new MeanVarNorm(n_inputs, &mdataset_adapt);

		//
		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution();

		File ifile;
		ifile.open(input_model_filename, "r");
		gmm->loadFile(ifile);
		ifile.close();
		if(verbose) gmm->print();
	
		gmm->setBOption("log mode", true);
		gmm->setBOption("variance update", true);
		gmm->prepare();

		MultiVariateMAPDiagonalGaussianDistribution *mapgmm = new MultiVariateMAPDiagonalGaussianDistribution(gmm);
		mapgmm->setBOption("log mode", true);
		mapgmm->setFOption("map factor", map);
		mapgmm->setBOption("variance adapt", variance_adapt);
		mapgmm->setBOption("weight adapt", weight_adapt);
		mapgmm->setFOption("min weights", prior_weights);
		mapgmm->prepare();

		//
   		if(verbose) print("\nBuilding EM trainer ...\n");
		EMTrainer *trainer = new EMTrainer();

		trainer->setData(&mdataset_adapt);
		trainer->setFOption("end accuracy", end_accuracy);

		mapgmm->setVarianceFlooring(vflooring->m_stdv, flooring);
		if(verbose) mapgmm->print();

		//
		print("MAP Training GMM ...\n");
		trainer->setMachine(mapgmm);
		trainer->setIOption("max iter", max_iter_gmm);
		
		trainer->train();
		
		if(verbose) mapgmm->print();

		//
   		print("\nTesting the GMM ...\n");
		testDataSet(&mdataset_adapt, mapgmm);
		  
		//
		print("Saving model file ...\n");
		File ofile;
		ofile.open(output_model_filename, "w");
		mapgmm->saveFile(ofile);
		ofile.close();

		//
		delete trainer;
		delete mapgmm;
		delete gmm;
		delete vflooring;
	}

	//==================================================================== 
	//====================== Testing Mode  ===============================
	//==================================================================== 
	if(mode == 3)
	{
		//
		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution();

		File ifile;
		ifile.open(input_model_filename, "r");
		gmm->loadFile(ifile);
		ifile.close();
		if(verbose) gmm->print();
	
		//
		FileList *file_list_test = new FileList(list_tensor_filename);

		int n_reminding_files = file_list_test->n_files;
		n_reminding_files = checkFiles(file_list_test->n_files, file_list_test->file_names, false);

		File ofile;
		ofile.open(score_filename, "w");

		for(int i = 0 ; i < n_reminding_files ; i++)
		{
		   	if(verbose) print("testing file %s\n", file_list_test->file_names[i]);
			
			char *temp = strFilename(file_list_test->file_names[i]);
			char *basename = strBasename(temp);
		   	
			//print("basename %s\n", basename);
		
			// Load in memory
   			if(verbose) print("\nLoading memory dataset ...\n");
			MemoryDataSet *mdataset = new MemoryDataSet;
			int n_inputs;
			if(loadDataSet(mdataset, &n_inputs, file_list_test->file_names[i], verbose) == false)
			{
				warning("Impossible to load the testing dataset.");

				delete mdataset;
				delete [] basename;
				delete file_list_test;
				return 1;
			}

			// Test
			double mean_nll = 0.0;

			long n_frames = mdataset->getNoExamples();

			for (long i=0 ; i<n_frames ; i++)
			{
				 DoubleTensor *x = (DoubleTensor *) mdataset->getExample(i);

				gmm->forward(x);

				DoubleTensor *o = (DoubleTensor *) &gmm->getOutput();

				mean_nll += o->get(0);
			}

			mean_nll /= (double) n_frames;

			ofile.printf("%s %g\n", basename, mean_nll);

			//
			delete mdataset;
			delete [] basename;
		}

		ofile.close();

		delete file_list_test;

		//
		delete gmm;
	}

	//==================================================================== 
	//====================== Merging Mode  ===============================
	//==================================================================== 
	if(mode == 4)
	{
	   	if(verbose) print("Merging mode\n");

		int n_model_files;
		int max_files;
		char *tmp;
		char *str;

		if(verbose) print("merging files: %s\n", model_string);

		str = new char [strlen(model_string)+1];
		strcpy(str, model_string);

		// counting the number of files
		tmp = strtok(model_string, " ");
		for(n_model_files = 1 ; (tmp = strtok(NULL, " ")) ; n_model_files++);
		if(verbose) print("Number of files: %d\n", n_model_files);

		// allocating memory for filenames
		max_files = n_model_files + 1;
		char **filename_in = new char* [max_files];
		MultiVariateDiagonalGaussianDistribution **gmm = new MultiVariateDiagonalGaussianDistribution* [max_files];
		for(int i = 0 ; i < max_files ; i++)
		{
			filename_in[i] = NULL;
			gmm[i] = NULL;
		}

		// extracting filenames
		filename_in[0] = strtok(str, " ");
		for(n_model_files = 1 ; (filename_in[n_model_files] = strtok(NULL, " ")) ; n_model_files++);
		
		if(verbose) print("Loading GMMs ...\n");

		int n_total_gaussians = 0;
		int n_inputs;

		for(int i = 0 ; i < n_model_files ; i++)
		{
			if(verbose) print(" + GMM %s:\n", filename_in[i]);

			gmm[i] = new MultiVariateDiagonalGaussianDistribution();

			File ifile;
			ifile.open(filename_in[i], "r");
			gmm[i]->loadFile(ifile);
			ifile.close();

			int n_inputs_ = gmm[i]->getNinputs();
			int n_gaussians_ = gmm[i]->getNmeans();

			if(i == 0) n_inputs = n_inputs_;
			else if(n_inputs_ != n_inputs) error("Incorrect number of inputs (read=%d expected=%d)", n_inputs_, n_inputs);

			print("   %d gaussians\n", n_gaussians_);
			print("   %d inputs\n", n_inputs_);

			n_total_gaussians += n_gaussians_;

		}
		delete [] filename_in;
		delete [] str;
	
		if(verbose) print("Merging %d GMMs into 1 GMM of %d gaussians...\n", n_model_files, n_total_gaussians);

		print("Building the merged GMM %d x %d ...\n", n_inputs, n_total_gaussians);
		MultiVariateDiagonalGaussianDistribution *merge_gmm = new MultiVariateDiagonalGaussianDistribution(n_inputs, n_total_gaussians);

		// copy params
		double *dst_means_ = merge_gmm->m_parameters->getDarray("means");
		double *dst_variances_ = merge_gmm->m_parameters->getDarray("variances");
		double *dst_weights_ = merge_gmm->m_parameters->getDarray("weigths");
		double sum_weights = 0.0;
		for(int i = 0 ; i < n_model_files ; i++)
		{
			int n_gaussians_ = gmm[i]->getNmeans();

			double *src_means_ = gmm[i]->m_parameters->getDarray("means");
			double *src_variances_ = gmm[i]->m_parameters->getDarray("variances");
			double *src_weights_ = gmm[i]->m_parameters->getDarray("weigths");

			for(int j = 0 ; j < n_gaussians_ ; j++)
			{
				dst_weights_[j] = src_weights_[j];
				sum_weights += src_weights_[j];

				for(int k = 0 ; k < n_inputs ; k++)
				{
					dst_means_[k] = src_means_[k];
					dst_variances_[k] = src_variances_[k];
				}

				dst_means_ += n_inputs;
				dst_variances_ += n_inputs;
			}
			
			dst_weights_ += n_gaussians_;
		}

		// renorm weights
		dst_weights_ = merge_gmm->m_parameters->getDarray("weigths");
		for(int i = 0 ; i < n_model_files ; i++)
		{
			int n_gaussians_ = gmm[i]->getNmeans();

			for(int j = 0 ; j < n_gaussians_ ; j++)
				dst_weights_[j] /= sum_weights;
			
			dst_weights_ += n_gaussians_;
		}

		merge_gmm->prepare();

		if(verbose) merge_gmm->print();

		File ofile;
		ofile.open(output_model_filename, "w");
		merge_gmm->saveFile(ofile);
		ofile.close();

		for(int i = 0 ; i < max_files ; i++) delete gmm[i];
		delete [] gmm;
		delete merge_gmm;
	}

	//==================================================================== 
	//========================= Read Mode  ===============================
	//==================================================================== 
	if(mode == 5)
	{
	   	if(verbose) print("Read mode\n");

		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution();

		File ifile;
		ifile.open(input_model_filename, "r");
		gmm->loadFile(ifile);
		ifile.close();
		
		if(verbose) gmm->print();
		else
		{
			int n_inputs_ = gmm->getNinputs();
			int n_gaussians_ = gmm->getNmeans();

			print("GMM with %d gaussians and %d inputs\n", n_gaussians_, n_inputs_);
		}

		delete gmm;
	}

	//==================================================================== 
	//========================= Copy Mode  ===============================
	//==================================================================== 
	if(mode == 6)
	{
	   	if(verbose) print("Copy mode\n");

		MultiVariateDiagonalGaussianDistribution *gmm = new MultiVariateDiagonalGaussianDistribution();

		File ifile;
		ifile.open(input_model_filename, "r");
		gmm->loadFile(ifile);
		ifile.close();
		
		if(verbose)
		{
			int n_inputs_ = gmm->getNinputs();
			int n_gaussians_ = gmm->getNmeans();

			print("Copying a GMM with %d gaussians and %d inputs ...\n", n_gaussians_, n_inputs_);
		}

		File ofile;
		ofile.open(output_model_filename, "w");
		gmm->saveFile(ofile);
		ofile.close();

		delete gmm;
	}

	return(0);
}

char *strFilename(char *filename, char dirsep) 
{
	char *p = strrchr(filename, dirsep);
	return p ? (p+1) : filename;
}

char *strBasename(char *filename, char extsep)
{
	char *copy = NULL;
	int len = strlen(filename);
	char *p = filename + len - 1;
	int i=len-1;
	while (*p != extsep && i-- >0) p--;
	if (i>0) 
	{
		copy = new char [i+1];
		strncpy(copy,filename,i);
		copy[i] = '\0';
	} 
	else 
	{
		copy = new char [len+1];
		strcpy(copy,filename);
	}
	return copy;
}

#include <sys/stat.h>

int checkFiles(int n_files, char **file_names, bool verbose)
{
	int reminding_files = n_files;
	
	struct stat st;

	if(verbose) print("Checking %d files ...\n", n_files);

	int i = 0;
	while(i < reminding_files)
	{
		if(verbose) print("Checking %s\n", file_names[i]);

		if(stat(file_names[i], &st) == -1)
		{
			warning("Couldn't stat file %s.", file_names[i]);

			for(int j = i ; j < reminding_files-1 ; j++) file_names[j] = file_names[j+1];
			file_names[reminding_files-1] = NULL;
			reminding_files--;
		}
		else
		{
			if(!S_ISREG (st.st_mode))
			{
				warning("not regular file %s.", file_names[i]);

				for(int j = i ; j < reminding_files-1 ; j++) file_names[j] = file_names[j+1];
				file_names[reminding_files-1] = NULL;
				reminding_files--;
			}
			else i++;
		}
	}
	
	if(verbose)
	{
		print("Checked files (%d):\n", reminding_files);
		for(int i = 0 ; i < reminding_files ; i++)
			print("-> %s\n", file_names[i]);
	}

	return reminding_files;
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

	//print("Number of samples = %d\n", n_examples);
	
	mean_nll /= (double) n_examples;

	print("Mean nll = %g\n", mean_nll);

	return true;
}

bool loadDataSet(MemoryDataSet *mdataset, int *n_inputs, FileList *file_list, bool verbose)
{
	int n_examples = 0;
	int input_size = 0;

	// check data files
	int n_reminding_files = file_list->n_files;
	n_reminding_files = checkFiles(file_list->n_files, file_list->file_names, verbose);

	if(verbose) print("Scanning %d files ...\n", n_reminding_files);
	for(int i = 0 ; i < n_reminding_files ; i++)
	{
		if(verbose) print("Tensor file %s:\n", file_list->file_names[i]);

		TensorFile tf;

		if(tf.openRead(file_list->file_names[i]) == false) return false;

		const TensorFile::Header& header = tf.getHeader();

		if(verbose)
		{
			print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
			print(" n_tensors:    [%d]\n", header.m_n_samples);
			print(" n_dimensions: [%d]\n", header.m_n_dimensions);
			print(" size[0]:      [%d]\n", header.m_size[0]);
			print(" size[1]:      [%d]\n", header.m_size[1]);
			print(" size[2]:      [%d]\n", header.m_size[2]);
			print(" size[3]:      [%d]\n", header.m_size[3]);
		}

		tf.close();

		if(header.m_type != Tensor::Float)
		{
			warning("Unsupported tensor type (Float only).");

			return 1;
		}

		if(header.m_n_dimensions == 1 || header.m_n_dimensions == 2 || header.m_n_dimensions == 3)
		{
			if(input_size == 0) input_size = header.m_size[0];
			else if(header.m_size[0] != input_size)
			{
				warning("Inconsistant input size (%d).", input_size);

				return 1;
			}
		}
		else
		{
			warning("Unsupported dimensions (1, 2 or 3 only).");

			return 1;
		}

		n_examples += header.m_n_samples;
	}

	mdataset->reset(n_examples, Tensor::Double);

	if(verbose) print("Loading ...\n");
	long p = 0;
	for(int c = 0 ; c < n_reminding_files ; c++)
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

	if(verbose) print("Number of examples of size %d loaded: %d\n", input_size, n_examples);

	if(input_size < 1) return false;
	if(n_examples < 1) return false;

	return true;
}

bool loadDataSet(MemoryDataSet *mdataset, int *n_inputs, char *filename, bool verbose)
{
	int n_examples = 0;
	int input_size = 0;

	if(verbose) print("Scanning tensor file %s ...\n", filename);

	TensorFile tf;

	if(tf.openRead(filename) == false) return false;

	const TensorFile::Header& header = tf.getHeader();

	if(verbose)
	{
		print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
		print(" n_tensors:    [%d]\n", header.m_n_samples);
		print(" n_dimensions: [%d]\n", header.m_n_dimensions);
		print(" size[0]:      [%d]\n", header.m_size[0]);
		print(" size[1]:      [%d]\n", header.m_size[1]);
		print(" size[2]:      [%d]\n", header.m_size[2]);
		print(" size[3]:      [%d]\n", header.m_size[3]);
	}

	tf.close();

	if(header.m_type != Tensor::Float)
	{
		warning("Unsupported tensor type (Float only).");

		return 1;
	}

	if(header.m_n_dimensions == 1 || header.m_n_dimensions == 2 || header.m_n_dimensions == 3)
	{
		if(input_size == 0) input_size = header.m_size[0];
		else if(header.m_size[0] != input_size)
		{
			warning("Inconsistant input size (%d).", input_size);

			return 1;
		}
	}
	else
	{
		warning("Unsupported dimensions (1, 2 or 3 only).");

		return 1;
	}

	input_size = header.m_size[0];
	n_examples += header.m_n_samples;

	//
	mdataset->reset(n_examples, Tensor::Double);


	//
	if(verbose) print("Loading ...\n");

	long p = 0;

	if(tf.openRead(filename) == false) return false;

	for(int j = 0 ; j < header.m_n_samples ; j++)
	{
		FloatTensor *tensor = NULL;
		tensor = (FloatTensor *)tf.load();

		Tensor* example = mdataset->getExample(p);
		example->resize(input_size);

		example->copy(tensor);

		delete tensor;

		p++;
	}

	tf.close();

	*n_inputs = input_size;

	if(verbose) print("Number of examples of size %d loaded: %d\n", input_size, n_examples);

	if(input_size < 1) return false;
	if(n_examples < 1) return false;

	return true;
}
