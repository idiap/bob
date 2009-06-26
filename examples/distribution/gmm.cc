#include "torch5spro.h"

using namespace Torch;

const char *help = "\
progname: gmm.cc\n\
code2html: This program perform the training and the test of the Gaussian Mixture Model (GMM).\n\
version: Torch5spro\n\
date: June 2009\n\
author: Sebastien Marcel (marcel@idiap.ch) 2005-2006\n";

using namespace Torch;

// check data files
int checkFiles(int n_files, char **file_names, bool verbose);

//bool loadDataSet(MemoryDataSet *, int *, FileList *);
//bool testDataSet(MemoryDataSet *, ProbabilityDistribution *);

int main(int argc, char **argv)
{
	/*
	real accuracy;
	real threshold;
	int max_iter_kmeans;
	int max_iter_gmm;
	int n_inputs;
	real prior;
	bool no_learn_means;
	bool learn_var;
	bool learn_weights;
	bool viterbi;
	real map_factor;

	int the_seed;
	int max_load;
	bool norm;
	int k_fold;
	bool binary_mode;
	bool disk;
	char *dir_name;
	char *model_file;
	char *save_model_file;
	char *output_file;
	int n_sequences;
	int oscore;
	*/
	char *list_tensor_filename;
	char *input_model_filename;
	char *output_model_filename;
	char *score_filename;
	char *model_string;

	int n_gaussians;

	bool verbose;

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
	//cmd.addRCmdOption("-threshold", &threshold, 0.001, "variance threshold");
	//cmd.addRCmdOption("-prior", &prior, 0.001, "prior on the weights");
	//cmd.addICmdOption("-iterk", &max_iter_kmeans, 25, "max number of iterations of KMeans");
	//cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	//cmd.addRCmdOption("-e", &accuracy, 0.00001, "end accuracy");
	//cmd.addICmdOption("-kfold", &k_fold, -1, "number of folds, if you want to do cross-validation");
	//cmd.addBCmdOption("-viterbi", &viterbi, false, "viterbi instead of EM");

	cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-seed", &the_seed, -1, "the random seed");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-norm", &norm, false, "normalize the datas");
	//cmd.addBCmdOption("-disk", &disk, false, "use a disk dataset");
	//cmd.addICmdOption("-n_sequences", &n_sequences, 1, "number of examples in a bindata file");
	cmd.addSCmdOption("-save", &output_model_filename, "model-train.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Retrain mode
	cmd.addMasterSwitch("--retrain");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");

	//cmd.addRCmdOption("-threshold", &threshold, 0.001, "variance threshold");
	//cmd.addRCmdOption("-prior", &prior, 0.001, "prior on the weights");
	//cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	//cmd.addRCmdOption("-e", &accuracy, 0.00001, "end accuracy");
	//cmd.addBCmdOption("-viterbi", &viterbi, false, "viterbi instead of EM");

	cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-disk", &disk, false, "use a disk dataset");
	//cmd.addICmdOption("-n_sequences", &n_sequences, 1, "number of examples in a bindata file");
	cmd.addSCmdOption("-save", &output_model_filename, "model-retrain.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Adaptation mode
	cmd.addMasterSwitch("--adapt");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");

	//cmd.addRCmdOption("-threshold", &threshold, 0.001, "variance threshold");
	//cmd.addRCmdOption("-prior", &prior, 0.001, "prior on the weights");
	//cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	//cmd.addRCmdOption("-e", &accuracy, 0.00001, "end accuracy");
	//cmd.addRCmdOption("-map", &map_factor, 0.5, "adaptation factor [0-1]");
	//cmd.addBCmdOption("-viterbi", &viterbi, false, "viterbi instead of EM");

	cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-no_means", &no_learn_means, false, "no enroll means");
	//cmd.addBCmdOption("-learn_var", &learn_var, false, "enroll var");
	//cmd.addBCmdOption("-learn_weights", &learn_weights, false, "enroll weights");
	//cmd.addBCmdOption("-disk", &disk, false, "use a disk dataset");
	//cmd.addICmdOption("-n_sequences", &n_sequences, 1, "number of examples in a bindata file");
	cmd.addSCmdOption("-save", &output_model_filename, "model-adapt.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Test mode
	cmd.addMasterSwitch("--test");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");
	cmd.addSCmdArg("list of tensor files", &list_tensor_filename, "list of tensor files to load");
	cmd.addSCmdArg("scores output file", &score_filename, "output score filename");

	cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-viterbi", &viterbi, false, "viterbi instead of EM");
	//cmd.addBCmdOption("-disk", &disk, false, "use a disk dataset");
	//cmd.addICmdOption("-n_sequences", &n_sequences, 1, "number of examples in a bindata file");
	//cmd.addICmdOption("-oscore", &oscore, 0, "output strategy for scores from the same file (0=all, 1=mean, 2=max)");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Merge mode
	cmd.addMasterSwitch("--merge");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("input models", &model_string, "input model file in double-quotes: \"model1 model2 ...\"");
	
	cmd.addText("\nMisc Options:");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	cmd.addSCmdOption("-save", &output_model_filename, "model-merge.gmm", "output model filename");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Read mode
	bool display_meanvar;
	cmd.addMasterSwitch("--read");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &input_model_filename, "input model filename");

	cmd.addText("\nMisc Options:");
	//cmd.addBCmdOption("-meanvar", &display_meanvar, false, "display mean and variance");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	cmd.addSCmdOption("-save", &output_model_filename, "model-copy.gmm", "output model filename");
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
	   	if(verbose) print("Training mode\n");

		/*
		// check data files
		int n_reminding_files = file_list.n_files;
		n_reminding_files = checkFiles(file_list.n_files, file_list.file_names);

		// Create the DataSet
		DataSet *data;
		if(n_sequences > 1) data = new(allocator) OneFileMultiSeqDataSet(file_list.file_names, n_reminding_files, n_sequences);
		else
		{
			if(disk) data = new (allocator) DiskMatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);	   
			else data = new (allocator) MatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);	   
		}

		n_inputs = data->n_inputs;

		if(the_seed == -1) Random::seed();
		else Random::manualSeed((long)the_seed);

		MeanVarNorm* mv_norm = NULL;
		if(norm)
		{
			mv_norm = new(allocator)MeanVarNorm (data);
			data->preProcess(mv_norm);
		}

		//=================== Create the GMM... =========================

		// create a KMeans object to initialize the GMM
		KMeans kmeans(n_inputs, n_gaussians);
		kmeans.setROption("prior weights",prior);

		// the kmeans trainer
		EMTrainer kmeans_trainer(&kmeans);
		kmeans_trainer.setROption("end accuracy", accuracy);
		kmeans_trainer.setIOption("max iter", max_iter_kmeans);

		// the kmeans measurer
		MeasurerList kmeans_measurers;
		NLLMeasurer nll_kmeans_measurer(kmeans.log_probabilities,data,cmd.getXFile("kmeans_train_val"));
		kmeans_measurers.addNode(&nll_kmeans_measurer);

		// create the GMM
		DiagonalGMM gmm(n_inputs,n_gaussians,&kmeans_trainer);
		
		// set the training options
		real* thresh = (real*)allocator->alloc(n_inputs*sizeof(real));
		initializeThreshold(data,thresh,threshold);	
		gmm.setVarThreshold(thresh);
		gmm.setROption("prior weights",prior);
		gmm.setOOption("initial kmeans trainer measurers", &kmeans_measurers);


		//=================== Measurers and Trainer  ===============================

		// Measurers on the training dataset
		MeasurerList measurers;
		NLLMeasurer nll_meas(gmm.log_probabilities, data, cmd.getXFile("gmm_train_val"));
		measurers.addNode(&nll_meas);

		// The Gradient Machine Trainer
		EMTrainer trainer(&gmm);
		trainer.setIOption("max iter", max_iter_gmm);
		trainer.setROption("end accuracy", accuracy);
		trainer.setBOption("viterbi", viterbi);

		//=================== Let's go... ===============================

		if(k_fold <= 0)
		{
			trainer.train(data, &measurers);

			if(strcmp(save_model_file, ""))
			{
				DiskXFile model_(save_model_file, "w");
				if(norm)
					unNormalizeParameters(&gmm,mv_norm->inputs_mean, mv_norm->inputs_stdv);
				model_.taggedWrite(&n_gaussians, sizeof(int), 1, "n_gaussians");
				model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
				gmm.saveXFile(&model_);
			}
		}
		else
		{
			KFold k(&trainer, k_fold);
			k.crossValidate(data, NULL, &measurers);
		}
		*/
	}

	//==================================================================== 
	//=================== Retraining Mode  ===============================
	//==================================================================== 
	if(mode == 1)
	{
	   	if(verbose) print("Retraining mode\n");

		/*
		// check data files
		int n_reminding_files = file_list.n_files;
		n_reminding_files = checkFiles(file_list.n_files, file_list.file_names);

		// Create the DataSet
		DataSet *data;
		if(n_sequences > 1) data = new(allocator) OneFileMultiSeqDataSet(file_list.file_names, n_reminding_files, n_sequences);
		else
		{
			if(disk) data = new (allocator) DiskMatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);	   
			else data = new (allocator) MatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);	   
		}

		DiskXFile model(model_file, "r");
		model.taggedRead(&n_gaussians, sizeof(int), 1, "n_gaussians");
		model.taggedRead(&n_inputs, sizeof(int), 1, "n_inputs");

		if(n_inputs != data->n_inputs)
			error("gmm: the input number of the GMM (%d) do not correspond to the input number of the dataset (%d)", n_inputs, data->n_inputs);

		DiagonalGMM gmm(n_inputs,n_gaussians);

		// set the training options
		real* thresh = (real*)allocator->alloc(n_inputs*sizeof(real));
		initializeThreshold(data,thresh,threshold);	
		gmm.setVarThreshold(thresh);
		gmm.setROption("prior weights",prior);

		gmm.loadXFile(&model);


		//=================== Measurers and Trainer ===================

		// Measurers on the training dataset
		MeasurerList measurers;
		NLLMeasurer nll_meas(gmm.log_probabilities, data, cmd.getXFile("gmm_retrain_val"));
		measurers.addNode(&nll_meas);


		//=================== The Trainer ===============================

		// The Gradient Machine Trainer
		EMTrainer trainer(&gmm);
		trainer.setIOption("max iter", max_iter_gmm);
		trainer.setROption("end accuracy", accuracy);
		trainer.setBOption("viterbi", viterbi);

		//=================== Let's go... ===============================

		trainer.train(data, &measurers);

		if(strcmp(save_model_file, ""))
		{
			DiskXFile model_(save_model_file, "w");
			model_.taggedWrite(&n_gaussians, sizeof(int), 1, "n_gaussians");
			model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
			gmm.saveXFile(&model_);
		}
		*/
	}

	//==================================================================== 
	//=================== Adaptation Mode  ===============================
	//==================================================================== 
	if(mode == 2)
	{
	   	if(verbose) print("Adaptation mode\n");

		/*
		// check data files
		int n_reminding_files = file_list.n_files;
		n_reminding_files = checkFiles(file_list.n_files, file_list.file_names);

		// Create the DataSet
		DataSet *data;
		if(n_sequences > 1) data = new(allocator) OneFileMultiSeqDataSet(file_list.file_names, n_reminding_files, n_sequences);
		else
		{
			if(disk) data = new (allocator) DiskMatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);
			else data = new (allocator) MatDataSet(file_list.file_names, n_reminding_files, -1, 0, true, max_load, binary_mode);	   
		}

		DiskXFile model(model_file, "r");
		model.taggedRead(&n_gaussians, sizeof(int), 1, "n_gaussians");
		model.taggedRead(&n_inputs, sizeof(int), 1, "n_inputs");

		if(n_inputs != data->n_inputs)
			error("gmm: the input number of the GMM (%d) do not correspond to the input number of the dataset (%d)", n_inputs, data->n_inputs);

		DiagonalGMM prior_gmm(n_inputs,n_gaussians);
		prior_gmm.loadXFile(&model);
		MAPDiagonalGMM gmm(&prior_gmm);

		// set the training options
		real* thresh = (real*)allocator->alloc(n_inputs*sizeof(real));
		initializeThreshold(data,thresh,threshold);	
		gmm.setVarThreshold(thresh);
		
		gmm.setROption("prior weights",prior);
		gmm.setROption("weight on prior",map_factor);
		gmm.setBOption("learn means",!no_learn_means);
		gmm.setBOption("learn variances",learn_var);
		gmm.setBOption("learn weights",learn_weights);



		//=================== Measurers and Trainer ===================

		// Measurers on the training dataset
		MeasurerList measurers;
		NLLMeasurer nll_meas(gmm.log_probabilities, data, cmd.getXFile("gmm_adaptation_val"));
		measurers.addNode(&nll_meas);


		//=================== The Trainer ===============================

		// The Gradient Machine Trainer
		EMTrainer trainer(&gmm);
		trainer.setIOption("max iter", max_iter_gmm);
		trainer.setROption("end accuracy", accuracy);
		trainer.setBOption("viterbi", viterbi);

		//=================== Let's go... ===============================

		trainer.train(data, &measurers);

		if(strcmp(save_model_file, ""))
		{
			DiskXFile model_(save_model_file, "w");
			model_.taggedWrite(&n_gaussians, sizeof(int), 1, "n_gaussians");
			model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
			gmm.saveXFile(&model_);
		}
		*/
	}

	//==================================================================== 
	//====================== Testing Mode  ===============================
	//==================================================================== 
	if(mode == 3)
	{
	   	/*
		if(oscore < 0) oscore = 0;
		if(oscore > 2) oscore = 2;

	   	if(verbose)
		{
		   	if(n_sequences > 1)
			{
		   		switch(oscore)
				{
				case 0:
					print("Testing mode (all scores)\n");
					break;
				case 1:
					print("Testing mode (mean score)\n");
					break;
				case 2:
					print("Testing mode (max score)\n");
					break;
				}
			}
			else print("Testing mode\n");
		}

		// check data files
		int n_reminding_files = file_list.n_files;
		n_reminding_files = checkFiles(file_list.n_files, file_list.file_names);

		DiskXFile model(model_file, "r");
		model.taggedRead(&n_gaussians, sizeof(int), 1, "n_gaussians");
		model.taggedRead(&n_inputs, sizeof(int), 1, "n_inputs");

		DiagonalGMM gmm(n_inputs,n_gaussians);
		gmm.loadXFile(&model);

		//=================== DataSets & Measurers... ===================

		DiskXFile sc_out_xfile(output_file,"w");
		DataSet *mdata;

		for(int i = 0 ; i < n_reminding_files ; i++)
		{
		   	if(verbose) print("testing file %s\n", file_list.file_names[i]);
			
			if(n_sequences > 1) mdata = new(allocator) OneFileMultiSeqDataSet(file_list.file_names[i], n_sequences);
			else mdata = new(allocator) MatDataSet(file_list.file_names[i], -1, 0, true, max_load, binary_mode);	   
		
			char *temp = strBaseName(file_list.file_names[i]);
			char *basename = strRemoveSuffix(temp);
			allocator->retain(basename);
		   
			//
			gmm.eMIterInitialize();

			//
			real sum = 0.0;
			int n_sum = 0;
			real max_ = -INF;
		
			for(int t = 0; t < mdata->n_examples; t++)
			{
				mdata->setExample(t);
				gmm.eMForward(mdata->inputs);

				// this is the log prob without normalization by the number of frames
				//print("> %g\n", gmm.log_probability);

				real mean_ = 0;
				real stdv_ = 0;
				for(int f = 0; f < mdata->inputs->n_frames; f++)
				{
					mean_ += gmm.log_probabilities->frames[f][0];
					stdv_ += mean_ * mean_;
				}

				sum += mean_;
				n_sum += mdata->inputs->n_frames;

				if(n_sequences > 1)
				{
					mean_ /= (real) mdata->inputs->n_frames;
					stdv_ /= (real) mdata->inputs->n_frames;
					stdv_ -= mean_*mean_;
					stdv_ = sqrt(stdv_);

					if(oscore == 0)
					{
						if(verbose) print("> %s_%d %g\n", basename, t, mean_);
						sc_out_xfile.printf("%s_%d %g\n", basename, t, mean_);
						//if(verbose) print("> %s %d %g\n", basename, t, mean_);
						//sc_out_xfile.printf("%s %d %g\n", basename, t, mean_);
					}

					if(mean_ > max_) max_ = mean_;
				}
			}
			
			if(n_sequences <= 1)
			{
				sum /= (real) n_sum;

				if(verbose) print("> %s %g\n", basename, sum);
				sc_out_xfile.printf("%s %g\n", basename, sum);
			}
			else
			{
				if(oscore == 1)
				{
					sum /= (real) n_sum;

					if(verbose) print("> %s %g\n", basename, sum);
					sc_out_xfile.printf("%s %g\n", basename, sum);
				}
				else if(oscore == 2)
				{
					if(verbose) print("> %s %g\n", basename, max_);
					sc_out_xfile.printf("%s %g\n", basename, max_);
				}
			}

			allocator->free(mdata);
		}
		
		if(strcmp(save_model_file, ""))
		{
			DiskXFile model_(save_model_file, "w");
			model_.taggedWrite(&n_gaussians, sizeof(int), 1, "n_gaussians");
			model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
			gmm.saveXFile(&model_);
		}
		*/
	}

	//==================================================================== 
	//====================== Merging Mode  ===============================
	//==================================================================== 
	if(mode == 4)
	{
	   	if(verbose) print("Merging mode\n");

		/*
		int n_model_files;
		int max_files;
		char *tmp;
		char *str;

		if(verbose) print("merging files: %s\n", model_file);

		str = (char *)allocator->alloc((strlen(model_file)+1) * sizeof(char));
		strcpy(str, model_file);

		// counting the number of files
		tmp = strtok(model_file, " ");
		for(n_model_files = 1 ; (tmp = strtok(NULL, " ")) ; n_model_files++);
		if(verbose) print("Number of files: %d\n", n_model_files);

		// allocating memory for filenames
		max_files = n_model_files + 1;
		char **filename_in = (char **)allocator->alloc(max_files*sizeof(char *));
		DiagonalGMM **gmm = (DiagonalGMM **)allocator->alloc(max_files*sizeof(DiagonalGMM *));
		for(int i = 0 ; i < max_files ; i++)
		{
			filename_in[i] = NULL;
			gmm[i] = NULL;
		}

		// extracting filenames
		filename_in[0] = strtok(str, " ");
		for(n_model_files = 1 ; (filename_in[n_model_files] = strtok(NULL, " ")) ; n_model_files++);

		if(verbose) print("Merging GMMs ...\n");

		int n_gaussians_merge = 0;
		int n_inputs_;

		for(int i = 0 ; i < n_model_files ; i++)
		{
			if(verbose) print(" + %s:\n", filename_in[i]);

			//
			DiskXFile *xfile = new(allocator) DiskXFile(filename_in[i], "r");
		
			//
			xfile->taggedRead(&n_gaussians, sizeof(int), 1, "n_gaussians");
			xfile->taggedRead(&n_inputs_, sizeof(int), 1, "n_inputs");

			if(verbose)
			{
				print("   n_gaussians = %d\n", n_gaussians);
				print("   n_inputs = %d\n", n_inputs_);
			}

			if(i == 0) n_inputs = n_inputs_;
			else if(n_inputs_ != n_inputs) error("Incorrect number of inputs (read=%d expected=%d)", n_inputs_, n_inputs);

			n_gaussians_merge += n_gaussians;

			//
			gmm[i] = new(allocator) DiagonalGMM(n_inputs,n_gaussians);
			gmm[i]->loadXFile(xfile);

			//
			allocator->free(xfile);
		}

		DiagonalGMM gmm_merge(n_inputs, n_gaussians_merge);
		
		int g = 0;
		real log_n = log(n_model_files);
		
		for(int n = 0 ; n < n_model_files ; n++)
		{
			for(int i = 0 ; i < gmm[n]->n_gaussians ; i++)
			{
				for(int j = 0 ; j < n_inputs ; j++)
				{
					gmm_merge.means[g][j] = gmm[n]->means[i][j];
					gmm_merge.var[g][j] = gmm[n]->var[i][j];
				}
		
				gmm_merge.log_weights[g++] = gmm[n]->log_weights[i] - log_n;
			}
		}
		
		DiskXFile model_(save_model_file, "w");
		model_.taggedWrite(&n_gaussians_merge, sizeof(int), 1, "n_gaussians");
		model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
		gmm_merge.saveXFile(&model_);
		*/
	}

	//==================================================================== 
	//========================= Read Mode  ===============================
	//==================================================================== 
	if(mode == 5)
	{
	   	if(verbose) print("Read mode\n");

		/*
		DiskXFile model(model_file, "r");
		model.taggedRead(&n_gaussians, sizeof(int), 1, "n_gaussians");
		model.taggedRead(&n_inputs, sizeof(int), 1, "n_inputs");
		
		DiagonalGMM gmm(n_inputs,n_gaussians);
		gmm.loadXFile(&model);

		print("GMM %s:\n", model_file);
		print(" n_gaussians = %d\n", n_gaussians);
		print(" n_inputs = %d\n", n_inputs);

		if(display_meanvar)
		{
			real sumw = 0.0;

			for(int i = 0 ; i < n_gaussians ; i++)
			{
			   	print(" Gaussian %d:\n", i);
				real z = exp(gmm.log_weights[i]);
				sumw += z;
				print("   weight = %g\n", z);
				print("   mean = [ ");
				for(int j = 0 ; j < n_inputs ; j++) print("%g ", gmm.means[i][j]);
				print("]\n");
				print("   var = [ ");
				for(int j = 0 ; j < n_inputs ; j++) print("%g ", gmm.var[i][j]);
				print("]\n");
			}
			print("sum weights = %g\n", sumw);
		}

		if(strcmp(save_model_file, ""))
		{
			DiskXFile model_(save_model_file, "w");
			model_.taggedWrite(&n_gaussians, sizeof(int), 1, "n_gaussians");
			model_.taggedWrite(&n_inputs, sizeof(int), 1, "n_inputs");
			gmm.saveXFile(&model_);
		}
		*/
	}

	return(0);
}

//==================================================================================================== 
//==================================== Functions ===================================================== 
//==================================================================================================== 

#include <sys/stat.h>

int checkFiles(int n_files, char **file_names, bool verbose)
{
	int reminding_files = n_files;
	
	struct stat st;

	if(verbose) print("Checking files:\n");

	int i = 0;
	//for(int i = 0 ; i < n_files ; i++)
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

