#include "torch5spro.h"

using namespace Torch;

const char *help = "\
progname: cmdfileTest.cc\n\
code2html: This program tests the class CmdFile.\n\
version: Torch5spro\n\
date: June 2009\n\
author: Sebastien Marcel (marcel@idiap.ch) 2009\n";

int main(int argc, char **argv)
{
  	char *model_name = NULL;
  	char *train_param = NULL;
  	char *data_param = NULL;
  	char *model_param = NULL;
  
  	CmdLine cmd;

  	cmd.info(help);

  	cmd.addText("\nArguments:");  
  	cmd.addSCmdArg("model name", &model_name, "the name of the model");
  	cmd.addText("\nOptions:");  
  	cmd.addSCmdOption("-tp", &train_param, "params/train.param", "the training parameter file");
  	cmd.addSCmdOption("-dp", &data_param, "params/data.param", "the data parameter file");
  	cmd.addSCmdOption("-mp", &model_param, "params/model.param", "the model parameter file");

  	// Read the command line
  	cmd.read(argc, argv);

	print("tp=%s\n", train_param);
	print("dp=%s\n", data_param);
	print("mp=%s\n", model_param);
	
  	int EpochPlotLimit, EpochPlotStep;  
  	int the_seed;  
  	double accuracy, criterion_min;
  	double learning_rate, learning_rate_decay, weight_mse, weight_mvse, weight_decay;
  	int max_iter;
  	int epoch_early_stop_train;
  	int epoch_early_stop_valid;
  	bool regression, cgm, gm;

  	message("# MODEL : %s ##", model_name);  

  	CmdFile cmdf_train;
  	cmdf_train.info("Training Parameters");

  	cmdf_train.addText("\nStandard training parameters:");  
  	cmdf_train.addDCmd("LearningRate", &learning_rate, "Learning rate");
  	cmdf_train.addDCmd("WeightMse", &weight_mse, "Weight for Mse Criterion");
  	cmdf_train.addDCmd("WeightMvse", &weight_mvse, "Weight for Mvse Criterion");
  	cmdf_train.addICmd("EpochMax", &max_iter, "Maximum number of epoch");   

  	cmdf_train.addBCmdOption("Regression", &regression, false, "Use regression mode");
  	cmdf_train.addBCmdOption("Generative", &gm, false, "Use generative mode");
  	cmdf_train.addBCmdOption("ConstrainedGenerative", &cgm, false, "Use constrained generative mode");

  	cmdf_train.addText("\nOptional parameters to control the training:");
  	cmdf_train.addDCmdOption("LearningRateDecay", &learning_rate_decay, 0.0, "Learning rate decay");
  	cmdf_train.addDCmdOption("WeightDecay", &weight_decay, 0.0, "Weight decay");

  	cmdf_train.addText("\nOptional parameters to stop the training:");
  	cmdf_train.addICmdOption("EarlyStoppingTrain", &epoch_early_stop_train, 0, "Maximum epoch number after the last minimum on the training set");
  	cmdf_train.addICmdOption("EarlyStoppingValid", &epoch_early_stop_valid, 0, "Maximum epoch number after the last minimum on the validation set");
  	cmdf_train.addDCmdOption("EndAccuracy", &accuracy, -1.0,"End accuracy");
  	cmdf_train.addDCmdOption("MinimumCriterion", &criterion_min, 0.0, "Minimum criterion");  

  	cmdf_train.addText("\nOthers optional training parameters:");
  	cmdf_train.addICmdOption("EpochPlotLimit", &EpochPlotLimit, 10, "Epoch plot limit");   
  	cmdf_train.addICmdOption("EpochPlotStep", &EpochPlotStep, 10, "Epoch plot step");   
  	cmdf_train.addICmdOption("ManualSeed", &the_seed, -1, "The random seed");   

  	cmdf_train.read(train_param);
	
  	cmdf_train.help();
  	
	print("LearningRate %g\n", learning_rate);
  	print("WeightMse %g\n", weight_mse);
  	print("WeightMvse %g\n", weight_mvse);
  	print("EpochMax %d\n", max_iter);   
  	print("Regression %d\n", regression);
  	print("Generative %d\n", gm);
  	print("ConstrainedGenerative %d\n", cgm);
  	print("LearningRateDecay %g\n", learning_rate_decay);
  	print("WeightDecay %g\n", weight_decay);
  	print("EarlyStoppingTrain %d\n", epoch_early_stop_train);
  	print("EarlyStoppingValid %d\n", epoch_early_stop_valid);
  	print("EndAccuracy %g\n", accuracy);
  	print("MinimumCriterion %g\n", criterion_min);  
  	print("EpochPlotLimit %d\n", EpochPlotLimit);   
  	print("EpochPlotStep %d\n", EpochPlotStep);   
  	print("ManualSeed %d\n", the_seed);   
  	print("\n\n");   




	
  	bool StdOrBindata, TargetInside, StdBin;  
  	int NbTrainingSets, NbValidationSets, NbTestSets;
  	char **TrainingSet;
  	char **ValidationSet;
  	char **TestSet;
  	char **RootName;
  	char *Extension;
	double *prior;
  
  	CmdFile cmdf_data;
  	cmdf_data.info("Database");

  	cmdf_data.addText("\nType of datasets :");  
  	cmdf_data.addBCmd("StdOrBindata", &StdOrBindata, "Type of datasets");
  	cmdf_data.addBCmd("TargetInside", &TargetInside, "Targets into datasets. If not you are only able to deal with 1 or 2 class problem");
  	cmdf_data.addBCmdOption("StdBin", &StdBin, false, "Std dataset bin or not ?");

  	cmdf_data.addText("\nNumber of datasets :");  
  	cmdf_data.addICmd("NbTrainingSets", &NbTrainingSets, "Number of training sets");
  	cmdf_data.addICmdOption("NbValidationSets", &NbValidationSets, 0, "Number of validation sets");
  	cmdf_data.addICmdOption("NbTestSets", &NbTestSets, 0, "Number of test sets");
  	cmdf_data.addSCmdOption("Extension", &Extension, "", "Extension");

  	cmdf_data.read(data_param, false);

  	cmdf_data.addText("\nDatasets :");  
  	cmdf_data.addHrefSCmdOption("RootName[%d]", &RootName, "", "Root name", NbTrainingSets);
  	cmdf_data.addHrefSCmd("TrainingSet[%d]", &TrainingSet, "Training set", NbTrainingSets);
  	cmdf_data.addHrefDCmdOption("Prior[%d]", &prior, -1.0, "Priors", NbTrainingSets);
  	if(NbValidationSets != 0)
	{		
		cmdf_data.addHrefSCmdOption("ValidationSet[%d]", &ValidationSet, "", "Validation set", NbValidationSets);
		
		if(NbTestSets != 0) cmdf_data.addHrefSCmdOption("TestSet[%d]", &TestSet, "", "Test set", NbTestSets);
	}
  
  	cmdf_data.readHref();
	
	cmdf_data.help();
  	
	print("StdOrBindata %d\n", StdOrBindata);
  	print("TargetInside %d\n", TargetInside);
  	print("StdBin %d\n", StdBin);
  	print("NbTrainingSets %d\n", NbTrainingSets);
  	print("NbValidationSets %d\n", NbValidationSets);
  	print("NbTestSets %d\n", NbTestSets);
  	print("Extension \"%s\"\n", Extension);
	for(int i = 0 ; i < NbTrainingSets ; i++)
	{
  		print("RootName [%d] \"%s\"\n", i, RootName[i]);
	
  		print("TrainingSet[%d] \"%s\"\n", i, TrainingSet[i]);

  		print("Prior[%d] %g\n", i, prior[i]);
	}
	
	for(int i = 0 ; i < NbValidationSets ; i++)
		print("ValidationSet[%d] \"%s\"\n", i, ValidationSet[i]);
		
	for(int i = 0 ; i < NbTestSets ; i++)
		print("TestSet[%d] \"%s\"\n", i, TestSet[i]);

  	print("\n\n");   



	
  	int n_inputs;
  	int n_targets;
  	int n_hidden_layer;
  	int *n_hidden;
  	bool bSigmoid, InputsToOutputs;

  	CmdFile cmdf_model;
  	cmdf_model.info("Model parameters");

  	cmdf_model.addICmd("NbInputs", &n_inputs, "Input dimension of the model");
  	cmdf_model.addICmd("NbOutputs", &n_targets, "Output dimension of the model");  
  	cmdf_model.addICmd("NbHiddenLayers", &n_hidden_layer, "Hidden layer in the model");
  	cmdf_model.addBCmdOption("SigmoidOrTanh", &bSigmoid, false, "Using Sigmoid units or not (Tanh if not)");
  	cmdf_model.addBCmdOption("InputsToOutputs", &InputsToOutputs, false, "Connect Inputs to Outputs (help sometimes)");
  	cmdf_model.read(model_param, false);  

  	cmdf_model.addHrefICmd("NbHidden[%d]", &n_hidden, "Hidden units", n_hidden_layer);
  	cmdf_model.readHref();
	
	cmdf_model.help();
  
  	print("NbInputs %d\n", n_inputs);
  	print("NbOutputs %d\n", n_targets);  
  	print("NbHiddenLayers %d\n", n_hidden_layer);
  	print("SigmoidOrTanh %d\n", bSigmoid);
  	print("InputsToOutputs %d\n", InputsToOutputs);

	for(int i = 0 ; i < n_hidden_layer ; i++)
  		print("NbHidden[%d] %d\n", i, n_hidden[i]);
  	
	print("\n\n");   

	File f;

	f.open("params/connect.mat", "r");

	int n_rows, n_cols;

	f.scanf("%d", &n_rows);
	f.scanf("%d", &n_cols);

	print("n rows = %d\n", n_rows);
	print("n cols = %d\n", n_cols);

	char str[50];
	for(int row = 0 ; row < n_rows ; row++)
	{
		for(int col = 0 ; col < n_cols ; col++)
		{
		   	f.scanf("%s", &str);
			
			print("%s ", str);
		}
		print("\n");   
	}
	
	print("\n\n");   

  	print("That's all folks !\n");   

	
    	return 1;
}
