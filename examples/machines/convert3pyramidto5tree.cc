#include "torch5spro.h"

using namespace Torch;

CascadeMachine* loadCascade(const char *dirname, char *filename)
{
	char model_filename[250];
	sprintf(model_filename, "%s/%s", dirname, filename); 

	CascadeMachine* cascade = dynamic_cast<CascadeMachine*>(Torch::loadMachineFromFile(model_filename));
	if (cascade == 0)
	{
		error("ERROR: loading model [%s]!\n", model_filename);
		return NULL;
	}
	print("Cascade [%s]:\n", model_filename);
	for (int i = 0; i < cascade->getNoStages(); i ++)
		print(">>> stage [%d/%d]: no. machines = %d, threshold = %lf\n", i + 1, cascade->getNoStages(), cascade->getNoMachines(i), cascade->getThreshold(i));

	return cascade;
}

bool convert(const char *input_dirname, const char *output_filename)
{
	char *toplevel;
	char *inplane_router;
	char *outplane_router;
	double threshold_toplevel;
	double threshold_inplanerouter;
	double threshold_outplanerouter;
	double mu_inplanerouter = 0.0;
	double mu_outplanerouter = 0.0;
	double stdv_inplanerouter = 1.0;
	double stdv_outplanerouter = 1.0;
	char *inplane_list;
	char *inplane_connect;
	char *outplane_list;
	char *outplane_connect;
      
	bool norm_ = true;

	CmdFile cmdf_pyramidcascade;

	cmdf_pyramidcascade.info("Pyramid-Cascade");

	//
	cmdf_pyramidcascade.addText("\nTop-level model and threshold : ");
	cmdf_pyramidcascade.addSCmd("toplevel_model", &toplevel, "top-level model");
	cmdf_pyramidcascade.addDCmd("toplevel_threshold", &threshold_toplevel, "top-level threshold");
	//
	cmdf_pyramidcascade.addText("\nInplane router model and threshold : ");
	cmdf_pyramidcascade.addSCmd("inplane_router_model", &inplane_router, "inplane router model");
	cmdf_pyramidcascade.addDCmd("inplane_router_threshold", &threshold_inplanerouter, "inplane router threshold");
	if(norm_)
	{
		cmdf_pyramidcascade.addDCmd("inplane_router_mu", &mu_inplanerouter, "inplane router mean");
		cmdf_pyramidcascade.addDCmd("inplane_router_stdv", &stdv_inplanerouter, "inplane router variance");
	}
	//
	cmdf_pyramidcascade.addText("\nOutplane router model and threshold : ");
	cmdf_pyramidcascade.addSCmd("outplane_router_model", &outplane_router, "outplane router model");
	cmdf_pyramidcascade.addDCmd("outplane_router_threshold", &threshold_outplanerouter, "outplane router threshold");
	if(norm_)
	{
		cmdf_pyramidcascade.addDCmd("outplane_router_mu", &mu_outplanerouter, "outplane router mean");
		cmdf_pyramidcascade.addDCmd("outplane_router_stdv", &stdv_outplanerouter, "outplane router variance");
	}
	//
	cmdf_pyramidcascade.addText("\nInplane models and connections : ");
	cmdf_pyramidcascade.addSCmd("inplane_list", &inplane_list, "list of inplane models");
	cmdf_pyramidcascade.addSCmd("inplane_connect", &inplane_connect, "connections between inplane models");
	//
	cmdf_pyramidcascade.addText("\nOutplane models and connections : ");
	cmdf_pyramidcascade.addSCmd("outplane_list", &outplane_list, "list of outplane models");
	cmdf_pyramidcascade.addSCmd("outplane_connect", &outplane_connect, "connections between outplane models");


	char input_filename[250];
	sprintf(input_filename, "%s/pyramid-cascade.model", input_dirname); 

	cmdf_pyramidcascade.read(input_filename, true);

	//
	Torch::print("toplevel machine: %s\n", toplevel);
	Torch::print("toplevel threshold: %g\n", threshold_toplevel);

	//...
	CascadeMachine *toplevel_cascade;

	toplevel_cascade = loadCascade(input_dirname, toplevel);

	//
	Torch::print("inplane_router machine: %s\n", inplane_router);
	Torch::print("inplane_router threshold: %g\n", threshold_inplanerouter);
	if(norm_)
	{
		Torch::print("inplane_router mu: %g\n", mu_inplanerouter);
		Torch::print("inplane_router stdv: %g\n", stdv_inplanerouter);
	}

	//...
	CascadeMachine *inplane_router_cascade;

	inplane_router_cascade = loadCascade(input_dirname, inplane_router);

	//
	Torch::print("outplane_router machine: %s\n", outplane_router);
	Torch::print("outplane_router threshold: %g\n", threshold_outplanerouter);
	if(norm_)
	{
		Torch::print("outplane_router mu: %g\n", mu_outplanerouter);
		Torch::print("outplane_router stdv: %g\n", stdv_outplanerouter);
	}

	//...
	CascadeMachine *outplane_router_cascade;

	outplane_router_cascade = loadCascade(input_dirname, outplane_router);

	//
	Torch::print("inplane pyramid machines: %s\n", inplane_list);
	Torch::print("inplane pyramid connect: %s\n", inplane_connect);

	//...
	int n_inplane_nodes = 0;
	char **inplane_models_filename = NULL;
	double *inplane_models_threshold = NULL;
	char **inplane_models_name = NULL;
	int *inplane_models_id = NULL;
	double *inplane_models_mu = NULL;
	double *inplane_models_stdv = NULL;
        
	CmdFile cmdf_inplane;

	cmdf_inplane.info("List of in-plane models and corresponding thresholds");
	
        cmdf_inplane.addText("\nNumber of in-plane models : ");
        cmdf_inplane.addICmd("n_models", &n_inplane_nodes, "Number of models");

	char inplane_models_filename_[250];
	sprintf(inplane_models_filename_, "%s/%s", input_dirname, inplane_list); 

        cmdf_inplane.read(inplane_models_filename_, false);

        cmdf_inplane.addText("\nInit models filename : ");
        cmdf_inplane.addHrefSCmdOption("model[%d]", &inplane_models_filename, "", "Model filename", n_inplane_nodes);

        cmdf_inplane.addText("\nInit thresholds : ");
        cmdf_inplane.addHrefDCmdOption("threshold[%d]", &inplane_models_threshold, -1000, "Threshold", n_inplane_nodes);

        cmdf_inplane.addText("\nInit models name : ");
        cmdf_inplane.addHrefSCmdOption("name[%d]", &inplane_models_name, "", "Model name", n_inplane_nodes);

        cmdf_inplane.addText("\nInit models id : ");
        cmdf_inplane.addHrefICmdOption("id[%d]", &inplane_models_id, 0, "Id", n_inplane_nodes);

	if(norm_)
	{
        	cmdf_inplane.addText("\nNormalisation : ");
        	cmdf_inplane.addHrefDCmdOption("mu[%d]", &inplane_models_mu, 0.0, "Mean", n_inplane_nodes);
        	cmdf_inplane.addHrefDCmdOption("stdv[%d]", &inplane_models_stdv, 1.0, "Variance", n_inplane_nodes);
	}

        cmdf_inplane.readHref();

        if(strcmp(inplane_models_filename[n_inplane_nodes-1],"") == 0)
        	warning("Number of models declared != Number of models actually listed");
	
        print("# Number of inplane models : %d\n", n_inplane_nodes);

	CascadeMachine **inplane_cascades = new CascadeMachine* [n_inplane_nodes];

        for(int i = 0 ; i < n_inplane_nodes ; i++)
        {
               	print(" + model[%d] : %s\n", i, inplane_models_filename[i]);
		Torch::print("   threshold = %g\n", inplane_models_threshold[i]);
               	print("   name : %s\n", inplane_models_name[i]);
               	print("   id = %d\n", inplane_models_id[i]);
		if(norm_)
		{
               		print("   mu = %g\n", inplane_models_mu[i]);
               		print("   stdv = %g\n", inplane_models_stdv[i]);
		}

		inplane_cascades[i] = loadCascade(input_dirname, inplane_models_filename[i]);
        }


	// connect the nodes of the inplane sub-tree
	int n_rows, n_cols;
	File *file_ = NULL;

	char inplane_connect_models_filename_[250];
	sprintf(inplane_connect_models_filename_, "%s/%s", input_dirname, inplane_connect); 

	file_ = new File;
	file_->open(inplane_connect_models_filename_, "r");

	file_->scanf("%d", &n_rows);
	file_->scanf("%d", &n_cols);

	print("Loading connect file %s ...\n", inplane_connect_models_filename_);
	print("	n rows = %d\n", n_rows);
	print("	n cols = %d\n", n_cols);

	if((n_rows != n_cols) && (n_rows != n_inplane_nodes)) error("Number of rows/cols incorrect in file %s.", inplane_connect_models_filename_);
	   
	char str[50];
	for(int row = 0 ; row < n_rows ; row++)
	{
		for(int col = 0 ; col < n_cols ; col++)
		{
		   	file_->scanf("%s", &str);
			
			print("%s ", str);

			switch(str[0])
			{
			case 'L': 
				//inplane_nodes[row]->connectLeft(inplane_nodes[col]);
				break;
			case 'R': 
				//inplane_nodes[row]->connectRight(inplane_nodes[col]);
				break;
			}
		}
		print("\n");   
	}

	file_->close();
	delete file_;


	//
	Torch::print("outplane pyramid machines: %s\n", outplane_list);
	Torch::print("outplane pyramid connect: %s\n", outplane_connect);
	
	//...
	int n_outplane_nodes = 0;
	char **outplane_models_filename = NULL;
	double *outplane_models_threshold = NULL;
	char **outplane_models_name = NULL;
	int *outplane_models_id = NULL;
	double *outplane_models_mu = NULL;
	double *outplane_models_stdv = NULL;
        
	CmdFile cmdf_outplane;

	cmdf_outplane.info("List of in-plane models and corresponding thresholds");
	
        cmdf_outplane.addText("\nNumber of in-plane models : ");
        cmdf_outplane.addICmd("n_models", &n_outplane_nodes, "Number of models");

	char outplane_models_filename_[250];
	sprintf(outplane_models_filename_, "%s/%s", input_dirname, outplane_list); 

        cmdf_outplane.read(outplane_models_filename_, false);

        cmdf_outplane.addText("\nInit models filename : ");
        cmdf_outplane.addHrefSCmdOption("model[%d]", &outplane_models_filename, "", "Model filename", n_outplane_nodes);

        cmdf_outplane.addText("\nInit thresholds : ");
        cmdf_outplane.addHrefDCmdOption("threshold[%d]", &outplane_models_threshold, -1000, "Threshold", n_outplane_nodes);

        cmdf_outplane.addText("\nInit models name : ");
        cmdf_outplane.addHrefSCmdOption("name[%d]", &outplane_models_name, "", "Model name", n_outplane_nodes);

        cmdf_outplane.addText("\nInit models id : ");
        cmdf_outplane.addHrefICmdOption("id[%d]", &outplane_models_id, 0, "Id", n_outplane_nodes);

	if(norm_)
	{
        	cmdf_outplane.addText("\nNormalisation : ");
        	cmdf_outplane.addHrefDCmdOption("mu[%d]", &outplane_models_mu, 0.0, "Mean", n_outplane_nodes);
        	cmdf_outplane.addHrefDCmdOption("stdv[%d]", &outplane_models_stdv, 1.0, "Variance", n_outplane_nodes);
	}

        cmdf_outplane.readHref();

        if(strcmp(outplane_models_filename[n_outplane_nodes-1],"") == 0)
        	warning("Number of models declared != Number of models actually listed");
	
        print("# Number of outplane models : %d\n", n_outplane_nodes);

	CascadeMachine **outplane_cascades = new CascadeMachine* [n_outplane_nodes];

        for(int i = 0 ; i < n_outplane_nodes ; i++)
        {
               	print(" + model[%d] : %s\n", i, outplane_models_filename[i]);
		Torch::print("   threshold = %g\n", outplane_models_threshold[i]);
               	print("   name : %s\n", outplane_models_name[i]);
               	print("   id = %d\n", outplane_models_id[i]);
		if(norm_)
		{
               		print("   mu = %g\n", outplane_models_mu[i]);
               		print("   stdv = %g\n", outplane_models_stdv[i]);
		}

		outplane_cascades[i] = loadCascade(input_dirname, outplane_models_filename[i]);
        }

	// connect the nodes of the outplane sub-tree

	char outplane_connect_models_filename_[250];
	sprintf(outplane_connect_models_filename_, "%s/%s", input_dirname, outplane_connect); 

	file_ = new File;
	file_->open(outplane_connect_models_filename_, "r");

	file_->scanf("%d", &n_rows);
	file_->scanf("%d", &n_cols);

	print("Loading connect file %s ...\n", outplane_connect_models_filename_);
	print("	n rows = %d\n", n_rows);
	print("	n cols = %d\n", n_cols);

	if((n_rows != n_cols) && (n_rows != n_outplane_nodes)) error("Number of rows/cols incorrect in file %s.", outplane_connect_models_filename_);
	   
	for(int row = 0 ; row < n_rows ; row++)
	{
		for(int col = 0 ; col < n_cols ; col++)
		{
		   	file_->scanf("%s", &str);
			
			print("%s ", str);

			switch(str[0])
			{
			case 'L': 
				//outplane_nodes[row]->connectLeft(outplane_nodes[col]);
				break;
			case 'R': 
				//outplane_nodes[row]->connectRight(outplane_nodes[col]);
				break;
			}
		}
		print("\n");   
	}

	file_->close();
	delete file_;






        TreeClassifier tree_classifier;

	int n_nodes = 0;

	// get the number of nodes
	// ... toplevel + 1 node (with inplane and outplane routers) + 1 node per sub-cascades 
	n_nodes = 2 + n_outplane_nodes + n_inplane_nodes;
	
	// set the number of nodes
        if (tree_classifier.resize(n_nodes) == false)
        {
        	return false;
        }

	int n_classes = 0;

	// set the number of classes (0=negative, 1=class1, 2=class2, ...)
	n_classes = n_outplane_nodes + n_inplane_nodes;
	tree_classifier.setClasses(n_classes);

        print("N_CLASSES = %d\n", n_classes);
        print("N_NODES = %d\n", n_nodes);
		
	// NODE 0: toplevel node
	tree_classifier.resize(0, 1); 
        tree_classifier.setClassifier(0, 0, toplevel_cascade);
        tree_classifier.setThreshold(0, 0, threshold_toplevel);
       
	// NODE 0 is connected to two nodes:
	// 	classifier 0 -> NODE 1
	tree_classifier.setChild(0, 0, 1);
	// 	reject node (classifier+1) -> NODE n_nodes
	tree_classifier.setChild(0, 1, n_nodes);

	// NODE 1: (inplane + outplane) node
	tree_classifier.resize(1, 2);
        tree_classifier.setClassifier(1, 0, inplane_router_cascade, mu_inplanerouter, stdv_inplanerouter);
        tree_classifier.setThreshold(1, 0, threshold_inplanerouter);
        tree_classifier.setClassifier(1, 1, outplane_router_cascade, mu_outplanerouter, stdv_outplanerouter);
        tree_classifier.setThreshold(1, 1, threshold_outplanerouter);
	
	int offset_inplane_node = 2;  			// the first of the inplane nodes
	int offset_outplane_node = 2+n_inplane_nodes;	// the first of the outplane nodes

	// NODE 1 is connected to two nodes:
	// 	classifier 0 -> NODE offset_inplane_node
	tree_classifier.setChild(1, 0, offset_inplane_node);
	// 	classifier 1 -> NODE offset_outplane_node
	tree_classifier.setChild(1, 1, offset_outplane_node);
	// 	reject node (classifier+1) -> NODE n_nodes
	tree_classifier.setChild(1, 2, n_nodes);

	int node_ = 2;
	// inplane nodes 
        for (int s = 0; s < n_inplane_nodes; s++) 
	{
		tree_classifier.resize(node_, 1); 

        	tree_classifier.setClassifier(node_, 0, inplane_cascades[s], inplane_models_mu[s], inplane_models_stdv[s]);
        	tree_classifier.setThreshold(node_, 0, inplane_models_threshold[s]);
	
		// NODE node_ is connected to NODE n_nodes + inplane_models_id[s] + 1
		tree_classifier.setChild(node_, 0, n_nodes + inplane_models_id[s] + 1);

		if(s == n_inplane_nodes-1)
		{
			// 	reject node (classifier+1) -> NODE n_nodes
			tree_classifier.setChild(node_, 1, n_nodes);
		}
		else
		{
			// 	reject node (classifier+1) -> NODE n_nodes
			tree_classifier.setChild(node_, 1, offset_inplane_node+s+1);
		}

		node_++;
	}
	// outplane nodes
        for (int s = 0; s < n_outplane_nodes; s++) 
	{
		tree_classifier.resize(node_, 1); 

        	tree_classifier.setClassifier(node_, 0, outplane_cascades[s], outplane_models_mu[s], outplane_models_stdv[s]);
        	tree_classifier.setThreshold(node_, 0, outplane_models_threshold[s]);

		// NODE node_ is connected to NODE n_nodes + outplane_models_id[s] + 1
		tree_classifier.setChild(node_, 0, n_nodes + outplane_models_id[s] + 1);

		if(s == n_outplane_nodes-1)
		{
			// 	reject node (classifier+1) -> NODE n_nodes
			tree_classifier.setChild(node_, 1, n_nodes);
		}
		else
		{
			// 	reject node (classifier+1) -> NODE n_nodes
			tree_classifier.setChild(node_, 1, offset_outplane_node+s+1);
		}
		node_++;
	}

        // For each node ...
        for (int s = 0; s < n_nodes; s ++)
        {
		// Number of machines per node
                const int n_trainers = tree_classifier.getNoClassifiers(s);

		print("\t[%d/%d] N_TRAINERS = %d\n", s + 1, n_nodes, n_trainers);
        }

	// Force the model size to all Machines
	TensorSize modelsize(19, 19);
	tree_classifier.setSize(modelsize);

	// OK, just let the TreeClassifier object to write his structure to the output file
	File file_out;
	CHECK_FATAL(file_out.open(output_filename, "w") == true);
        tree_classifier.saveFile(file_out);
	file_out.close();

	delete [] inplane_cascades;
	delete [] outplane_cascades;

	return true;
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		print("Parameters: <torch3 pyramid cascade model filename> <torch5spro tree model filename>!\n");
		return -1;
	}

	const char* in_dirname = argv[1];
	const char* out_filename = argv[2];

	print("---------------------------------------------------\n");
	print("INPUT DIR   : [%s]\n", in_dirname);
	print("OUTPUT FILE : [%s]\n", out_filename);
	print("---------------------------------------------------\n");

	// Do the conversion
	if (convert(in_dirname, out_filename) == false)
	{
		print("Failed to convert!\n\n");
	}
	else
	{
		print("Conversion finished!\n\n");

		// Test the loading
		{
			TreeClassifier tree;
			File file_in;
			CHECK_FATAL(file_in.open(out_filename, "r") == true);
			if (tree.loadFile(file_in) == true)
			{
				print(">>>>>>>>>>>>>> CHECKED! <<<<<<<<<<<<<<<\n\n");

				int n_classes = tree.getClasses();
				print("Number of classes: %d\n", n_classes);

				for (int i = 0; i < tree.getNoNodes(); i ++)
				{
					print(">>> node [%d/%d]: no. classifiers = %d\n", i, tree.getNoNodes(), tree.getNoClassifiers(i));

					for(int j = 0 ; j < tree.getNoClassifiers(i)+1 ; j++)
						print("   child [%d] = %02d\n", j, tree.getChild(i, j));
				}
			}
			else
			{
				print(">>>>>>>>>>>>>> The converted file model is NOT valid! <<<<<<<<<<<<<<\n\n");
			}
			file_in.close();
		}
	}
	print("---------------------------------------------------\n");

        print("\nOK\n");

	return 0;
}
