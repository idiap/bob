#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
        // Check arguments
        if (argc != 3)
        {
                print("\nUsage: <test19x19models> <model filename> <tensor file to test>\n\n");
                return 1;
        }
        const char* model_filename = argv[1];
        const char* data_filename = argv[2];

        // Load the tree classifier
	TreeClassifier* tree = dynamic_cast<TreeClassifier*>(Torch::loadMachineFromFile(model_filename));
	if (tree == 0)
	{
		print("ERROR: loading model [%s]!\n", model_filename);
		return 1;
	}
	const int model_h = tree->getSize().size[0];
	const int model_w = tree->getSize().size[1];
	print("Tree [%s]: width = %d, height = %d\n", model_filename, model_w, model_h);
	tree->setRegion(TensorRegion(0, 0, model_h, model_w));

	int n_classes = tree->getClasses();
	print("Number of classes: %d\n", n_classes);

	if(n_classes == 0)
	{
		print("ERROR: no classes in the model !\n");
		return 1;
	}

	double *confidence_classes = new double [n_classes+1];
	int    *detection_classes = new int [n_classes+1];

	for (int i = 0; i < tree->getNoNodes(); i ++)
	{
		print(">>> node [%d/%d]: no. classifiers = %d\n", i, tree->getNoNodes(), tree->getNoClassifiers(i));

		for(int j = 0 ; j < tree->getNoClassifiers(i)+1 ; j++)
			print("   child [%d] = %02d\n", j, tree->getChild(i, j));
	}
	
	for (int j = 0; j < n_classes+1; j++)
	{
		confidence_classes[j] = 0.0;
		detection_classes[j] = 0;
	}

	// Load the tensors to test
	TensorFile tf;
	if (tf.openRead(data_filename) == true)
	{
		Tensor* sample = 0;
		int n_samples = 0;
		int n_patterns = 0;

		while ((sample = tf.load()) != 0)
		{
			if (tree->forward(*sample) == false)
			{
				print("ERROR: failed to run the tree on the tensor [%d]!\n", n_samples);
				return 1;
			}

			if(tree->isPattern())
			{
				n_patterns++;

				int c = tree->getPatternClass();
				double z = tree->getConfidence();

				if(c < 1) warning("Incorrect class %d < 1 !!!", c);
				if(c > n_classes) warning("Incorrect class %d > %d !!!", c, n_classes);

				confidence_classes[c] += z;
				detection_classes[c] += 1;
			}
			else
			{
				int c = tree->getPatternClass();

				if(c != 0) warning("Sample is not a pattern, the class should be 0 !!!");
				
				double z = tree->getConfidence();

				confidence_classes[0] += z;
				detection_classes[0] += 1;
			}

			n_samples++;
		}

		print("\n");

		// Print the results
		print("---------------------------------------------------\n");
		print("  Overall classification rate: [%d/%d] = %f%% \n", n_patterns, n_samples, 100.0f * (n_patterns + 0.0f) / (n_samples + 0.0f));
		print("  Classification rate per class:\n");
		for (int j = 0; j < n_classes+1; j++)
		{
			if(detection_classes[j] == 0)
				print("  Class [%d]: [%d/%d] = %f%% \n", j, detection_classes[j], n_samples, 100.0f * (detection_classes[j] + 0.0f) / (n_samples + 0.0f));
			else print("  Class [%d]: [%d/%d] = %f%% with mean confidence %g \n", j, detection_classes[j], n_samples, 100.0f * (detection_classes[j] + 0.0f) / (n_samples + 0.0f), (confidence_classes[j] + 0.0f) / detection_classes[j]);
		}
		print("---------------------------------------------------\n");


	}
	else
	{
		print("ERROR: loading tensor file [%s]!\n", data_filename);
	}

	delete []confidence_classes;
	delete []detection_classes;

	return 0;
}

