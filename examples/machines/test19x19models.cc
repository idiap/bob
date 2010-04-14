#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
        // Check arguments
        if (argc != 4)
        {
                print("\nUsage: <test19x19models> <model filename> <tensor file to test> <threshold>\n\n");
                return 1;
        }
        const char* model_filename = argv[1];
        const char* data_filename = argv[2];
	const char* threshold = argv[3];

        // Load the cascade machine
	CascadeMachine* cascade = dynamic_cast<CascadeMachine*>(Torch::loadMachineFromFile(model_filename));
	if (cascade == 0)
	{
		print("ERROR: loading model [%s]!\n", model_filename);
		return 1;
	}
	const int model_h = cascade->getSize().size[0];
	const int model_w = cascade->getSize().size[1];
	print("Cascade [%s]: width = %d, height = %d\n", model_filename, model_w, model_h);
	cascade->setRegion(TensorRegion(0, 0, model_h, model_w));

	for (int i = 0; i < cascade->getNoStages(); i ++)
	{
		print(">>> stage [%d/%d]: no. machines = %d, threshold = %lf\n",
			i + 1, cascade->getNoStages(), cascade->getNoMachines(i), cascade->getThreshold(i));
	}

	double theta = atof(threshold);

	print("threshold = %g\n", theta);
	
	// Load the tensors to test
	TensorFile tf;
	if (tf.openRead(data_filename) == true)
	{
		Tensor* sample = 0;
		int n_samples = 0;
		int n_patterns = 0;

		while ((sample = tf.load()) != 0)
		{
			if (cascade->forward(*sample) == false)
			{
				print("ERROR: failed to run the cascade on the tensor [%d]!\n", n_samples);
				return 1;
			}

			print("CONFIDENCE %f\n", cascade->getConfidence());
			
			//n_patterns += cascade->isPattern() ? 1 : 0;

			if(cascade->isPattern())
			{
				n_patterns += (cascade->getConfidence() >= theta) ? 1 : 0;
			}

			n_samples ++;
		}

		print("\n");

		// Print the results
		print("---------------------------------------------------\n");
		print(">>> classification rate: [%d/%d] = %f%% <<<\n",
			n_patterns, n_samples, 100.0f * (n_patterns + 0.0f) / (n_samples + 0.0f));
		print("---------------------------------------------------\n");
	}

	else
	{
		print("ERROR: loading tensor file [%s]!\n", data_filename);
	}

	return 0;
}

