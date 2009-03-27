#include "LutTrainer.h"

namespace Torch
{
        LutTrainer::LutTrainer(LutMachine *lut_machine_, int n_features_, spCore **features_) 
	   : WeakLearner(lut_machine_, n_features_, features_)
	{
	   	m_lut_machine = lut_machine_;

	}

	bool LutTrainer::train()
	{
        	print("\n");
        	print("   LutTrainer::train()\n");


	   	if(m_shuffledindex_dataset == NULL)
		{
		  	Torch::error("LutTrainer::train() no shuffle index provided.");

			return false;
		}

	   	if(m_weights_dataset == NULL)
		{
		  	Torch::error("LutTrainer::train() no weights provided.");

			return false;
		}

		if(m_lut_machine == NULL)
		{
		  	Torch::error("LutTrainer::train() no LUT machine provided.");

			return false;
		}

		// test the number of features
		print("    + Number of features = %d\n", m_n_features);
		if(m_n_features <= 0)
		{
		  	Torch::error("LutTrainer::train() no features available.");

			return false;
		}

		// test the number of examples
		int n_examples = m_dataset->getNoExamples();
		print("    + Number of examples = %d\n", n_examples);
		if(n_examples <= 0)
		{
		  	Torch::error("LutTrainer::train() no examples available.");

			return false;
		}

		// test if the dataset has targets
		if(m_dataset->hasTargets() != true)
		{
		   	Torch::error("LutTrainer::train() no targets in the dataset.");

			return false;
		}

		// test target type, size and value
		for(int i = 0; i < n_examples ; i++)
		{
			Tensor *tensor = m_dataset->getTarget(i);

			// test the type
			if(tensor->getDatatype() != Tensor::Short)
			{
			   	Torch::error("LutTrainer::train() targets should be ShortTensor.");

				return false;
			}

			// test the size
			ShortTensor *target = (ShortTensor *) tensor;

			if(target->nDimension() != 1)
			{
			   	Torch::error("LutTrainer::train() target tensor should be 1 dimension.");

				return false;
			}

			if(target->size(0) != 1)
			{
			   	Torch::error("LutTrainer::train() target tensor should be of size 1.");

				return false;
			}

			// test the value
			short target_value = (*target)(0);

			if(target_value != 0 && target_value != 1)
			{
			  	Torch::error("LutTrainer::train() target values should be 0 or 1.");

				return false;
			}
		}


		//
		for(int f = 0; f < m_n_features ; f++)
		{
			// do a first pass to determine the min and max
			float min_ = FLT_MAX;
			float max_ = -FLT_MAX;

			for(int i = 0; i < n_examples ; i++)
			{
			   	// Warning use shuffled index here !!!
			   	Tensor *example = m_dataset->getExample(i);

				// here we should test first the type and size of the returned tensor
				m_features[f]->process(*example);
				DoubleTensor *feature_value = (DoubleTensor *) &m_features[f]->getOutput(0);
				//feature->print();

				// store the features for more efficiency
				float z = (*feature_value)(0);
				//features_values[i] = z;

				if(z < min_) min_ = z;
				else if(z > max_) max_ = z;
			}

		}

		//m_weak_classifier->setCore(m_features[bestFeature]);
		//m_lut_machine->setParams(min, max, n_bins, lut);

        	print("\n");

        	return true;
	}


	LutTrainer::~LutTrainer()
	{
	}

}
