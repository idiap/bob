#include "MultiVariateNormalDistribution.h"

namespace Torch {

MultiVariateNormalDistribution::MultiVariateNormalDistribution()
{
	addFOption("min weights", 1e-3, "minimum weights for each mean");

	//
	n_means = 0;
	means = NULL;
	weights = NULL;
	variances = NULL;
	threshold_variances = NULL;

	//
	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	buffer_acc_posteriors_variances = NULL;
	acc_posteriors_variances = NULL;
	current_likelihood_one_mean = NULL;

	frame_ = new DoubleTensor();
	sequence_ = new DoubleTensor();
	
	//
	m_parameters->addI("n_inputs", 0, "number of dimensions of the multi-variate normal distribution");
	m_parameters->addI("n_means", 0, "number of means of the multi-variate normal distribution");
	m_parameters->addDarray("weigths", 0, 0.0, "weights of the multi-variate normal distribution");
	m_parameters->addDarray("means", 0, 0.0, "means of the multi-variate normal distribution");
	m_parameters->addDarray("variances", 0, 0.0, "variances of the diagonal gaussian distribution");
}

MultiVariateNormalDistribution::MultiVariateNormalDistribution(int n_inputs_, int n_means_) : ProbabilityDistribution(n_inputs_)
{
	addFOption("min weights", 1e-3, "minimum weights for each mean");

   	//
	n_means = n_means_;
	means = NULL;
	weights = NULL;
	variances = NULL;
	threshold_variances = NULL;

	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	buffer_acc_posteriors_variances = NULL;
	acc_posteriors_variances = NULL;
	current_likelihood_one_mean = NULL;

	frame_ = new DoubleTensor();
	sequence_ = new DoubleTensor();

	//
	m_parameters->addI("n_inputs", n_inputs, "number of dimensions of the multi-variate normal distribution");
	m_parameters->addI("n_means", n_means, "number of means of the multi-variate normal distribution");
	m_parameters->addDarray("weigths", n_means, 0.0, "weights of the multi-variate normal distribution");
	m_parameters->addDarray("means", n_means*n_inputs, 0.0, "means of the multi-variate normal distribution");
	m_parameters->addDarray("variances", n_means*n_inputs, 0.0, "variances of the diagonal gaussian distribution");

	//
	resize(n_inputs_, n_means_);
}

bool MultiVariateNormalDistribution::resize(int n_inputs_, int n_means_)
{
	//Torch::print("MultiVariateNormalDistribution::resize(%d, %d)\n", n_inputs_, n_means_);
	
	//
	weights = m_parameters->getDarray("weigths");
	double *means_ = m_parameters->getDarray("means");
	means = (double **) THAlloc(n_means_ * sizeof(double *));
	double *p = means_;
	for(int j = 0 ; j < n_means_ ; j++)
	{
		means[j] = p; 
		p += n_inputs_;
	}

	//
	current_likelihood_one_mean = (double *) THAlloc(n_means_ * sizeof(double));
	for(int j = 0 ; j < n_means_ ; j++) current_likelihood_one_mean[j] = 0.0;

	//
	acc_posteriors_weights = (double *) THAlloc(n_means_ * sizeof(double));
	buffer_acc_posteriors_means = (double *) THAlloc(n_means_ * n_inputs_ * sizeof(double));
	acc_posteriors_means = (double **) THAlloc(n_means_ * sizeof(double *));

	for(int j = 0 ; j < n_means_ ; j++)
		acc_posteriors_means[j] = &buffer_acc_posteriors_means[j*n_inputs_];

	//
	double *variances_ = m_parameters->getDarray("variances");

	variances = (double **) THAlloc(n_means_ * sizeof(double *));
	p = variances_;
	for(int j = 0 ; j < n_means_ ; j++)
	{
		variances[j] = p;
		p += n_inputs_;
	}
	
	//
	threshold_variances = (double *) THAlloc(n_inputs_ * sizeof(double));
	for(int i = 0 ; i < n_inputs_ ; i++) threshold_variances[i] = 1e-10;

	buffer_acc_posteriors_variances = (double *) THAlloc(n_means_ * n_inputs_ * sizeof(double));
	acc_posteriors_variances = (double **) THAlloc(n_means_ * sizeof(double *));

	for(int j = 0 ; j < n_means_ ; j++)
		acc_posteriors_variances[j] = &buffer_acc_posteriors_variances[j*n_inputs_];

	return true;
}

bool MultiVariateNormalDistribution::cleanup()
{
	//Torch::print("MultiVariateNormalDistribution::cleanup()\n");

	if(acc_posteriors_means != NULL) THFree(acc_posteriors_means);
	if(buffer_acc_posteriors_means != NULL) THFree(buffer_acc_posteriors_means);
	if(acc_posteriors_weights != NULL) THFree(acc_posteriors_weights);
	if(current_likelihood_one_mean != NULL) THFree(current_likelihood_one_mean);
	if(means != NULL) THFree(means);
	if(acc_posteriors_variances != NULL) THFree(acc_posteriors_variances);
	if(buffer_acc_posteriors_variances != NULL) THFree(buffer_acc_posteriors_variances);
	if(threshold_variances != NULL) THFree(threshold_variances);
	if(variances != NULL) THFree(variances);

	return true;
}

MultiVariateNormalDistribution::~MultiVariateNormalDistribution()
{	
	cleanup();
	delete frame_;
	delete sequence_;
}

bool MultiVariateNormalDistribution::EMinit()
{
	float min_weights = getFOption("min weights");

	acc_posteriors_sum_weights = 0.0;
	for(int j = 0 ; j < n_means ; j++)
	{
		acc_posteriors_weights[j] = min_weights;

		for(int k = 0 ; k < n_inputs ; k++)
		{
			acc_posteriors_means[j][k] = 0.0;
			acc_posteriors_variances[j][k] = 0.0;
		}
	}

	return true;
}
bool MultiVariateNormalDistribution::forward(const DoubleTensor *input)
{
	//
	// If the tensor is 1D then considers it as a vector
	if (	input->nDimension() == 1)
	{
		if (	input->size(0) != n_inputs)
		{
			warning("MultiVariateNormalDistribution::forward() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
			
			return false;
		}

		double *src = (double *) input->dataR();
		double *dst = (double *) m_output.dataW();

		dst[0] = sampleProbability(src);
	}
	else
	{
		//
		// If the tensor is 2D/3D then considers it as a sequence along the first dimension

   		if(input->nDimension() == 2)
		{
			if (	input->size(0) != n_inputs)
			{
				warning("MultiVariateNormalDistribution::forward() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
				
				return false;
			}
		
			int n_frames_per_sequence = input->size(1);

			//Torch::print("MultiVariateNormalDistribution::forward() processing a sequence of %d frames of size %d\n", n_frames_per_sequence, n_inputs);

			double ll = 0;
			for(int f = 0 ; f < n_frames_per_sequence ; f++)
			{
				frame_->select(input, 1, f);

				double *src = (double *) frame_->dataR();

				ll += sampleProbability(src);
			}

			double *dst = (double *) m_output.dataW();
			dst[0] = ll / (double) n_frames_per_sequence;
		}
		else if(input->nDimension() == 3)
		{
			if (	input->size(0) != n_inputs)
			{
				warning("MultiVariateNormalDistribution::forward() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
				
				return false;
			}
			int n_sequences_per_sequence = input->size(2);
			int n_frames_per_sequence = input->size(1);

			//Torch::print("MultiVariateNormalDistribution::forward() processing a sequence of %d sequences of %d frames of size %d\n", n_sequences_per_sequence, n_frames_per_sequence, n_inputs);

			double ll = 0;
			for(int s = 0 ; s < n_sequences_per_sequence ; s++)
			{
				sequence_->select(input, 2, s);
				
				for(int f = 0 ; f < n_frames_per_sequence ; f++)
				{
					frame_->select(sequence_, 1, f);

					double *src = (double *) frame_->dataR();

					ll += sampleProbability(src);
				}
			}

			double *dst = (double *) m_output.dataW();
			dst[0] = ll / (double) (n_frames_per_sequence * n_sequences_per_sequence);
		}
		else 
		{
			warning("MultiVariateNormalDistribution::forward() : don't know how to deal with %d dimensions sorry :-(", input->nDimension());
			
			return false;
		}

	}

	return true;
}

bool MultiVariateNormalDistribution::setMeans(double **means_)
{
	for(int j = 0 ; j < n_means ; j++) 
	{
		for(int k = 0 ; k < n_inputs ; k++)
		{
			means[j][k] = means_[j][k];
			variances[j][k] = threshold_variances[k];
		}
		weights[j] = 1.0 / (double) n_means;
	}

	return true;
}

bool MultiVariateNormalDistribution::setMeans(DataSet *dataset_)
{
	// init only means from assigning a random sample per partitions

	/* !! Warning !!
 
	   To fix to deal with 2D and 3D tensors

	*/

	if(dataset_ == NULL) return false;

	// checking tensors in dataset
	int n_data = 0;
	int tensor_dim = 0;
	Tensor *example = NULL;

	Torch::print("Number of examples in the dataset %d\n", dataset_->getNoExamples());
	for(int i = 0 ; i < dataset_->getNoExamples() ; i++) 
	{
		example = dataset_->getExample(i);

		if (	example->getDatatype() != Tensor::Double)
		{
			warning("MultiVariateNormalDistribution::setMeans() : incorrect tensor type.");
			return false;
		}
		if (	example->size(0) != n_inputs)
		{
			warning("MultiVariateNormalDistribution::setMeans() : incorrect input size along dimension 0 (%d != %d).", example->size(0), n_inputs);
			return false;
		}

		if(i == 0) tensor_dim = example->nDimension();
		else
		{
			if(example->nDimension() != tensor_dim)
			{
				warning("MultiVariateNormalDistribution::setMeans() : all tensors should have the same number of dimensions (%d).", tensor_dim);
				return false;
			}
		}

		if (	example->nDimension() == 1) n_data++;
		else if (example->nDimension() == 2) n_data += example->size(1);
		else if (example->nDimension() == 3) n_data += example->size(1) * example->size(2);
		else 
		{
			warning("MultiVariateNormalDistribution::setMeans() : incorrect number of dimensions (only 1, 2 or 3).");
			return false;
		}
	}

	Torch::print("Total number of data samples %d\n", n_data);

	if(n_means > n_data) warning("MultiVariateNormalDistribution::setMeans() There are more means than samples. This could creates some troubles.");

	int n_partitions = (int)(n_data / (double) n_means);

	double *src = NULL;
	DoubleTensor *t_input = NULL;
	int n_ = 0;

	DoubleTensor *frame_ = new DoubleTensor;
	DoubleTensor *sequence_ = new DoubleTensor;

	for(int j = 0 ; j < n_means ; j++) 
	{
		int offset = j*n_partitions;
		int index = offset + (int)(THRandom_uniform(0, 1)*(double) n_partitions);

		//Torch::print("Index to match %d\n", index);
		
		if(index < 0) warning("under limit");
		if(index >= n_data) warning("over limit");

		switch(tensor_dim)
		{
		case 1: 
		   	// An example of dimension 1 is necessary 1 vector of size n_inputs
		   	example = dataset_->getExample(index);
			t_input = (DoubleTensor *) example;
			src = (double *) t_input->dataR();
			break;
		case 2:
		   	// An example of dimension 2 is necessary a sequence of vectors of size n_inputs
			n_ = 0;
			for(int i = 0 ; i < dataset_->getNoExamples() ; i++) 
			{
				example = dataset_->getExample(i);
				if(n_ + example->size(1) > index)
				{
				   	int offset_ = index - n_; 
				   	//Torch::print("Offset %d\n", offset_);
					t_input = (DoubleTensor *) example;
					frame_->select(t_input, 1, offset_);
					src = (double *) frame_->dataR();
					break;
				}
				n_ += example->size(1);
				//Torch::print("+%d (%d) ", example->size(1), n_);
			}
			break;
		case 3:
		   	// An example of dimension 3 is necessary a sequence of sequence of vectors of size n_inputs
			n_ = 0;
			for(int i = 0 ; i < dataset_->getNoExamples() ; i++) 
			{
				example = dataset_->getExample(i);
				t_input = (DoubleTensor *) example;

				//Torch::print("sequence of %d sequence of %d vectors\n", example->size(2), example->size(1));
				
				if(n_ + example->size(1)*example->size(2) > index)
				{
				   	int offset_seq = index - n_;
					   			
					int offset_seq_seq = offset_seq / example->size(1);
					int offset_frame_seq = offset_seq % example->size(1);

					sequence_->select(t_input, 2, offset_seq_seq);
					frame_->select(sequence_, 1, offset_frame_seq);
					src = (double *) frame_->dataR();
					break;
				}
				n_ += example->size(1) * example->size(2);
				//Torch::print("+%dx%d (%d) ", example->size(2), example->size(1), n_);
			}
			break;
		}

		for(int k = 0 ; k < n_inputs ; k++)
		{
		   	means[j][k] = src[k];
			variances[j][k] = threshold_variances[k];
		}
		weights[j] = 1.0 / (double) n_means;
	}
	
	delete sequence_;
	delete frame_;

	return true;
}

bool MultiVariateNormalDistribution::shuffle()
{
	//Torch::print("MultiVariateNormalDistribution::shuffle()\n");
	
   	double z = 0.0;

	for(int j = 0 ; j < n_means ; j++)
	{
		weights[j] = THRandom_uniform(0, 1);
		z += weights[j];

		for(int k = 0 ; k < n_inputs ; k++)
		{
			means[j][k] = THRandom_uniform(0, 1);
			variances[j][k] = THRandom_uniform(0, 1);
		}
	}

	for(int j = 0 ; j < n_means ; j++) weights[j] /= z;

	return true;
}

bool MultiVariateNormalDistribution::print()
{
   	double z = 0.0;

	for(int j = 0 ; j < n_means ; j++)
	{
		Torch::print("Mean [%d]\n", j);

		Torch::print("   weight = %g\n", weights[j]);
		z += weights[j];

		Torch::print("   mean = [ ");
		for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", means[j][k]);
		Torch::print("]\n");

		Torch::print("   variance = [ ");
		for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", variances[j][k]);
		Torch::print("]\n");

	}
	Torch::print("Sum weights = %g\n", z);

	Torch::print("Variance flooring = [ ");
	for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", threshold_variances[k]);
	Torch::print("]\n");

	return true;
}
	
bool MultiVariateNormalDistribution::setVariances(double **variances_)
{
	for(int k = 0 ; k < n_inputs ; k++)
		for(int j = 0 ; j < n_means ; j++)
			variances[j][k] = variances_[j][k];

	return true;
}

bool MultiVariateNormalDistribution::setVariances(double *stdv_, double factor_variance_threshold_)
{
	// init variances and variance flooring to the given variance
	// Note: it could be interesting to compute the variance of samples for each cluster !

	//Torch::print("MultiVariateNormalDistribution::setVariances() flooring = %g\n", factor_variance_threshold_);

	for(int k = 0 ; k < n_inputs ; k++)
	{
	   	double z = stdv_[k];
		double zz = z * z;
		for(int j = 0 ; j < n_means ; j++)
			variances[j][k] = zz;
		threshold_variances[k] = zz * factor_variance_threshold_;

		//Torch::print("vflooring [%d] = %g (stdv = %g)\n", k, threshold_variances[k], stdv_[k]);
	}

	return true;
}

bool MultiVariateNormalDistribution::setVarianceFlooring(double *stdv_, double factor_variance_threshold_)
{
	//Torch::print("MultiVariateNormalDistribution::setVarianceFlooring() flooring = %g\n", factor_variance_threshold_);

	for(int k = 0 ; k < n_inputs ; k++)
	{
	   	double z = stdv_[k];
		threshold_variances[k] = z * z * factor_variance_threshold_;

		//Torch::print("vflooring [%d] = %g (stdv = %g)\n", k, threshold_variances[k], stdv_[k]);
	}

	return true;
}

bool MultiVariateNormalDistribution::EMaccPosteriors(const DoubleTensor *input, const double input_posterior)
{
	//
	// If the tensor is 1D then considers it as a vector
	if (	input->nDimension() == 1)
	{
		if (	input->size(0) != n_inputs)
		{
			warning("MultiVariateNormalDistribution::EMaccPosteriors() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
			
			return false;
		}

		double *src = (double *) input->dataR();

		sampleEMaccPosteriors(src, input_posterior);
	}
	else
	{
		//
		// If the tensor is 2D/3D then considers it as a sequence along the first dimension

   		if(input->nDimension() == 2)
		{
			if (	input->size(0) != n_inputs)
			{
				warning("MultiVariateNormalDistribution::EMaccPosteriors() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
				
				return false;
			}
		
			int n_frames_per_sequence = input->size(1);

			//Torch::print("MultiVariateNormalDistribution::EMaccPosteriors() processing a sequence of %d frames of size %d\n", n_frames_per_sequence, n_inputs);

			for(int f = 0 ; f < n_frames_per_sequence ; f++)
			{
				frame_->select(input, 1, f);

				double *src = (double *) frame_->dataR();

				sampleEMaccPosteriors(src, input_posterior);
			}
		}
		else if(input->nDimension() == 3)
		{
			if (	input->size(0) != n_inputs)
			{
				warning("MultiVariateNormalDistribution::EMaccPosteriors() : incorrect input size along dimension 0 (%d != %d).", input->size(0), n_inputs);
				
				return false;
			}
			int n_sequences_per_sequence = input->size(2);
			int n_frames_per_sequence = input->size(1);

			//Torch::print("MultiVariateNormalDistribution::EMaccPosteriors() processing a sequence of %d sequences of %d frames of size %d\n", n_sequences_per_sequence, n_frames_per_sequence, n_inputs);

			for(int s = 0 ; s < n_sequences_per_sequence ; s++)
			{
				sequence_->select(input, 2, s);
				
				for(int f = 0 ; f < n_frames_per_sequence ; f++)
				{
					frame_->select(sequence_, 1, f);

					double *src = (double *) frame_->dataR();

					sampleEMaccPosteriors(src, input_posterior);
				}
			}
		}
		else 
		{
			warning("MultiVariateNormalDistribution::EMaccPosteriors() : don't know how to deal with %d dimensions sorry :-(", input->nDimension());
			
			return false;
		}
	}
	
	return true;
}

}

