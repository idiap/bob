#include "FLDATrainer.h"
#include "FLDAMachine.h"
#include "mat.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

FLDATrainer::FLDATrainer()
	:	m_validation_dataset(0)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

FLDATrainer::~FLDATrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Set the validation dataset

bool FLDATrainer::setValidationData(DataSet* dataset)
{
	m_validation_dataset = dataset;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the given machine on the given dataset

bool FLDATrainer::train()
{
	// Check parameters
	if (	m_machine == 0 ||
		m_dataset == 0 ||
		m_dataset->getNoExamples() < 1 ||
		m_dataset->getExample(0)->getDatatype() != Tensor::Double ||
		m_dataset->getExample(0)->nDimension() != 1 ||
		m_dataset->getTarget(0)->getDatatype() != Tensor::Double ||
		m_dataset->getTarget(0)->nDimension() != 1)
	{
		print("FLDATrainer::train - invalid parameters!\n");
		return false;
	}
	FLDAMachine* flda_machine = dynamic_cast<FLDAMachine*>(m_machine);
	if (flda_machine == 0)
	{
		print("FLDATrainer::train - can only train FLDA machines!\n");
		return false;
	}

	// Allocate the between&within class covariance and averages
	const int size = m_dataset->getExample(0)->size(0);
	if (flda_machine->resize(size) == false)
	{
		print("FLDATrainer::train - could not resize FLDA machine!\n");
		return false;
	}

	double*	avg_pos = new double[size];
	double*	avg_neg = new double[size];
	double*	avg_all = new double[size];
	double** cov_between = new double*[size];
	double** cov_within = new double*[size];

	for (int i = 0; i < size; i ++)
	{
		avg_pos[i] =  avg_neg[i] = avg_all[i] = 0.0;

		cov_between[i] = new double[size];
		cov_within[i] = new double[size];

		for (int j = 0; j < size; j ++)
		{
			cov_between[i][j] = 0.0;
			cov_within[i][j] = 0.0;
		}
	}

	const long n_samples = m_dataset->getNoExamples();

	// Estimate the average within classes
	int n_pos = 0, n_neg = 0;
	for (long s = 0; s < n_samples; s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);
		const DoubleTensor* target = (const DoubleTensor*)m_dataset->getExample(s);

		if (target->get(0) < 0.0)
		{
			// negative sample
			for (int i = 0; i < size; i ++)
			{
				avg_neg[i] += example->get(i);
			}
			n_neg ++;
		}
		else
		{
			// positive sample
			for (int i = 0; i < size; i ++)
			{
				avg_pos[i] += example->get(i);
			}
			n_pos ++;
		}
	}

	const double inv_pos = n_pos == 0 ? 1.0 : 1.0 / n_pos;
	const double inv_neg = n_neg == 0 ? 1.0 : 1.0 / n_neg;
	for (int i = 0; i < size; i ++)
	{
		avg_pos[i] *= inv_pos;
		avg_neg[i] *= inv_neg;
	}

	// Estimate the average between classes
	const double inv_all = n_pos + n_neg == 0 ? 1.0 : 1.0 / (n_pos + n_neg);
	for (int i = 0; i < size; i ++)
	{
		avg_all[i] = inv_all * (avg_pos[i] * n_pos + avg_neg[i] * n_neg);
	}

	// Compute the between classes covariance matrix
	for (int i = 0; i < size; i ++)
		for (int j = 0; j < size; j ++)
		{
			cov_between[i][j] =	(avg_pos[i] - avg_all[i]) * (avg_pos[j] - avg_all[j]) * n_pos +
						(avg_neg[i] - avg_all[i]) * (avg_neg[j] - avg_all[j]) * n_neg;
		}

	// Compute the within classes covariance matrix
	for (long s = 0; s < n_samples; s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);
		const DoubleTensor* target = (const DoubleTensor*)m_dataset->getExample(s);

		if (target->get(0) < 0.0)
		{
			// negative sample
			for (int i = 0; i < size; i ++)
				for (int j = 0; j < size; j ++)
				{
					cov_within[i][j] += 	(example->get(i) - avg_neg[i]) *
								(example->get(j) - avg_neg[j]);
				}
		}
		else
		{
			// positive sample
			for (int i = 0; i < size; i ++)
				for (int j = 0; j < size; j ++)
				{
					cov_within[i][j] += 	(example->get(i) - avg_pos[i]) *
								(example->get(j) - avg_pos[j]);
				}
		}
	}

	// Compute Sb^1/2
	double** sqrt_Sb = new double*[size];
	for (int i = 0; i < size; i ++)
	{
		sqrt_Sb[i] = new double[size];
	}
	mat_sqrt_sym(cov_between, sqrt_Sb, size);

	// Compute the inverse of the within class covariance matrix
	double** inv_Sw = new double*[size];
	for (int i = 0; i < size; i ++)
	{
		inv_Sw[i] = new double[size];
	}
	mat_invert(cov_within, inv_Sw, size);

	// Compute Sb^1/2 * SW-1 * SW^1/2
	double** final_mat = new double*[size];
	for (int i = 0; i < size; i ++)
	{
		final_mat[i] = new double[size];
		for (int j = 0; j < size; j ++)
		{
			double sum = 0.0;
			for (int k = 0; k < size; k ++)
				for (int l = 0; l < size; l ++)
				{
					sum += sqrt_Sb[i][k] * inv_Sw[k][l] * sqrt_Sb[l][j];
				}

			final_mat[i][j] = sum;
		}
	}

	// Get the largest eigen vector of the matrix product -> it will be the projection
	double* proj = new double[size];
	mat_eigen_sym(final_mat, inv_Sw, proj, size);
	for (int i = 0; i < size; i ++)
	{
		proj[i] = inv_Sw[i][size - 1];
	}

	// Project the between class average
	double proj_avg = 0.0;
	for (int i = 0; i < size; i ++)
	{
		proj_avg += proj[i] * avg_all[i];
	}

	// Set the parameters to the machine
	flda_machine->setThreshold(0.0);
	flda_machine->setProjection(proj, proj_avg);

	// Cleanup
	delete[] avg_pos;
	delete[] avg_neg;
	delete[] avg_all;
	delete[] proj;
	for (int i = 0; i < size; i ++)
	{
		delete[] cov_between[i];
		delete[] cov_within[i];
		delete[] sqrt_Sb[i];
		delete[] inv_Sw[i];
		delete[] final_mat[i];
	}
	delete[] cov_between;
	delete[] cov_within;
	delete[] sqrt_Sb;
	delete[] inv_Sw;
	delete[] final_mat;

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
