/**
 * @file cxx/old/trainer/src/SVDPCATrainer.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "trainer/SVDPCATrainer.h"
#include "core/File.h"

// Declaration of the external Fortran library (Lapack). 
// This function performs a Single Value Decomposition
extern "C" void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
		double *s, double *u, int *ldu, double *vt, int *ldvt, double *work,
		int *lwork, int *info, short, short);

namespace Torch {

SVDPCATrainer::SVDPCATrainer() : Trainer()
{
}

bool SVDPCATrainer::train()
{
  const bool verbose = getBOption("verbose");

	print("SVDPCATrainer::train() ...\n");

	if (m_machine == NULL)
 	{
		warning("SVDPCATrainer::train() no machine.");
		return false;
	}

	if (m_dataset == NULL)
	{
		warning("SVDPCATrainer::train() no dataset.");
		return false;
	}

  long m_n_examples = m_dataset->getNoExamples();
	if (m_n_examples < 2)
	{
		warning("SVDPCATrainer::train() not enough examples in the dataset.");
		return false;
	}

	// testing at least if the dataset has targets
	if (m_dataset->hasTargets() == true)
		warning("SVDPCATrainer::train() dataset has targets but they will be ignore.");

	PCAMachine *pca_machine = (PCAMachine*) m_machine;

	//
	int n_inputs = pca_machine->n_inputs;

	// TODO: It would be good to be able to modify the MemoryDataset directly
	double* NormalizedMat = new double[ m_n_examples* n_inputs ];

	if(m_n_examples < n_inputs)
		warning("SVDPCATrainer::train() Just to let you know that n_examples (%d) < n_inputs (%d).", m_n_examples, n_inputs);

	// Computes the mean of the training data
	for(int i = 0 ; i < n_inputs ; ++i)
		pca_machine->Xm[i]=0.;

	for (long i=0 ; i<m_n_examples ; ++i)
	{
		DoubleTensor *example = (DoubleTensor *) m_dataset->getExample(i);
		if( example->size(0) != n_inputs) 
		{
			warning("SVDPCATrainer::train() size of sample %d and corresponding value in the machine do not match: %d vs %d ", i, example->size(i) , n_inputs);
			return false;	
		}
		for(int j=0 ; j < n_inputs ; j++ )
		{
//			print("%f ", example->get(j));
			pca_machine->Xm[j] += example->get(j);
		}
//	print("\n");
	}

	for(int j=0 ; j < n_inputs ; j++ )
		pca_machine->Xm[j] /= m_n_examples;

  if( verbose )
  {
    message("Mean");
	  for(int j=0 ; j < n_inputs ; j++ )
 		  print("%f ", pca_machine->Xm[j] );
  	print("\n");
  }


	// Normalizing the data matrix
	double z_ = 1.0 / sqrt((float) (m_n_examples - 1));
	for(int i = 0 ; i < m_n_examples ; i++)
	{
		DoubleTensor *example = (DoubleTensor *) m_dataset->getExample(i);
		for(int j = 0 ; j < n_inputs ; j++)
		{
			// Column-major order for Fortran
			NormalizedMat[ i*n_inputs + j] = (example->get(j) - pca_machine->Xm[j]) / z_; 
		}
	}


	// Prepare to use LAPACK function and perform singular value decomposition
  char jobu, jobvt;
  int lda, ldu, ldvt, lwork, info;
  double *S, *u, *vt, *work;

  int minmn, maxmn;

	int m = n_inputs;
	int n = (int)m_n_examples;

  jobu = 'S';
  jobvt = 'S';

  lda = m;
  ldu = m;

  if (m>=n) { minmn = n; maxmn = m; } else { minmn = m; maxmn = n; }

	S = new double[minmn]; // Array which will contain the singular values

	ldu = n;	
	u = new double[ldu * minmn]; // Left singular vector matrix

	ldvt = minmn; 
	vt = new double[ldvt * n]; // Right singular vector matrix 

	lwork = 5*maxmn; // Set up the work array, larger than needed.
  work = new double[lwork];

	// Call Lapack Single Value Decomposition function
  dgesvd_(&jobu, &jobvt, &m, &n, NormalizedMat, &lda, S, u, &ldu, vt, &ldvt, work, &lwork, &info, 1, 1);

	if (info != 0) error("SVDPCATraining: failure with error %d when performing Single Value Decomposition\n", info);

	// Infer eingenvalues and eigenvectors
	double* eigen_val = new double[minmn];
	for (int i=0; i<minmn; i++)
		eigen_val[i] = S[i]*S[i];

	double* eigen_vec = new double[ldu * minmn];
	for (int i=0; i<ldu; i++)
		for (int j=0; j<minmn; j++) 
			eigen_vec[i*ldu+j] = u[i*ldu+j];

	// Indexing eigenvalues and eigenvectors
	int_double *index_eigenvalues = new int_double[n_inputs];

	index_eigenvalues[0].the_double = eigen_val[0];
	index_eigenvalues[0].the_int = 0;
	
	bool need_to_sort = false;
	
	for(int i = 1 ; i < n_inputs ; i++)
	{
		index_eigenvalues[i].the_double = eigen_val[i];
		index_eigenvalues[i].the_int = i;
		
		if(index_eigenvalues[i].the_double > index_eigenvalues[i-1].the_double) need_to_sort = true;
	}

	if(need_to_sort)
	{
		// Sort eigenvalues
		qsort(index_eigenvalues, n_inputs, sizeof(int_double), sort_dec_int_double);
	}

	for(int i = 0 ; i < n_inputs ; i++)
	{
  	int index_ = index_eigenvalues[i].the_int;
	
//		if((verbose_level >= 3) && (need_to_sort))
//			printf(" Swap %d <- %d\n", i, index_eigenvalues[i].the_int);
		
		//
		pca_machine->eigenvalues[index_] = index_eigenvalues[i].the_double;
		
		//
		double sum = 0.0;
		
  	for(int j = 0 ; j < n_inputs ; j++)
		{
 			double z = eigen_vec[ i*ldvt + j ];
			// Convert column major order matrix (Fortran/Lapack) to row major order matrix for the PCA_MACHINE
			pca_machine->eigenvectors[ index_ + j * n_inputs] = z;

			sum += z * z;
		}

		double len = sqrt(sum);

		if(!IS_NEAR(len, 1.0, 0.01))
		{
			warning("Eigenvector is not unit (length=%g).", len);

      //
			for(int j = 0 ; j < n_inputs ; j++)
		   		pca_machine->eigenvectors[ index_ + j * n_inputs] = pca_machine->eigenvectors[ index_ + j * n_inputs] / len;
		}
	}

  if(verbose)
  {
  	// Display eigenvalues in decreasing order
	  message("Eigenvalues in decreasing order");
  	for( int j = 0 ; j < minmn ; j++ )
	  	print("%f ",  index_eigenvalues[j].the_double );
  	print("\n");

	  // Display eigenvectors
  	message("Eigenvectors");
	  for( int i = 0 ; i < n_inputs ; i++ )
	  {
		  for(int j = 0; j < n_inputs; j++)	
			  print("%f ",  pca_machine->eigenvectors[ i + j * n_inputs] );
  		print("\n");
	  }
  	print("\n");
  }

	// Clean up
	delete [] index_eigenvalues;
	delete [] eigen_val;
	delete [] eigen_vec;
	delete [] work;
	delete [] S;
	delete [] u;
	delete [] vt;
	delete [] NormalizedMat;
	
//	delete CovMat;

	return true;
}

SVDPCATrainer::~SVDPCATrainer()
{
}

}

