/**
 * @file cxx/trainer/src/TwoDPCATrainer.cc
 * @date Wed May 18 21:51:16 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "trainer/TwoDPCATrainer.h"
#include "math/eig.h"
#include "math/linear.h"

void bob::trainer::TwoDPCATrainer::train(bob::machine::TwoDPCAMachine& machine, const bob::io::Arrayset& data) 
{
  int n_samples = data.size();
  int m = data.getShape()[0];
  int n = data.getShape()[1];

  blitz::Array<double,2> mean(m,n);
  mean = 0.;
  // 1/ Compute the mean of the training data
  for( int i=0; i<n_samples; ++i)
    mean += data.get<double,2>(i);
  mean /= static_cast<double>(n_samples);

  // 2/ Generate the image covariance (scatter) matrix
  blitz::Array<double,2> G_mat(n,n);
  blitz::Array<double,2> tmp_i(n,n);
  G_mat = 0.;
  blitz::Array<double,2> sample_nomean(m,n);
  for( int i=0; i<n_samples; ++i)
  {
    sample_nomean = data.get<double,2>(i) - mean;
    blitz::Array<double,2> sample_nomean_t = sample_nomean.transpose(1,0);
    // Compute tmp_i=(Aj-Am)'*(Aj-Am)
    bob::math::prod(sample_nomean_t, sample_nomean, tmp_i);
    // Update G_mat
    G_mat = G_mat + tmp_i; 
  } 
  G_mat /= static_cast<double>(n_samples);

  // 3/ Compute the eigenvalue decomposition of G_mat
  blitz::Array<double,1> sigma(n);
  blitz::Array<double,2> V(n,n);
  bob::math::eigSymReal(G_mat, V, sigma);

  // 4/ Sort the eigenvalues/eigenvectors (no blitz++ way unfortunately)
  std::vector< std::pair<double,int> > eigenvalues_sort;
  for( int i=0; i<n; ++i)
    eigenvalues_sort.push_back( std::pair<double,int>(sigma(i),i) );
  std::sort(eigenvalues_sort.begin(), eigenvalues_sort.end());

  // 5/ Update the machine
  int n_outputs_set = machine.getNOutputs();
  if( n_outputs_set <=0 || n_outputs_set > n)
    n_outputs_set = n;
  blitz::Array<double,1> eigenvalues(n_outputs_set);
  blitz::Array<double,2> eigenvectors(n, n_outputs_set);
  for(int ind=0; ind<n_outputs_set; ++ind)
  {
    // Convert them to covariance matrix eigenvalues
    eigenvalues(n_outputs_set-ind-1) = eigenvalues_sort[ind].first * eigenvalues_sort[ind].first / (n_samples - 1);
    blitz::Array<double,1> vec = V(eigenvalues_sort[ind].second, blitz::Range::all());
    double norm = sqrt( blitz::sum(vec*vec) );
    blitz::Array<double,1> eigen_vec = eigenvectors(blitz::Range::all(), n_outputs_set-ind-1);
    eigen_vec = vec / norm;
  }
  machine.setDimOutputs(n);
  machine.setEigenvaluesvectors(eigenvalues,eigenvectors);
  machine.setPreMean(mean);
}

