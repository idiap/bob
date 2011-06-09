#include "trainer/SVDPCATrainer.h"
#include "math/svd.h"

void Torch::trainer::SVDPCATrainer::train(Torch::machine::EigenMachine& machine, const Torch::database::Arrayset& data) 
{
  int n_samples = data.getNSamples();
  int n_features = data.getShape()[0];

  std::vector<size_t> ids;
  data.index(ids);

  blitz::Array<double,1> mean(n_features);
  mean = 0.;
  // 1/ Compute the mean of the training data
  for( int i=0; i<n_samples; ++i)
    mean += data.get<double,1>(ids[i]);
  mean /= static_cast<double>(n_samples);

  // 2/ Generate the data matrix
  blitz::Array<double,2> data_mat(n_features,n_samples);
  for( int i=0; i<n_samples; ++i)
  {
    blitz::Array<double,1> data_col = data_mat(blitz::Range::all(), i);
    data_col = (data.get<double,1>(ids[i]) - mean);
  } 

  // 3/ Compute the singular value decomposition 
  blitz::Array<double,2> U(n_features,n_features);
  const int n_sigma = std::min(n_features,n_samples);
  blitz::Array<double,1> sigma(n_sigma);
  blitz::Array<double,2> V(n_samples,n_samples);
  Torch::math::svd(data_mat, U, sigma, V);

  // 4/ Sort the eigenvalues/eigenvectors (no blitz++ way unfortunately)
  std::vector< std::pair<double,int> > eigenvalues_sort;
  for( int i=0; i<n_sigma; ++i)
    eigenvalues_sort.push_back( std::pair<double,int>(sigma(i),i) );
  std::sort(eigenvalues_sort.begin(), eigenvalues_sort.end());

  // 5/ Update the machine
  int n_outputs_set = machine.getNOutputs();
  if( n_outputs_set <=0 || n_outputs_set > n_sigma)
    n_outputs_set = n_sigma;
  blitz::Array<double,1> eigenvalues(n_outputs_set);
  blitz::Array<double,2> eigenvectors(n_outputs_set,n_features);
  for(int ind=0; ind<n_outputs_set; ++ind)
  {
    // Convert them to covariance matrix eigenvalues
    eigenvalues(n_outputs_set-ind-1) = eigenvalues_sort[ind].first * eigenvalues_sort[ind].first / (n_samples - 1);
    blitz::Array<double,1> vec = U(eigenvalues_sort[ind].second, blitz::Range::all());
    double norm = sqrt( blitz::sum(vec*vec) );
    blitz::Array<double,1> eigen_vec = eigenvectors(n_outputs_set-ind-1,blitz::Range::all());
    eigen_vec = vec / norm;
  }
  machine.setEigenvaluesvectors(eigenvalues,eigenvectors);
  machine.setPreMean(mean);
}

