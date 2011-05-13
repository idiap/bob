#include "trainer/SVDPCATrainer.h"

void Torch::trainer::SVDPCATrainer::train(Torch::machine::EigenMachine& machine, const Sampler<Torch::machine::FrameSample>& data) 
{
  Torch::core::info << "# SVDPCATrainer:" << std::endl;

  int n_samples = data.getNSamples();
  int n_features = data.getSample(0).getFrame().extent(0);

  blitz::Array<double,1> mean(n_features);
  mean = 0.;
  // 1/ Compute the mean of the training data
  for( int i=0; i<n_samples; ++i)
    mean += data.getSample(i).getFrame();
  mean /= static_cast<double>(n_samples);

  // 2/ Generate the data matrix
  blitz::Array<double,2> data_mat(n_features,n_samples);
  double norm_factor = 1. / sqrt(n_samples - 1);
  for( int i=0; i<n_samples; ++i)
  {
    blitz::Array<double,1> data_col = data_mat(blitz::Range::all(), i);
    data_col = (data.getSample(i).getFrame() - mean) / norm_factor;
  } 

  // 3/ Compute the singular value decomposition 
  blitz::Array<double,2> U(n_features,n_features);
  const int n_sigma = std::min(n_features,n_samples);
  blitz::Array<double,1> sigma(n_sigma);
  blitz::Array<double,2> V(n_samples,n_samples);

  // 4/ Sort the eigenvalues/eigenvectors (no blitz++ way unfortunately)
  std::vector< std::pair<double,int> > eigenvalues_sort;
  for( int i=0; i<n_sigma; ++i)
    eigenvalues_sort.push_back( std::pair<double,int>(sigma(0),i) );
  std::sort(eigenvalues_sort.begin(), eigenvalues_sort.end());

  // 5/ Update the machine
  blitz::Array<double,1> eigenvalues(machine.getNOutputs());
  blitz::Array<double,2> eigenvectors(machine.getNOutputs(),n_features);
  for(int ind=0; ind<machine.getNOutputs(); ++ind)
  {
    eigenvalues(ind) = eigenvalues_sort[ind].first;
    blitz::Array<double,1> vec = U(eigenvalues_sort[ind].second, blitz::Range::all());
    double norm = sqrt( blitz::sum(vec*vec) );
    blitz::Array<double,1> eigen_vec = eigenvectors(ind,blitz::Range::all());
    eigen_vec = vec / norm;
  }
  machine.setEigenvaluesvectors(eigenvalues,eigenvectors);
  machine.setPreMean(mean);
}

