#include "trainer/FisherLDATrainer.h"
#include "math/linear.h"
#include "math/eig.h"

void Torch::trainer::FisherLDATrainer::computeScatterMatrices( const Sampler<Torch::machine::FrameClassificationSample>& data) 
{
  int n_samples = data.getNSamples();

  // Get dimension of first sample
  int n_features = data.getSample(0).getFrameSize();

  // Declare/initialize mean vectors
  blitz::Array<double,1> full_mean(n_features);
  full_mean = 0.;
  blitz::Array<double,2> class_mean(m_n_classes,n_features);
  class_mean = 0.;
  blitz::Array<int,1> class_n_samples(m_n_classes);
  class_n_samples = 0;

  // Compute the mean vectors
  for(int i=0; i<n_samples; ++i)
  {
    Torch::machine::FrameClassificationSample sample = data.getSample(i);
    blitz::Array<double,1> sample_data = sample.getFrame();
    int sample_class = sample.getTarget();
    
    // TODO: Should we perform these checks?
    if( n_features != sample_data.extent(0) )
      throw Torch::core::Exception();
    if( m_n_classes <= sample_class )
      throw Torch::core::Exception();

    // Add to mean vectors
    full_mean += sample_data;
    blitz::Array<double,1> current_class_mean = class_mean(sample_class, blitz::Range::all() );
    current_class_mean += sample_data;

    // Increment the number of sample of the current sample class
    class_n_samples(sample_class) += 1;
  }
  // Normalize with the numbe of samples
  full_mean /= n_samples;
  for(int i=0; i<m_n_classes; ++i)
  {
    blitz::Array<double,1> current_class_mean = class_mean(i, blitz::Range::all() );
    current_class_mean /= class_n_samples(i);
  }
  
  // Declare and allocate a temporary vector and matrix
  blitz::Array<double,1> tmp_vec(n_features);
  blitz::Array<double,2> tmp_mat(n_features,n_features);


  // Compute the between-class scatter matrix
  m_Sb.resize(n_features,n_features);
  m_Sb = 0.;
  for(int i=0; i<m_n_classes; ++i)
  {
    blitz::Array<double,1> current_class_mean = class_mean(i, blitz::Range::all() );
    tmp_vec = current_class_mean - full_mean;
    Torch::math::prod( tmp_vec, tmp_vec, tmp_mat);

    m_Sb += static_cast<double>(class_n_samples(i)) * tmp_mat;
  }

  // Compute the within-class scatter matrix
  m_Sw.resize(n_features,n_features);
  m_Sw = 0.;
  for(int i=0; i<n_samples; ++i)
  {
    Torch::machine::FrameClassificationSample sample = data.getSample(i);
    blitz::Array<double,1> sample_data = sample.getFrame();
    int sample_class = sample.getTarget(); 
   
    blitz::Array<double,1> current_class_mean = class_mean(sample_class, blitz::Range::all() ); 
    tmp_vec = sample_data - current_class_mean;
    Torch::math::prod( tmp_vec, tmp_vec, tmp_mat);

    m_Sw += static_cast<double>(class_n_samples(i)) * tmp_mat;
  }
}

void Torch::trainer::FisherLDATrainer::train(Torch::machine::EigenMachine& machine, const Sampler<Torch::machine::FrameClassificationSample>& data) 
{
  // 1/ Compute the scatter matrices
  computeScatterMatrices(data);

  // 2/ Compute the generalized eigenvalue decomposition 
  int n_features = data.getSample(0).getFrame().extent(0);
  blitz::Array<double,2> V(n_features,n_features);
  blitz::Array<double,1> sigma(n_features);
  Torch::math::eig(m_Sb, m_Sw, V, sigma);

  // 3/ Sort the eigenvalues/eigenvectors (no blitz++ way unfortunately)
  std::vector< std::pair<double,int> > eigenvalues_sort;
  for( int i=0; i<n_features; ++i)
    eigenvalues_sort.push_back( std::pair<double,int>(sigma(i),i) );
  std::sort(eigenvalues_sort.begin(), eigenvalues_sort.end());

  // 4/ Update the machine
  int n_outputs_set = machine.getNOutputs();
  if( n_outputs_set <=0 || n_outputs_set > n_features)
    n_outputs_set = n_features;
  blitz::Array<double,1> eigenvalues(n_outputs_set);
  blitz::Array<double,2> eigenvectors(n_outputs_set,n_features);
  for(int ind=0; ind<n_outputs_set; ++ind)
  {
    eigenvalues(n_outputs_set-ind-1) = eigenvalues_sort[ind].first;
    // Put a normalized eigenvector into the projection matrix eigen_vec
    blitz::Array<double,1> vec = V(blitz::Range::all(), eigenvalues_sort[ind].second);
    double norm = sqrt( blitz::sum(vec*vec) );
    blitz::Array<double,1> eigen_vec = eigenvectors(n_outputs_set-ind-1,blitz::Range::all());
    eigen_vec = vec / norm;
  }
  machine.setEigenvaluesvectors(eigenvalues,eigenvectors);
  blitz::Array<double,1> pre_mean(n_features);
  pre_mean = 0.;
  machine.setPreMean(pre_mean);
}

