#include "trainer/KMeansTrainer.h"

#include <database/Arrayset.h>
#include <cfloat>
#include <core/logging.h>
#include <boost/random.hpp>

using namespace Torch::machine;

Torch::trainer::KMeansTrainer::KMeansTrainer(double convergence_threshold, int max_iterations) :
  EMTrainer<KMeansMachine, FrameSample>(convergence_threshold, max_iterations) {
  seed = -1;
}
  
void Torch::trainer::KMeansTrainer::initialization(KMeansMachine& kMeansMachine, const Sampler<FrameSample>& sampler) {
  // split data into as many chunks as there are means
  size_t n_data = sampler.getNSamples();
  unsigned int n_chunk = n_data / kMeansMachine.getNMeans();
  
  boost::mt19937 rng;
  if (seed != -1) {
    rng.seed((uint32_t)seed);
  }
  
  // assign the i'th mean to a random example within the i'th chunk
  for(int i = 0; i < kMeansMachine.getNMeans(); i++) {
    boost::uniform_int<> range(i*n_chunk,(i+1)*n_chunk);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(rng, range);
    
    // get random index within chunk
    unsigned int index = die();

    // get the example at that index
    FrameSample frame = sampler.getSample(index);
    
    const blitz::Array<double, 1>& mean = frame.getFrame();
    
    // set the mean
    kMeansMachine.setMean(i, mean);
  } 
}

double Torch::trainer::KMeansTrainer::eStep(KMeansMachine& kmeans, const Sampler<FrameSample>& data) {
    // initialise the accumulators
    double average_min_distance = 0;
    resetAccumulators(kmeans);

    // iterate over data samples
    for (int64_t i=0; i < data.getNSamples(); i++) {
      // get example
      blitz::Array<double, 1> x(data.getSample(i).getFrame());

      // find closest mean, and distance from that mean
      int closest_mean = -1;
      double min_distance = -1;
      kmeans.getClosestMean(x,closest_mean,min_distance);

      // accumulate the stats
      average_min_distance += min_distance;
      m_zeroethOrderStats(closest_mean)++;
      m_firstOrderStats(closest_mean,blitz::Range::all()) += x;
    }
    average_min_distance /= data.getNSamples();
    
    return average_min_distance;
}

void Torch::trainer::KMeansTrainer::mStep(KMeansMachine& kmeans, const Sampler<FrameSample>&) {
    blitz::Array<double,2> newMeans(kmeans.getNMeans(),kmeans.getNInputs());
    blitz::firstIndex i;
    blitz::secondIndex j;
    newMeans = m_firstOrderStats(i,j) / m_zeroethOrderStats(i);
    kmeans.setMeans(newMeans);
}

bool Torch::trainer::KMeansTrainer::resetAccumulators(KMeansMachine& kMeansMachine) {
  m_zeroethOrderStats.resize(kMeansMachine.getNMeans());
  m_zeroethOrderStats = 0;
  m_firstOrderStats.resize(kMeansMachine.getNMeans(), kMeansMachine.getNInputs());
  m_firstOrderStats = 0;
  return true;
}

void Torch::trainer::KMeansTrainer::setSeed(int seed) {
  this->seed = seed;
}

int Torch::trainer::KMeansTrainer::getSeed() {
  return seed;
}



