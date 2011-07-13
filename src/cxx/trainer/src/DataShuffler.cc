/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 13 Jul 2011 12:48:56 CEST
 *
 * @brief Implementation of the DataShuffler.
 */

#include "core/array_assert.h"
#include "trainer/Exception.h"
#include "trainer/DataShuffler.h"

namespace array = Torch::core::array;
namespace train = Torch::trainer;

train::DataShuffler::DataShuffler(const std::vector<Torch::io::Arrayset>& data,
    const std::vector<blitz::Array<double,1> >& target):
  m_data(data),
  m_target(target.size()),
  m_selector(data.size()),
  m_do_stdnorm(false),
  m_mean(),
  m_stddev()
{
  if (data.size() == 0) throw train::WrongNumberOfClasses(0);
  if (target.size() == 0) throw train::WrongNumberOfClasses(0);
  
  array::assertSameDimensionLength(data.size(), target.size());
  
  // checks shapes, minimum number of examples
  for (size_t k=0; k<data.size(); ++k) {
    if (data[k].size() == 0) throw WrongNumberOfFeatures(0, 1, k);
    //this may also trigger if I cannot get doubles from the Arrayset
    array::assertSameShape(data[0].get<double,1>(0), data[k].get<double,1>(0));
    array::assertSameShape(target[0], target[k]);
  }

  // set save values for the mean and stddev (even if not used at start)
  m_mean.resize(data[0].getShape()[0]);
  m_mean = 0.;
  m_stddev.resize(data[0].getShape()[0]);
  m_stddev = 1.;

  // copies the target data to my own variable
  for (size_t k=0; k<target.size(); ++k) 
    m_target[k].reference(target[k].copy());

  // creates one generator tailored for the range of each Arrayset
  boost::mt19937 rng;
  for (size_t i=0; i<m_selector.size(); ++i) {
    boost::uniform_int<size_t> range(0, m_data[i].size()-1);
    m_selector[i].reset(new 
        boost::variate_generator<boost::mt19937&, boost::uniform_int<size_t> >
        (rng, range));
  }
}

train::DataShuffler::DataShuffler(const train::DataShuffler& other):
  m_data(other.m_data),
  m_target(other.m_target.size()),
  m_selector(other.m_data.size()),
  m_do_stdnorm(other.m_do_stdnorm),
  m_mean(other.m_mean.copy()),
  m_stddev(other.m_stddev.copy())
{
  for (size_t k=0; k<m_target.size(); ++k) 
    m_target[k].reference(other.m_target[k].copy());

  // One generator tailored for the range of each Arrayset
  for (size_t i=0; i<m_selector.size(); ++i) {
    m_selector[i].reset(new 
        boost::variate_generator<boost::mt19937&, boost::uniform_int<size_t> >
        (other.m_selector[i]->engine(), other.m_selector[i]->distribution()));
  }
}

train::DataShuffler::~DataShuffler() { }

train::DataShuffler& train::DataShuffler::operator= 
(const train::DataShuffler::DataShuffler& other) {
  m_data = other.m_data;
  m_target.resize(other.m_target.size());
  for (size_t k=0; k<m_target.size(); ++k) 
    m_target[k].reference(other.m_target[k].copy());
 
  m_selector.resize(other.m_selector.size());
  for (size_t i=0; i<m_selector.size(); ++i) {
    m_selector[i].reset(new 
        boost::variate_generator<boost::mt19937&, boost::uniform_int<size_t> >
        (other.m_selector[i]->engine(), other.m_selector[i]->distribution()));
  }
  
  m_mean.reference(other.m_mean.copy());
  m_stddev.reference(other.m_stddev.copy());

  return *this;
}

void train::DataShuffler::setSeed(size_t s) {
  for (size_t i=0; i<m_selector.size(); ++i) {
    m_selector[i]->engine().seed(s++); //don't use the same seed for all dist.
    m_selector[i]->distribution().reset(); //needs to reset distribution
  }
}

/**
 * Calculates mean and std.dev. in a single loop.
 * see: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 */
void evaluateStdNormParameters(const std::vector<Torch::io::Arrayset>& data,
    blitz::Array<double,1>& mean, blitz::Array<double,1>& stddev) {
  
  mean = 0.;
  stddev = 0.; ///< temporarily used to accumulate square sum!
  double samples = 0;

  for (size_t k=0; k<data.size(); ++k) {
    for (size_t i=0; i<data[k].size(); ++i) {
      mean += data[k].get<double,1>(i);
      stddev += blitz::pow2(data[k].get<double,1>(i));
      ++samples;
    }
  }
  stddev -= blitz::pow2(mean);
  stddev /= (samples-1); ///< note: unbiased sample variance
  stddev = blitz::sqrt(stddev);

  mean /= (samples);
}

void train::DataShuffler::setAutoStdNorm(bool s) {
  if (s) evaluateStdNormParameters(m_data, m_mean, m_stddev);
  else {
    //reset mean and std.dev. values (just in case)
    m_mean = 0.;
    m_stddev = 1.;
  }
  m_do_stdnorm = s;
}

void train::DataShuffler::getStdNorm(blitz::Array<double,1>& mean,
    blitz::Array<double,1>& stddev) const {
  if (m_do_stdnorm) {
    mean.reference(m_mean.copy());
    stddev.reference(m_stddev.copy());
  }
  else { //do it on-the-fly
    evaluateStdNormParameters(m_data, mean, stddev);
  }
}

void train::DataShuffler::operator() (blitz::Array<double,2>& data,
    blitz::Array<double,2>& target) const {
  
  array::assertSameDimensionLength(data.extent(0), target.extent(0));

  size_t counter = data.extent(0);
  blitz::Range all = blitz::Range::all();
  while (true) {
    for (size_t i=0; i<m_data.size(); ++i) { //for all classes
      size_t index = (*m_selector[i])(); //pick a random position within class
      if (m_do_stdnorm)
        data(counter, all) = (m_data[i].get<double,1>(index) - m_mean)/m_stddev;
      else
        data(counter, all) = m_data[i].get<double,1>(index);
      target(counter, all) = m_target[i];
      --counter;
      if (counter == 0) break;
    }
    if (counter == 0) break;
  }

}
