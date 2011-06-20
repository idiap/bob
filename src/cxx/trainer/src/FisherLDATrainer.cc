/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jun 09:33:28 2011 CEST
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
 */

#include "core/blitz_compat.h"
#include "math/eig.h"
#include "trainer/Exception.h"
#include "trainer/FisherLDATrainer.h"

namespace train = Torch::trainer;
namespace mach = Torch::machine;
namespace db = Torch::database;

train::FisherLDATrainer::FisherLDATrainer() { }

train::FisherLDATrainer::FisherLDATrainer(const train::FisherLDATrainer& other)
{ }

train::FisherLDATrainer::~FisherLDATrainer() {}

train::FisherLDATrainer& train::FisherLDATrainer::operator=
(const train::FisherLDATrainer& other) {
  return *this;
}

/**
 * Calculates the within and between class covariance matrix Sw. Returns that
 * matrix and the class means matrix: each column represents the means of one
 * of the classes.
 *
 * The input data matrices should be preloaded in column-wise order: Each
 * feature is encoded in a separate row. Each column represents a different
 * sample.
 */
static void evalCorrelations (const std::vector<blitz::Array<double,2> >& data,
    blitz::Array<double,2>& Sw, blitz::Array<double,2>& Sb,
    blitz::Array<double,1>& preMean) {
  
  // checks for data shape should have been done before...
  int n_features = data[0].extent(0);

  blitz::Array<double,2> classMeans(n_features, data.size());

  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range a = blitz::Range::all();
  blitz::Array<double,1> counts(data.size());
  counts = 0;
  
  // within class correlations
  blitz::Array<double,1> buffer(n_features);
  for (size_t cl=0; cl<data.size(); ++cl) { //class loop
    counts = data[cl].extent(1);
    classMeans(a,cl) = blitz::mean(data[cl],j);
    for (blitz::sizeType z=0; z<data[cl].extent(1); ++z) { //examples 
      buffer = data[cl](a,z) - classMeans(a,cl);
      Sw += buffer(i) * buffer(j); //outer product
    }
  }

  // between class correlations
  // Note: The global mean is calculated according to formula 4.44 on Bishop's
  // book: the global mean is the mean of all samples.
  preMean = blitz::mean(classMeans(i,j)*counts(j),j);
  for (size_t cl=0; cl<data.size(); ++cl) { //class loop
    buffer = classMeans(a,cl) - preMean;
    Sb += buffer(i) * buffer(j);
  }
  Sb /= data.size();
}

void train::FisherLDATrainer::train(mach::LinearMachine& machine,
    blitz::Array<double,1>& eigen_values,
    const std::vector<Torch::database::Arrayset>& data) const {

  // if #classes < 2, then throw
  if (data.size() < 2) throw train::WrongNumberOfClasses(data.size());

  // checks for arrayset data type and shape once
  int n_features = data[0].getShape()[0];

  for (size_t cl=0; cl<data.size(); ++cl) {
    if (data[cl].getElementType() != Torch::core::array::t_float64) {
      throw db::TypeError(data[cl].getElementType(),
          Torch::core::array::t_float64);
    }
    if (data[cl].getNDim() != 1) {
      throw Torch::database::DimensionError(data[cl].getNDim(), 1);
    }
    if (data[cl].getShape()[0] != (size_t)n_features) {
      throw Torch::trainer::WrongNumberOfFeatures(data[cl].getShape()[0],
          n_features, cl);
    }
  }

  // preloads the data (can be removed or re-implemented later if have very big
  // datasets).
  std::vector<blitz::Array<double,2> > preloadedData;
  blitz::Range a = blitz::Range::all();
  for (size_t cl=0; cl<data.size(); ++cl) {
    //a trick to use std::vectors and avoid copy
    blitz::Array<double,2> tmp;
    preloadedData.push_back(tmp);
    preloadedData[cl].reference(tmp);
    tmp.resize(n_features, data[cl].getNSamples());
    //now preload
    std::vector<size_t> index;
    data[cl].index(index);
    for (size_t s=0; s<index.size(); ++s) {
      tmp(a,s) = data[cl].get<double,1>(index[s]); //loads the data
    }
  }

  blitz::Array<double,2> Sw(n_features, n_features);
  Sw = 0;
  blitz::Array<double,2> Sb(n_features, n_features);
  Sb = 0;
  blitz::Array<double,1> preMean(n_features);
  preMean = 0;
  evalCorrelations(preloadedData, Sw, Sb, preMean);

  // computes the generalized eigenvalue decomposition 
  // so to find the eigen vectors/values of Sw^(-1) * Sb
  blitz::Array<double,2> V(n_features, n_features);
  eigen_values.resize(n_features);
  eigen_values = 0;
  Torch::math::eig(Sb, Sw, V, eigen_values);

  // updates the machine
  V.transposeSelf(1,0);
  V.resizeAndPreserve(V.extent(0), V.extent(1)-1);
  machine.setWeights(V);
  machine.setInputSubtraction(preMean);
}

void train::FisherLDATrainer::train(mach::LinearMachine& machine,
    const std::vector<Torch::database::Arrayset>& data) const {
  blitz::Array<double,1> throw_away;
  train(machine, throw_away, data);
}
