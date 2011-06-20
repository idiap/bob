/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 10 Jun 2011 16:01:23 CEST
 *
 * @brief Python bindings to LinearMachine trainers
 */

#include <boost/python.hpp>
#include "trainer/SVDPCATrainer.h"
#include "trainer/FisherLDATrainer.h"

using namespace boost::python;
namespace db = Torch::database;
namespace mach = Torch::machine;
namespace train = Torch::trainer;

tuple eig_train1 (const train::SVDPCATrainer& t, const db::Arrayset& data) {
  blitz::Array<double,1> eig_val(data.getShape()[0]);
  mach::LinearMachine m;
  t.train(m, eig_val, data);
  return make_tuple(m, eig_val);
}

object eig_train2 (const train::SVDPCATrainer& t, mach::LinearMachine& m,
    const db::Arrayset& data) {
  blitz::Array<double,1> eig_val(data.getShape()[0]);
  t.train(m, eig_val, data);
  return object(eig_val);
}

tuple lda_train1 (const train::FisherLDATrainer& t,
    const std::vector<db::Arrayset>& data) {
  blitz::Array<double,1> eig_val(data[0].getShape()[0]);
  mach::LinearMachine m;
  t.train(m, eig_val, data);
  return make_tuple(m, eig_val);
}

object lda_train2 (const train::FisherLDATrainer& t, mach::LinearMachine& m,
    const std::vector<db::Arrayset>& data) {
  blitz::Array<double,1> eig_val(data[0].getShape()[0]);
  t.train(m, eig_val, data);
  return object(eig_val);
}

void bind_trainer_linear() {

  class_<train::SVDPCATrainer>("SVDPCATrainer", "Sets a linear machine to perform the Karhunen-Loeve Transform (KLT) on a given dataset using Singular Value Decomposition (SVD). References:\n\n 1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n 2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n 3. http://en.wikipedia.org/wiki/Principal_component_analysis\n\nTests are executed against the Matlab printcomp output for correctness.", init<bool>((arg("zscore_convert")), "Initializes a new SVD/PCD trainer. The training stage will place the resulting principal components in the linear machine and set it up to extract the variable means automatically. As an option, you may preset the trainer so that the normalization performed by the resulting linear machine also divides the variables by the standard deviation of each variable ensemble.\n\nIf zscore_convert is set to 'true' set up the resulting linear machines to also perform zscore convertion. This will make the input data to be divided by the train data standard deviation after mean subtraction."))
    .def(init<>("Default constructor. This is equivalent to calling SVDPCATrainer(False)."))
    .def("train", &eig_train1, (arg("self"), arg("data")), "Trains a LinearMachine to perform the KLT. The resulting machine will have the eigen-vectors of the covariance matrix arranged by decreasing energy automatically. You don't need to sort the results. This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array.")
    .def("train", &eig_train2, (arg("self"), arg("machine"), arg("data")), "Trains the LinearMachine to perform the KLT. The resulting machine will have the eigen-vectors of the covariance matrix arranged by decreasing energy automatically. You don't need to sort the results. This method returns the eigen values in a 1D array.")
    ;

  class_<train::FisherLDATrainer>("FisherLDATrainer", "Implements a multi-class Fisher/LDA linear machine Training using Singular Value Decomposition (SVD). For more information on Linear Machines and associated methods, please consult Bishop, Machine Learning and Pattern Recognition chapter 4.", init<>())
    .def("train", &lda_train1, (arg("self"), arg("data")), "Creates a LinearMachine that performs Fisher/LDA discrimination. The resulting machine will have the eigen-vectors of the Sigma-1 * Sigma_b product, arranged by decreasing 'energy'. Each input arrayset represents data from a given input class. This method returns a tuple containing the resulting linear machine and the eigen values in a 1D array. This way you can reset the machine as you see fit.")
    .def("train", &lda_train2, (arg("self"), arg("machine"), arg("data")), "Trains a given LinearMachine to perform Fisher/LDA discrimination. After this method has been called, the input machine will have the eigen-vectors of the Sigma-1 * Sigma_b product, arranged by decreasing 'energy'. Each input arrayset represents data from a given input class. This method also returns the eigen values allowing you to implement your own compression scheme.")
  ;

}
