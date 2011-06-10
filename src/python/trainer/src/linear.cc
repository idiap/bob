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

void bind_trainer_linear() {

  class_<train::SVDPCATrainer, boost::noncopyable>("SVDPCATrainer", "Sets a linear machine to perform the Karhunen-Loève Transform (KLT) on a given dataset using Singular Value Decomposition (SVD). References:\n\n 1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n 2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n 3. http://en.wikipedia.org/wiki/Principal_component_analysis\n\nTests are executed against the Matlab printcomp output for correctness.", init<bool>((arg("zscore_convert")), "Initializes a new SVD/PCD trainer. The training stage will place the resulting principal components in the linear machine and set it up to extract the variable means automatically. As an option, you may preset the trainer so that the normalization performed by the resulting linear machine also divides the variables by the standard deviation of each variable ensemble.\n\nIf zscore_convert is set to 'true' set up the resulting linear machines to also perform zscore convertion. This will make the input data to be divided by the train data standard deviation after mean subtraction."))
    .def(init<>("Default constructor. This is equivalent to calling SVDPCATrainer(False)."))
    .def("train", (void (train::SVDPCATrainer::*)(mach::LinearMachine&, const db::Arrayset&) const)&train::SVDPCATrainer::train, (arg("self"), arg("machine"), arg("data")), "Trains the LinearMachine to perform the KLT. The resulting machine will have the eigen-vectors of the covariance matrix arranged by decreasing energy automatically. You don't need to sort the results.")
    .def("train", (void (train::SVDPCATrainer::*)(mach::LinearMachine&, blitz::Array<double,1>&, const db::Arrayset&) const)&train::SVDPCATrainer::train, (arg("self"), arg("machine"), arg("eigen_values"), arg("data")), "Trains the LinearMachine to perform the KLT. The resulting machine will have the eigen-vectors of the covariance matrix arranged by decreasing energy automatically. You don't need to sort the results. Also returns the eigen values of the covariance matrix so you can use that to choose which components to keep.")
    ;

  class_<train::FisherLDATrainer, boost::noncopyable >("FisherLDATrainer", init<int>())
    .def("train", &train::FisherLDATrainer::train, (arg("machine"), arg("data")), "Train a machine using some data")
  ;

}
