#include <boost/python.hpp>
#include "database/Arrayset.h"
#include "trainer/SVDPCATrainer.h"
#include "trainer/FisherLDATrainer.h"

using namespace boost::python;
namespace db = Torch::database;
namespace mach = Torch::machine;
namespace train =Torch::trainer;


void bind_trainer_eigen() {

  class_<train::SVDPCATrainer, boost::noncopyable >("SVDPCATrainer", init<>())
    .def("train", &train::SVDPCATrainer::train, (arg("machine"), arg("data")), "Train a machine using some data")
  ;

  class_<train::FisherLDATrainer, boost::noncopyable >("FisherLDATrainer", init<int>())
    .def("train", &train::FisherLDATrainer::train, (arg("machine"), arg("data")), "Train a machine using some data")
  ;

}
