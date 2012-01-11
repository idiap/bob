/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Fri 29 Sep 2011
 *
 * @brief Python bindings to WienerTrainer
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "trainer/WienerTrainer.h"

using namespace boost::python;
namespace io = bob::io;
namespace mach = bob::machine;
namespace train = bob::trainer;

boost::shared_ptr<mach::WienerMachine> wiener_train1 (const train::WienerTrainer& t, const io::Arrayset& data) {
  boost::shared_ptr<mach::WienerMachine> m;
  t.train(*m, data);
  return m;
}

void wiener_train2 (const train::WienerTrainer& t, mach::WienerMachine& m,
    const io::Arrayset& data) {
  t.train(m, data);
}

void bind_trainer_wiener() {

  class_<train::WienerTrainer>("WienerTrainer", "Sets a Wiener machine and train it on a given dataset.", init<>("Initializes a new Wiener Trainer."))
    .def("train", &wiener_train1, (arg("self"), arg("data")), "Trains a WienerMachine to perform the filtering. This method returns a tuple containing the resulting Wiener machine in a 2D array.")
    .def("train", &wiener_train2, (arg("self"), arg("machine"), arg("data")), "Trains the provided WienerMachine.")
    ;

}
