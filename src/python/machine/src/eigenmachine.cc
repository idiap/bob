#include <boost/python.hpp>
#include <machine/EigenMachine.h>

using namespace boost::python;
using namespace Torch::machine;


class Machine_FrameSample_A1double_Wrapper : public Machine<FrameSample, blitz::Array<double,1> >, public wrapper<Machine<FrameSample, blitz::Array<double,1> > > {
public:
  blitz::Array<double,1> forward (const FrameSample& input) const {
    return this->get_override("forward")(input);
  }
};


void bind_machine_eigenmachine()
{
  class_<Machine_FrameSample_A1double_Wrapper, boost::noncopyable>("Machine_FrameSample_A1double_")
  .def("forward", &Machine<FrameSample, blitz::Array<double,1> >::forward, args("input"))
  ;
  
  // TODO: add constructor variants, get/set: functions or properties?
  class_<EigenMachine, bases<Machine<FrameSample, blitz::Array<double,1> > > >("EigenMachine", init<>())
  .def("getNOutputs", &EigenMachine::getNOutputs)
  .def("setNOutputs", &EigenMachine::setNOutputs, args("n_outputs"))
  .def("getPVariance", &EigenMachine::getPVariance)
  .def("setPVariance", &EigenMachine::setPVariance, args("p_variance"))
  .def("getEigenvalues", make_function(&EigenMachine::getEigenvalues, return_value_policy<copy_const_reference>()))
  .def("getEigenvectors", make_function(&EigenMachine::getEigenvectors, return_value_policy<copy_const_reference>()))
  .def("setEigenvaluesvectors", &EigenMachine::setEigenvaluesvectors, args("eigenvalues","eigenvectors"))
  .def("getPreMean", make_function(&EigenMachine::getPreMean, return_value_policy<copy_const_reference>()))
  .def("setPreMean", &EigenMachine::setPreMean, args("pre_mean"))
  .def("forward", &EigenMachine::forward, args("input"))
  ;
}
