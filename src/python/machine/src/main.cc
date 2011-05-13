#include <boost/python.hpp>
#include <database/Arrayset.h>
#include <machine/KMeansMachine.h>
#include <machine/GMMMachine.h>
#include <machine/EigenMachine.h>

using namespace boost::python;
using namespace Torch::machine;


class Machine_FrameSample_double_Wrapper : public Machine<FrameSample, double>, public wrapper<Machine<FrameSample, double> > {
public:
  double forward (const FrameSample& input) const {
    return this->get_override("forward")(input);
  }
};

class Machine_FrameSample_A1double_Wrapper : public Machine<FrameSample, blitz::Array<double,1> >, public wrapper<Machine<FrameSample, blitz::Array<double,1> > > {
public:
  blitz::Array<double,1> forward (const FrameSample& input) const {
    return this->get_override("forward")(input);
  }
};


static tuple getVariancesAndWeightsForEachCluster(const KMeansMachine& machine, Torch::trainer::Sampler<FrameSample>& sampler) {
  boost::shared_ptr<blitz::Array<double, 2> > variances(new blitz::Array<double, 2>);
  boost::shared_ptr<blitz::Array<double, 1> > weights(new blitz::Array<double, 1>);
  machine.getVariancesAndWeightsForEachCluster(sampler, *variances.get(), *weights.get());
  return boost::python::make_tuple(variances, weights);
}

static boost::shared_ptr<blitz::Array<double, 1> > Gaussian_getMean(const Gaussian& gaussian) {
  boost::shared_ptr<blitz::Array<double, 1> > mean(new blitz::Array<double, 1>);
  gaussian.getMean(*mean.get());
  return mean;
}

static boost::shared_ptr<blitz::Array<double, 1> > Gaussian_getVariance(const Gaussian& gaussian) {
  boost::shared_ptr<blitz::Array<double, 1> > variance(new blitz::Array<double, 1>);
  gaussian.getVariance(*variance.get());
  return variance;
}

static boost::shared_ptr<blitz::Array<double, 1> > Gaussian_getVarianceThresholds(const Gaussian& gaussian) {
  boost::shared_ptr<blitz::Array<double, 1> > varianceThresholds(new blitz::Array<double, 1>);
  gaussian.getVarianceThresholds(*varianceThresholds.get());
  return varianceThresholds;
}

static boost::shared_ptr<blitz::Array<double, 1> > KMeansMachine_getMean(const KMeansMachine& kMeansMachine, int i) {
  boost::shared_ptr<blitz::Array<double, 1> > mean(new blitz::Array<double, 1>);
  kMeansMachine.getMean(i, *mean.get());
  return mean;
}

BOOST_PYTHON_MODULE(libpytorch_machine) {
  
  class_<FrameSample>("FrameSample", init<const blitz::Array<float, 1>& >())
  .def("getFrame", &FrameSample::getFrame, return_value_policy<copy_const_reference>())
  ;
  
  class_<Machine_FrameSample_double_Wrapper, boost::noncopyable>("Machine_FrameSample_double_")
  .def("forward", &Machine<FrameSample, double>::forward, args("input"))
  ;
  
  class_<Machine_FrameSample_A1double_Wrapper, boost::noncopyable>("Machine_FrameSample_A1double_")
  .def("forward", &Machine<FrameSample, blitz::Array<double,1> >::forward, args("input"))
  ;
  
  class_<KMeansMachine, bases<Machine<FrameSample, double> > >("KMeansMachine", init<int, int>())
  .add_property("means", (blitz::Array<double,2> (KMeansMachine::*)() const)&KMeansMachine::getMeans, &KMeansMachine::setMeans)
  .def("getMean", &KMeansMachine::getMean, args("i", "mean"))
  .def("getMean", &KMeansMachine_getMean, args("i", "mean"))
  .def("setMean", &KMeansMachine::setMean, args("i" "mean"))
  .def("getDistanceFromMean", &KMeansMachine::getDistanceFromMean, args("x", "i"))
  .def("getClosestMean", &KMeansMachine::getClosestMean, args("x", "closest_mean", "min_distance"))
  .def("getMinDistance", &KMeansMachine::getMinDistance, args("input"))
  .def("getNMeans", &KMeansMachine::getNMeans)
  .def("getNInputs", &KMeansMachine::getNInputs)
  .def("forward", &KMeansMachine::forward, args("input"))
  .def("getVariancesAndWeightsForEachCluster", &getVariancesAndWeightsForEachCluster, args("machine", "sampler"))
  ;
  
  class_<Gaussian>("Gaussian", init<int>())
  .def(init<Gaussian&>(args("other")))
  .add_property("nInputs", &Gaussian::getNInputs, &Gaussian::setNInputs)
  .add_property("mean", &Gaussian_getMean, &Gaussian::setMean)
  .add_property("variance", &Gaussian_getVariance, &Gaussian::setVariance)
  .add_property("varianceThresholds", &Gaussian_getVarianceThresholds, (void (Gaussian::*)(const blitz::Array<double,1>&)) &Gaussian::setVarianceThresholds)
  .def("setVarianceThresholds", (void (Gaussian::*)(double))&Gaussian::setVarianceThresholds)
  .def("resize", &Gaussian::resize)
  .def("logLikelihood", &Gaussian::logLikelihood)
  .def("print_", &Gaussian::print)
  ;

  class_<GMMStats>("GMMStats", init<>())
  .def(init<int, int>(args("n_gaussians","n_inputs")))
  .def_readwrite("T", &GMMStats::T)
  .def_readwrite("n", &GMMStats::n)
  .def_readwrite("sumPx", &GMMStats::sumPx)
  .def_readwrite("sumPxx", &GMMStats::sumPxx)
  .def("resize", &GMMStats::resize, args("n_gaussians", "n_inputs"))
  .def("init", &GMMStats::init)
  .def("save", &GMMStats::save)
  .def("load", &GMMStats::load)
  .def("print_", &GMMStats::print)
  ;
  
  class_<GMMMachine, bases<Machine<FrameSample, double> > >("GMMMachine", init<int, int>())
  .def(init<GMMMachine&>())
  .add_property("nInputs", &GMMMachine::getNInputs, &GMMMachine::setNInputs)
  .def("resize", &GMMMachine::resize, args("n_gaussians", "n_inputs"))
  .add_property("weights", (blitz::Array<double, 1> (GMMMachine::*)() const)&GMMMachine::getWeights, &GMMMachine::setWeights)
  .add_property("means", (blitz::Array<double, 2> (GMMMachine::*)() const) &GMMMachine::getMeans, &GMMMachine::setMeans)
  .add_property("variances", (blitz::Array<double, 2> (GMMMachine::*)() const) &GMMMachine::getVariances, &GMMMachine::setVariances)
  .add_property("varianceThresholds", (blitz::Array<double, 2> (GMMMachine::*)() const) &GMMMachine::getVarianceThresholds, (void (GMMMachine::*)(const blitz::Array<double,2>&))&GMMMachine::setVarianceThresholds)
  .def("setVarianceThresholds", (void (GMMMachine::*)(double))&GMMMachine::setVarianceThresholds, args("factor"))
  .def("setVarianceThresholds", (void (GMMMachine::*)(blitz::Array<double,1>))&GMMMachine::setVarianceThresholds, args("variance_thresholds"))
  .def("logLikelihood", (double (GMMMachine::*)(const blitz::Array<float,1>&, blitz::Array<double,1>&) const)&GMMMachine::logLikelihood, args("x", "log_weighted_gaussian_likelihoods"))
  .def("logLikelihood", (double (GMMMachine::*)(const blitz::Array<float,1>&) const)&GMMMachine::logLikelihood, args("x"))
  .def("forward", &GMMMachine::forward, args("input"))
  .def("accStatistics", (void (GMMMachine::*)(const Torch::trainer::Sampler<FrameSample>&, GMMStats&) const)&GMMMachine::accStatistics, args("sampler", "stats"))
  .def("accStatistics", (void (GMMMachine::*)(const blitz::Array<float,1>&, GMMStats&) const)&GMMMachine::accStatistics, args("x", "stats"))
  .def("getGaussian", &GMMMachine::getGaussian, return_value_policy<reference_existing_object>(), args("i"))
  .def("getNGaussians", &GMMMachine::getNGaussians)
  .def("print_", &GMMMachine::print)
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
