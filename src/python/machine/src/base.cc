#include <boost/python.hpp>
#include <database/Arrayset.h>
#include <machine/KMeansMachine.h>
#include <machine/GMMMachine.h>
#include <machine/FrameClassificationSample.h>
#include <boost/concept_check.hpp>

using namespace boost::python;
using namespace Torch::machine;


class Machine_FrameSample_double_Wrapper : public Machine<FrameSample, double>, public wrapper<Machine<FrameSample, double> > {
public:
  void forward (const FrameSample& input, double& output) const {
    this->get_override("forward")(input, output);
  }
};

static double Machine_FrameSample_double_forward(const Machine<FrameSample, double>& machine, const FrameSample& input) {
  double output;
  machine.forward(input, output);
  return output;
}
  
static tuple getVariancesAndWeightsForEachCluster(const KMeansMachine& machine, Torch::trainer::Sampler<FrameSample>& sampler) {
  boost::shared_ptr<blitz::Array<double, 2> > variances(new blitz::Array<double, 2>);
  boost::shared_ptr<blitz::Array<double, 1> > weights(new blitz::Array<double, 1>);
  machine.getVariancesAndWeightsForEachCluster(sampler, *variances.get(), *weights.get());
  return boost::python::make_tuple(variances, weights);
}


#define GETTER(class, fct, type, dim) \
static boost::shared_ptr<blitz::Array<type, dim> > class##_##fct(const class& c) {\
  boost::shared_ptr<blitz::Array<type, dim> > v(new blitz::Array<type, dim>);\
  c.fct(*v.get());\
  return v;\
}

GETTER(Gaussian, getMean, double, 1)
GETTER(Gaussian, getVariance, double, 1)
GETTER(Gaussian, getVarianceThresholds, double, 1)

GETTER(KMeansMachine, getMeans, double, 2)

static boost::shared_ptr<blitz::Array<double, 1> > KMeansMachine_getMean(const KMeansMachine& kMeansMachine, int i) {
  boost::shared_ptr<blitz::Array<double, 1> > mean(new blitz::Array<double, 1>);
  kMeansMachine.getMean(i, *mean.get());
  return mean;
}

GETTER(GMMMachine, getMeans, double, 2)
GETTER(GMMMachine, getWeights, double, 1)
GETTER(GMMMachine, getVariances, double, 2)
GETTER(GMMMachine, getVarianceThresholds, double, 2)

void bind_machine_base() {

  class_<FrameSample>("FrameSample",
                      "This class represents one Frame. It encapsulates a blitz::Array<double, 1>",
                      init<const blitz::Array<double, 1>& >(args("array")))
  .def("getFrame",
       &FrameSample::getFrame, return_value_policy<copy_const_reference>(),
       "Get the Frame")
  .def("getFrameSize",
       &FrameSample::getFrameSize,
       "Get the frame size")
  ;
  
  class_<FrameClassificationSample>("FrameClassificationSample",
                      "This class represents one Frame with a classification label. It encapsulates a blitz::Array<double, 1> and an int",
                      init<const blitz::Array<double, 1>&, int64_t >(args("array","target")))
  .def("getFrame",
       &FrameClassificationSample::getFrame, return_value_policy<copy_const_reference>(),
       "Get the Frame")
  .def("getFrameSize",
       &FrameClassificationSample::getFrameSize,
       "Get the frame size")
  .def("getTarget",
       &FrameClassificationSample::getTarget,
       "Get the target")
  ;
  
  class_<Machine_FrameSample_double_Wrapper, boost::noncopyable>("Machine_FrameSample_double_",
                                                                 "Root class for all Machine<FrameSample, double>")
  .def("forward",
       &Machine_FrameSample_double_forward,
       args("input"),
       "Execute the machine")
  ;
  
  class_<KMeansMachine, bases<Machine<FrameSample, double> > >("KMeansMachine",
                                                               "This class implements a k-means classifier.\n"
                                                               "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006",
                                                               init<int, int>(args("n_means", "n_inputs")))
  .add_property("means",
                &KMeansMachine_getMeans,
                &KMeansMachine::setMeans,
               "Means")
  .add_property("nInputs",
                &KMeansMachine::getNInputs,
                "Number of inputs")
  .def("getMean",
       &KMeansMachine_getMean,
       args("i", "mean"),
       "Get the i'th mean")
  .def("setMean",
       &KMeansMachine::setMean,
       args("i" "mean"),
       "Set the i'th mean")
  .def("getDistanceFromMean",
       &KMeansMachine::getDistanceFromMean,
       args("x", "i"),
       "Return the Euclidean distance of the sample, x, to the i'th mean")
  .def("getClosestMean",
       &KMeansMachine::getClosestMean,
       args("x", "closest_mean", "min_distance"),
       "Calculate the index of the mean that is closest (in terms of Euclidean distance) to the data sample, x")
  .def("getMinDistance",
       &KMeansMachine::getMinDistance,
       args("input"),
       "Output the minimum distance between the input and one of the means")
  .def("getNMeans",
       &KMeansMachine::getNMeans,
       "Return the number of means")
  .def("getVariancesAndWeightsForEachCluster",
       &getVariancesAndWeightsForEachCluster,
       args("machine", "sampler"),
       "For each mean, find the subset of the samples that is closest to that mean, and calculate\n"
       "1) the variance of that subset (the cluster variance)\n"
       "2) the proportion of the samples represented by that subset (the cluster weight)")
  ;
  
  class_<Gaussian>("Gaussian",
                   "This class implements a multivariate diagonal Gaussian distribution",
                   init<>())
  .def(init<int>(args("n_inputs")))
  .def(init<Gaussian&>(args("other")))
  .def(init<Torch::database::HDF5File&>(args("config")))
  .def(self == self)
  .add_property("nInputs",
                &Gaussian::getNInputs,
                &Gaussian::setNInputs,
                "Input dimensionality")
  .add_property("mean",
                &Gaussian_getMean,
                &Gaussian::setMean,
                "Mean of the Gaussian")
  .add_property("variance",
                &Gaussian_getVariance,
                &Gaussian::setVariance,
                "The diagonal of the covariance matrix")
  .add_property("varianceThresholds",
                &Gaussian_getVarianceThresholds,
                (void (Gaussian::*)(const blitz::Array<double,1>&)) &Gaussian::setVarianceThresholds,
                "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
                "The variance will be set to this value if an attempt is made to set it to a smaller value.")
  .def("setVarianceThresholds",
       (void (Gaussian::*)(double))&Gaussian::setVarianceThresholds,
       "Set the variance flooring thresholds")
  .def("resize",
       &Gaussian::resize,
       "Set the input dimensionality, reset the mean to zero and the variance to one.")
  .def("logLikelihood",
       &Gaussian::logLikelihood,
       "Output the log likelihood of the sample, x")
  .def("save",
       &Gaussian::save,
       "Save to a Configuration")
  .def("load",
       &Gaussian::load,
       "Load from a Configuration")
  .def(self_ns::str(self_ns::self))
  ;

  class_<GMMStats>("GMMStats",
                   "A container for GMM statistics.\n"
                   "With respect to Reynolds, \"Speaker Verification Using Adapted "
                   "Gaussian Mixture Models\", DSP, 2000:\n"
                   "Eq (8) is n(i)\n"
                   "Eq (9) is sumPx(i) / n(i)\n"
                   "Eq (10) is sumPxx(i) / n(i)\n",
                   init<>())
  .def(init<int, int>(args("n_gaussians","n_inputs")))
  .def(init<Torch::database::HDF5File&>(args("config")))
  .def_readwrite("log_likelihood",
                 &GMMStats::log_likelihood,
                 "The accumulated log likelihood of all samples")
  .def_readwrite("T",
                 &GMMStats::T,
                 "The accumulated number of samples")
  .def_readwrite("n",
                 &GMMStats::n,
                 "For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)")
  .def_readwrite("sumPx",
                 &GMMStats::sumPx,
                 "For each Gaussian, the accumulated sum of responsibility times the sample ")
  .def_readwrite("sumPxx",
                 &GMMStats::sumPxx,
                 "For each Gaussian, the accumulated sum of responsibility times the sample squared")
  .def("resize",
       &GMMStats::resize, args("n_gaussians", "n_inputs"),
       " Allocates space for the statistics and resets to zero.")
  .def("init",
       &GMMStats::init,
       "Resets statistics to zero.")
  .def("save",
       &GMMStats::save,
       " Save to a Configuration")
  .def("load",
       &GMMStats::load,
       "Load from a Configuration")
  .def(self_ns::str(self_ns::self))
  ;
  
  class_<GMMMachine, bases<Machine<FrameSample, double> > >("GMMMachine",
                                                            "This class implements a multivariate diagonal Gaussian distribution.\n"
                                                            "See Section 2.3.9 of Bishop, \"Pattern recognition and machine learning\", 2006",
                                                            init<int, int>(args("n_gaussians", "n_inputs")))
  .def(init<GMMMachine&>())
  .def(init<Torch::database::HDF5File&>(args("config")))
  .def(self == self)
  .add_property("nInputs",
                &GMMMachine::getNInputs,
                &GMMMachine::setNInputs,
                "The feature dimensionality")
  .add_property("weights",
                &GMMMachine_getWeights,
                &GMMMachine::setWeights,
                "The weights (also known as \"mixing coefficients\")")
  .add_property("means",
                &GMMMachine_getMeans,
                &GMMMachine::setMeans,
                "The means of the gaussians")
  .add_property("variances",
                &GMMMachine_getVariances,
                &GMMMachine::setVariances,
                "The variances")
  .add_property("varianceThresholds",
                &GMMMachine_getVarianceThresholds,
                (void (GMMMachine::*)(const blitz::Array<double,2>&))&GMMMachine::setVarianceThresholds,
                "The variance flooring thresholds for each Gaussian in each dimension")
  .def("resize",
       &GMMMachine::resize,
       args("n_gaussians", "n_inputs"),
       "Reset the input dimensionality, and the number of Gaussian components.\n"
       "Initialises the weights to uniform distribution.")
  .def("setVarianceThresholds",
       (void (GMMMachine::*)(double))&GMMMachine::setVarianceThresholds,
       args("factor"),
       "Set the variance flooring thresholds in each dimension "
       "to a proportion of the current variance, for each Gaussian")
  .def("setVarianceThresholds",
       (void (GMMMachine::*)(blitz::Array<double,1>))&GMMMachine::setVarianceThresholds,
       args("variance_thresholds"),
       "Set the variance flooring thresholds in each dimension "
       "(equal for all Gaussian components)")
  .def("logLikelihood",
       (double (GMMMachine::*)(const blitz::Array<double,1>&, blitz::Array<double,1>&) const)&GMMMachine::logLikelihood,
       args("x", "log_weighted_gaussian_likelihoods"),
       "Output the log likelihood of the sample, x, i.e. log(p(x|GMMMachine))")
  .def("logLikelihood",
       (double (GMMMachine::*)(const blitz::Array<double,1>&) const)&GMMMachine::logLikelihood,
       args("x"),
       " Output the log likelihood of the sample, x, i.e. log(p(x|GMM))")
  .def("accStatistics",
       (void (GMMMachine::*)(const Torch::trainer::Sampler<FrameSample>&, GMMStats&) const)&GMMMachine::accStatistics,
       args("sampler", "stats"),
       "Accumulates the GMM statistics over a set of samples.")
  .def("accStatistics",
       (void (GMMMachine::*)(const blitz::Array<double,1>&, GMMStats&) const)&GMMMachine::accStatistics,
       args("x", "stats"),
       "Accumulate the GMM statistics for this sample.")
  .def("getGaussian",
       &GMMMachine::getGaussian, return_value_policy<reference_existing_object>(),
       args("i"),
       "Get a pointer to a particular Gaussian component")
  .def("getNGaussians",
       &GMMMachine::getNGaussians,
       "Return the number of Gaussian components")
  .def("load",
       &GMMMachine::load,
       "Load from a Configuration")
  .def("save",
       &GMMMachine::save,
       "Save to a Configuration")
  .def(self_ns::str(self_ns::self))
  ;
}
