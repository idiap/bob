#include <boost/python.hpp>
#include <boost/concept_check.hpp>
#include "io/Arrayset.h"
#include "machine/KMeansMachine.h"
#include "machine/GMMMachine.h"
#include "machine/GMMLLRMachine.h"

#include "core/python/pycore.h"

using namespace boost::python;
namespace io = Torch::io;
namespace mach = Torch::machine;
namespace tp = Torch::python;

static tuple getVariancesAndWeightsForEachCluster(const mach::KMeansMachine& machine, io::Arrayset& ar) {
  blitz::Array<double, 2> variances;
  blitz::Array<double, 1> weights;
  machine.getVariancesAndWeightsForEachCluster(ar, variances, weights);
  return boost::python::make_tuple(variances, weights);
}


#define GETTER(class, class_name, fct, type, dim) \
static blitz::Array<type, dim> class_name##_##fct(const class& c) {\
  blitz::Array<type, dim> v;\
  c.fct(v);\
  return v;\
}

GETTER(mach::Gaussian, mach_Gaussian, getMean, double, 1)
GETTER(mach::Gaussian, mach_Gaussian, getVariance, double, 1)
GETTER(mach::Gaussian, mach_Gaussian, getVarianceThresholds, double, 1)

GETTER(mach::KMeansMachine, mach_KMeansMachine, getMeans, double, 2)

static blitz::Array<double, 1> mach_KMeansMachine_getMean(const mach::KMeansMachine& kMeansMachine, int i) {
  blitz::Array<double, 1> mean;
  kMeansMachine.getMean(i, mean);
  return mean;
}

GETTER(mach::GMMMachine, mach_GMMMachine, getMeans, double, 2)
GETTER(mach::GMMMachine, mach_GMMMachine, getWeights, double, 1)
GETTER(mach::GMMMachine, mach_GMMMachine, getVariances, double, 2)
GETTER(mach::GMMMachine, mach_GMMMachine, getVarianceThresholds, double, 2)
GETTER(mach::GMMMachine, mach_GMMMachine, getMeanSupervector, double, 1)
GETTER(mach::GMMMachine, mach_GMMMachine, getVarianceSupervector, double, 1)

static double forward(const mach::Machine<blitz::Array<double,1>, double>& m,
    const blitz::Array<double,1>& input) {
  double output;
  m.forward(input, output);
  return output;
}

static void gmmmach_getmeansupervector(const mach::GMMMachine& m,
    numeric::array a) {
  blitz::Array<double,1> a_ = tp::numpy_bz<double,1>(a);
  m.getMeanSupervector(a_);
}

static void gmmmach_getvariancesupervector(const mach::GMMMachine& m,
    numeric::array a) {
  blitz::Array<double,1> a_ = tp::numpy_bz<double,1>(a);
  m.getVarianceSupervector(a_);
}

static blitz::Array<double,1> gmmstats_get_n(mach::GMMStats& s) {
  return s.n;
}

static void gmmstats_set_n(mach::GMMStats& s, const blitz::Array<double,1>& n) {
  s.n = n;
}

static blitz::Array<double,2> gmmstats_get_sumpx(mach::GMMStats& s) {
  return s.sumPx;
}

static void gmmstats_set_sumpx(mach::GMMStats& s, const blitz::Array<double,2>& n) {
  s.sumPx = n;
}

static blitz::Array<double,2> gmmstats_get_sumpxx(mach::GMMStats& s) {
  return s.sumPxx;
}

static void gmmstats_set_sumpxx(mach::GMMStats& s, const blitz::Array<double,2>& n) {
  s.sumPxx = n;
}

void bind_machine_gmm() {

  class_<mach::Machine<blitz::Array<double,1>, double>, boost::noncopyable>("MachineDoubleBase", 
      "Root class for all Machine<blitz::Array<double,1>, double>", no_init)
    .def("__call__", &mach::Machine<blitz::Array<double,1>, double>::forward, (arg("input"), arg("output")), "Execute the machine")
    .def("forward", &mach::Machine<blitz::Array<double,1>, double>::forward, (arg("input"), arg("output")), "Execute the machine")
    .def("forward_", &mach::Machine<blitz::Array<double,1>, double>::forward_, (arg("input"), arg("output")), "Execute the machine. NO check is performed.")
    .def("__call__", &forward, (arg("self"), arg("input")), "Execute the machine, and returns the output")
    .def("forward", &forward, (arg("self"), arg("input")), "Execute the machine, and returns the output")
  ;

  class_<mach::KMeansMachine, bases<mach::Machine<blitz::Array<double,1>, double> > >("KMeansMachine",
      "This class implements a k-means classifier.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<int, int>(args("n_means", "n_inputs")))
    .add_property("means", &mach_KMeansMachine_getMeans, &mach::KMeansMachine::setMeans, "Means")
    .add_property("nInputs", &mach::KMeansMachine::getNInputs, "Number of inputs")
    .def("getMean", &mach_KMeansMachine_getMean, (arg("i"), arg("mean")), "Get the i'th mean")
    .def("setMean", &mach::KMeansMachine::setMean, (arg("i"), arg("mean")), "Set the i'th mean")
    .def("getDistanceFromMean", &mach::KMeansMachine::getDistanceFromMean, (arg("x"), arg("i")),
        "Return the Euclidean distance of the sample, x, to the i'th mean")
    .def("getClosestMean", &mach::KMeansMachine::getClosestMean, (arg("x"), arg("closest_mean"), arg("min_distance")),
        "Calculate the index of the mean that is closest (in terms of Euclidean distance) to the data sample, x")
    .def("getMinDistance", &mach::KMeansMachine::getMinDistance, (arg("input")),
        "Output the minimum distance between the input and one of the means")
    .def("getNMeans", &mach::KMeansMachine::getNMeans, "Return the number of means")
    .def("getVariancesAndWeightsForEachCluster", &getVariancesAndWeightsForEachCluster, (arg("machine"), arg("data")),
        "For each mean, find the subset of the samples that is closest to that mean, and calculate\n"
        "1) the variance of that subset (the cluster variance)\n"
        "2) the proportion of the samples represented by that subset (the cluster weight)")
  ;
  
  class_<mach::Gaussian>("Gaussian",
                   "This class implements a multivariate diagonal mach::Gaussian distribution",
                   init<>())
  .def(init<int>(args("n_inputs")))
  .def(init<mach::Gaussian&>(args("other")))
  .def(init<Torch::io::HDF5File&>(args("config")))
  .def(self == self)
  .add_property("nInputs",
                &mach::Gaussian::getNInputs,
                &mach::Gaussian::setNInputs,
                "Input dimensionality")
  .add_property("mean",
                &mach_Gaussian_getMean,
                &mach::Gaussian::setMean,
                "Mean of the mach::Gaussian")
  .add_property("variance",
                &mach_Gaussian_getVariance,
                &mach::Gaussian::setVariance,
                "The diagonal of the covariance matrix")
  .add_property("varianceThresholds",
                &mach_Gaussian_getVarianceThresholds,
                (void (mach::Gaussian::*)(const blitz::Array<double,1>&)) &mach::Gaussian::setVarianceThresholds,
                "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
                "The variance will be set to this value if an attempt is made to set it to a smaller value.")
  .def("setVarianceThresholds",
       (void (mach::Gaussian::*)(double))&mach::Gaussian::setVarianceThresholds,
       "Set the variance flooring thresholds")
    .def("resize", &mach::Gaussian::resize, "Set the input dimensionality, reset the mean to zero and the variance to one.")
    .def("logLikelihood", &mach::Gaussian::logLikelihood, "Output the log likelihood of the sample, x")
    .def("save", &mach::Gaussian::save, "Save to a Configuration")
    .def("load", &mach::Gaussian::load, "Load from a Configuration")
    .def(self_ns::str(self_ns::self))
  ;

  class_<mach::GMMStats>("GMMStats",
                   "A container for GMM statistics.\n"
                   "With respect to Reynolds, \"Speaker Verification Using Adapted "
                   "Gaussian Mixture Models\", DSP, 2000:\n"
                   "Eq (8) is n(i)\n"
                   "Eq (9) is sumPx(i) / n(i)\n"
                   "Eq (10) is sumPxx(i) / n(i)\n",
                   init<>())
  .def(init<int, int>(args("n_gaussians","n_inputs")))
  .def(init<Torch::io::HDF5File&>(args("config")))
  .def_readwrite("log_likelihood",
                 &mach::GMMStats::log_likelihood,
                 "The accumulated log likelihood of all samples")
  .def_readwrite("T",
                 &mach::GMMStats::T,
                 "The accumulated number of samples")
  .add_property("n", &gmmstats_get_n, &gmmstats_set_n, "For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)")
  .add_property("sumPx", &gmmstats_get_sumpx, &gmmstats_set_sumpx, "For each Gaussian, the accumulated sum of responsibility times the sample ")
  .add_property("sumPxx", &gmmstats_get_sumpxx, &gmmstats_set_sumpxx, "For each Gaussian, the accumulated sum of responsibility times the sample squared")
  .def("resize",
       &mach::GMMStats::resize, args("n_gaussians", "n_inputs"),
       " Allocates space for the statistics and resets to zero.")
  .def("init",
       &mach::GMMStats::init,
       "Resets statistics to zero.")
  .def("save",
       &mach::GMMStats::save,
       " Save to a Configuration")
  .def("load",
       &mach::GMMStats::load,
       "Load from a Configuration")
  .def(self_ns::str(self_ns::self))
  ;
  
  class_<mach::GMMMachine, bases<mach::Machine<blitz::Array<double,1>, double> > >("GMMMachine",
                                                            "This class implements a multivariate diagonal Gaussian distribution.\n"
                                                            "See Section 2.3.9 of Bishop, \"Pattern recognition and machine learning\", 2006",
                                                            init<int, int>(args("n_gaussians", "n_inputs")))
  .def(init<mach::GMMMachine&>())
  .def(init<Torch::io::HDF5File&>(args("config")))
  .def(self == self)
  .add_property("nInputs",
                &mach::GMMMachine::getNInputs,
                &mach::GMMMachine::setNInputs,
                "The feature dimensionality")
  .add_property("nGaussians",
                &mach::GMMMachine::getNGaussians,
                "The number of Gaussian components")
  .add_property("weights",
                &mach_GMMMachine_getWeights,
                &mach::GMMMachine::setWeights,
                "The weights (also known as \"mixing coefficients\")")
  .add_property("means",
                &mach_GMMMachine_getMeans,
                &mach::GMMMachine::setMeans,
                "The means of the gaussians")
  .add_property("meanSupervector",
                &mach_GMMMachine_getMeanSupervector,
                &mach::GMMMachine::setMeanSupervector,
                "The mean supervector of the GMMMachine "
                "(concatenation of the mean vectors of each Gaussian of the GMMMachine")
  .add_property("variances",
                &mach_GMMMachine_getVariances,
                &mach::GMMMachine::setVariances,
                "The variances")
  .add_property("varianceSupervector",
                &mach_GMMMachine_getVarianceSupervector,
                &mach::GMMMachine::setVarianceSupervector,
                "The variance supervector of the GMMMachine "
                "(concatenation of the variance vectors of each Gaussian of the GMMMachine")
  .add_property("varianceThresholds",
                &mach_GMMMachine_getVarianceThresholds,
                (void (mach::GMMMachine::*)(const blitz::Array<double,2>&))&mach::GMMMachine::setVarianceThresholds,
                "The variance flooring thresholds for each Gaussian in each dimension")
  .def("resize",
       &mach::GMMMachine::resize,
       args("n_gaussians", "n_inputs"),
       "Reset the input dimensionality, and the number of Gaussian components.\n"
       "Initialises the weights to uniform distribution.")
  .def("setVarianceThresholds",
       (void (mach::GMMMachine::*)(double))&mach::GMMMachine::setVarianceThresholds,
       args("factor"),
       "Set the variance flooring thresholds in each dimension "
       "to a proportion of the current variance, for each Gaussian")
  .def("setVarianceThresholds",
       (void (mach::GMMMachine::*)(blitz::Array<double,1>))&mach::GMMMachine::setVarianceThresholds,
       args("variance_thresholds"),
       "Set the variance flooring thresholds in each dimension "
       "(equal for all Gaussian components)")
  .def("logLikelihood",
       (double (mach::GMMMachine::*)(const blitz::Array<double,1>&, blitz::Array<double,1>&) const)&mach::GMMMachine::logLikelihood,
       args("x", "log_weighted_gaussian_likelihoods"),
       "Output the log likelihood of the sample, x, i.e. log(p(x|mach::GMMMachine))")
  .def("logLikelihood",
       (double (mach::GMMMachine::*)(const blitz::Array<double,1>&) const)&mach::GMMMachine::logLikelihood,
       args("x"),
       " Output the log likelihood of the sample, x, i.e. log(p(x|GMM))")
  .def("accStatistics",
       (void (mach::GMMMachine::*)(const blitz::Array<double,1>&, mach::GMMStats&) const)&mach::GMMMachine::accStatistics,
       args("x", "stats"),
       "Accumulate the GMM statistics for this sample.")
  .def("accStatistics",
       (void (mach::GMMMachine::*)(const io::Arrayset&, mach::GMMStats&) const)&mach::GMMMachine::accStatistics,
       args("sampler", "stats"),
       "Accumulates the GMM statistics over a set of samples.")
  .def("getGaussian",
       &mach::GMMMachine::getGaussian, return_value_policy<reference_existing_object>(),
       args("i"),
       "Get a pointer to a particular Gaussian component")
  .def("getMeanSupervector", &gmmmach_getmeansupervector,
       args("mean_supervector"),
       "Get the mean supervector of the GMMMachine "
       "(concatenation of the mean vectors of each Gaussian of the GMMMachine)")
  .def("getVarianceSupervector", &gmmmach_getvariancesupervector,
       args("variance_supervector"),
       "Get the variance supervector of the GMMMachine "
       "(concatenation of the variance vectors of each Gaussian of the GMMMachine)")
  .def("load",
       &mach::GMMMachine::load,
       "Load from a Configuration")
  .def("save",
       &mach::GMMMachine::save,
       "Save to a Configuration")
  .def(self_ns::str(self_ns::self))
  ;

  class_<mach::GMMLLRMachine, bases<mach::Machine<blitz::Array<double,1>, double> > >("GMMLLRMachine",
       "This class implements computes log likelihood ratio, given a client and a UBM GMM.\n",
        no_init)
  .def(init<mach::GMMLLRMachine&>())
  .def(init<Torch::io::HDF5File&>(args("config")))
  .def(init<Torch::io::HDF5File&,Torch::io::HDF5File&>(args("client", "ubm")))
  .def(init<mach::GMMMachine&,mach::GMMMachine&>(args("client", "ubm")))
  .def(self == self)
  .def("getGMMClient",
       &mach::GMMLLRMachine::getGMMClient, return_value_policy<reference_existing_object>(),
       "Get a pointer to the client GMM")
  .def("getGMMUBM",
       &mach::GMMLLRMachine::getGMMUBM, return_value_policy<reference_existing_object>(),
       "Get a pointer to the UBM GMM")
 .add_property("nInputs", &mach::GMMMachine::getNInputs, "The feature dimensionality")
  .def("load",
       &mach::GMMLLRMachine::load,
       "Load from a Configuration")
  .def("save",
       &mach::GMMLLRMachine::save,
       "Save to a Configuration")
  .def(self_ns::str(self_ns::self))
  ;
}
