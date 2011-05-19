#include <boost/python.hpp>
#include <trainer/SimpleFrameSampler.h>
#include <trainer/KMeansTrainer.h>
#include <trainer/Sampler.h>
#include <trainer/GMMTrainer.h>
#include <trainer/MAP_GMMTrainer.h>
#include <trainer/ML_GMMTrainer.h>
#include <trainer/SVDPCATrainer.h>

using namespace boost::python;
using namespace Torch::machine;
using namespace Torch::trainer;

class Sampler_FrameSample_Wrapper: public Sampler<FrameSample>, public wrapper<Sampler<FrameSample> > {
public:
    const FrameSample getSample(int index) const {
      return this->get_override("getSample")(index);
    }
    
    int getNSamples() const {
      return this->get_override("getNSamples")();
    }
};

class Trainer_KMeansMachine_FrameSample_Wrapper: public Trainer<KMeansMachine, FrameSample>, public wrapper<Trainer<KMeansMachine, FrameSample> > {
    void train(KMeansMachine& machine, const Sampler<FrameSample>& data) {
      this->get_override("train")(machine, data);
    }
};

class EMTrainer_Machine_FrameSample_double_FrameSample_Wrapper : public EMTrainer<Machine<FrameSample, double>, FrameSample>, public wrapper<EMTrainer<Machine<FrameSample, double>, FrameSample> > {
public:
  EMTrainer_Machine_FrameSample_double_FrameSample_Wrapper(double convergence_threshold = 0.001, int max_iterations = 10) :
    EMTrainer<Machine<FrameSample, double>, FrameSample>(convergence_threshold, max_iterations) {}
  
  void initialization(Machine<FrameSample, double>& machine, const Sampler<FrameSample>& data) {
    this->get_override("initializtion")(machine, data);
  }
  
  double eStep(Machine<FrameSample, double>& machine, const Sampler<FrameSample>& data) {
    return this->get_override("eStep")(machine, data);
  }
  
  void mStep(Machine<FrameSample, double>& machine, const Sampler<FrameSample>& data) {
    this->get_override("mStep")(machine, data);
  }
  
};


class GMMTrainer_Wrapper: public GMMTrainer, public wrapper<GMMTrainer> {
public:
  GMMTrainer_Wrapper(bool update_means = true, bool update_variances = false, bool update_weights = false) :
    GMMTrainer(update_means, update_variances, update_weights) {
    }
  
  void mStep(GMMMachine& machine, const Sampler<FrameSample>& data) {
    this->get_override("mStep")(machine, data);
  }
  
};

class Trainer_EigenMachine_FrameSample_Wrapper: public Trainer<EigenMachine, FrameSample>, public wrapper<Trainer<EigenMachine, FrameSample> > {
    void train(EigenMachine& machine, const Sampler<FrameSample>& data) {
      this->get_override("train")(machine, data);
    }
};

void bind_trainer_exception();

BOOST_PYTHON_MODULE(libpytorch_trainer) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch classes and sub-classes for trainers";
  
  class_<Sampler_FrameSample_Wrapper, boost::noncopyable>("Sampler_FrameSample_",
                                                          "This class provides a list of FrameSample.")
  .def("getSample",
       &Sampler<FrameSample>::getSample, args("index"),
       "Get a sample")
  .def("getNSamples",
       &Sampler<FrameSample>::getNSamples,
       "Get the number of Samples")
  ;
  
  class_<SimpleFrameSampler, bases<Sampler<FrameSample> > >("SimpleFrameSampler",
                                                            "This class provides a list of FrameSample from an arrayset",
                                                            init<Torch::database::Arrayset>(args("arrayset")));
  
  class_<Trainer_KMeansMachine_FrameSample_Wrapper, boost::noncopyable>("Trainer_KMeansMachine_FrameSample_",
                                                                        "Trainer<KMeansMachine, FrameSample>")
  .def("train",
       &Trainer<KMeansMachine, FrameSample>::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  ;
  
  class_<Trainer_EigenMachine_FrameSample_Wrapper, boost::noncopyable>("Trainer_EigenMachine_FrameSample_",
                                                                       "Trainer<EigenMachine, FrameSample>")
  .def("train",
       &Trainer<EigenMachine, FrameSample>::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  ;
  
  class_<EMTrainer_Machine_FrameSample_double_FrameSample_Wrapper, boost::noncopyable>("EMTrainer_Machine_FrameSample_double_FrameSample_",
                                                                                       init<optional<int, int> >(args("convergence_threshold", "max_iterations")))
  .def("train",
       &EMTrainer<Machine<FrameSample, double>, FrameSample>::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  .def("initialization",
       &EMTrainer<Machine<FrameSample, double>, FrameSample>::initialization,
       args("machine", "sampler"),
       "This method is called before the EM algorithm")
  .def("eStep",
       &EMTrainer<Machine<FrameSample, double>, FrameSample>::eStep,
       args("machine", "sampler"),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
  .def("mStep",
       &EMTrainer<Machine<FrameSample, double>, FrameSample>::mStep,
       args("machine", "sampler"),
       "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;
  
  class_<KMeansTrainer, bases<EMTrainer<Machine<FrameSample, double>, FrameSample> > >("KMeansTrainer",
                                                                                       "Trains a KMeans machine.\n"
                                                                                       "This class implements the expectation-maximisation algorithm for a k-means machine.\n"
                                                                                       "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
                                                                                       "It uses a random initialisation of the means followed by the expectation-maximization algorithm"
                                                                                       )
  .add_property("seed",
                &KMeansTrainer::getSeed, &KMeansTrainer::setSeed,
                "Seed used to genrated pseudo-random numbers")
  .def("train",
       &KMeansTrainer::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  ;
  
  class_<GMMTrainer_Wrapper, bases<EMTrainer<Machine<FrameSample, double>, FrameSample> >, boost::noncopyable >("GMMTrainer",
                                                                                                                "This class implements the E-step of the expectation-maximisation algorithm for a GMM Machine.\n"
                                                                                                                "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
                                                                                                                init<optional<bool, bool, bool> >(args("update_means", "update_variances", "update_weights")))
  .def("train",
       &GMMTrainer::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  ;
  
  class_<MAP_GMMTrainer, bases<GMMTrainer> >("MAP_GMMTrainer",
                                             "This class implements the maximum a posteriori M-step "
                                             "of the expectation-maximisation algorithm for a GMM Machine. "
                                             "The prior parameters are encoded in the form of a GMM (e.g. a universal background model). "
                                             "The EM algorithm thus performs GMM adaptation.\n"
                                             "See Section 3.4 of Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000. We use a \"single adaptation coefficient\", alpha_i, and thus a single relevance factor, r.",
                                             init<optional<double> >(args("relevance_factor")))
  .def("setPriorGMM",
       &MAP_GMMTrainer::setPriorGMM,
       "Set the GMM to use as a prior for MAP adaptation. "
       "Generally, this is a \"universal background model\" (UBM), "
       "also referred to as a \"world model\".")
  ;
  
  class_<ML_GMMTrainer, bases<GMMTrainer> >("ML_GMMTrainer",
                                            "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.\n"
                                            "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",
                                            init<optional<bool, bool, bool> >(args("update_means", "update_variances", "update_weights")))
  ;

  class_<SVDPCATrainer, bases<Trainer<EigenMachine, FrameSample>, FrameSample>, boost::noncopyable >("SVDPCATrainer", init<>())
  .def("train",
       &SVDPCATrainer::train,
       args("machine", "sampler"),
       "Train a machine using a sampler")
  ;
  
  bind_trainer_exception();
}
