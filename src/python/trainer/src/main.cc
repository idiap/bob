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
  
  class_<Sampler_FrameSample_Wrapper, boost::noncopyable>("Sampler_FrameSample_")
  .def("getSample", &Sampler<FrameSample>::getSample, args("index"))
  .def("getNSamples", &Sampler<FrameSample>::getNSamples)
  ;
  
  class_<SimpleFrameSampler, bases<Sampler<FrameSample> > >("SimpleFrameSampler", init<Torch::database::Arrayset>(args("arrayset")));
  
  class_<Trainer_KMeansMachine_FrameSample_Wrapper, boost::noncopyable>("Trainer_KMeansMachine_FrameSample_")
  .def("train", &Trainer<KMeansMachine, FrameSample>::train, args("machine", "sampler"));
  ;
  
  class_<Trainer_EigenMachine_FrameSample_Wrapper, boost::noncopyable>("Trainer_EigenMachine_FrameSample_")
  .def("train", &Trainer<EigenMachine, FrameSample>::train, args("machine", "sampler"));
  ;
  
  class_<EMTrainer_Machine_FrameSample_double_FrameSample_Wrapper, boost::noncopyable>("EMTrainer_Machine_FrameSample_double_FrameSample_", init<int, int>((arg("convergence_threshold") = 0.001, arg("max_iterations") = 10)))
  .def("train", &EMTrainer<Machine<FrameSample, double>, FrameSample>::train, args("machine", "sampler"))
  .def("initialization", &EMTrainer<Machine<FrameSample, double>, FrameSample>::initialization, args("machine", "sampler"))
  .def("eStep", &EMTrainer<Machine<FrameSample, double>, FrameSample>::eStep, args("machine", "sampler"))
  .def("mStep", &EMTrainer<Machine<FrameSample, double>, FrameSample>::mStep, args("machine", "sampler"))
  ;
  
  class_<KMeansTrainer, bases<EMTrainer<Machine<FrameSample, double>, FrameSample> > >("KMeansTrainer", init<>())
  .add_property("seed", &KMeansTrainer::getSeed, &KMeansTrainer::setSeed)
  .def("train", &KMeansTrainer::train, args("machine", "sampler"))
  ;
  
  class_<GMMTrainer_Wrapper, bases<EMTrainer<Machine<FrameSample, double>, FrameSample> >, boost::noncopyable >("GMMTrainer", init<bool, bool, bool>((arg("update_means") = true, arg("update_variances") = false, arg("update_weights") = false)))
  .def("train", &GMMTrainer::train, args("machine", "sampler"))
  ;
  
  class_<MAP_GMMTrainer, bases<GMMTrainer> >("MAP_GMMTrainer", init<double>((arg("relevance_factor") = 0)))
  .def("setPriorGMM", &MAP_GMMTrainer::setPriorGMM)
  ;
  
  class_<ML_GMMTrainer, bases<GMMTrainer> >("ML_GMMTrainer", init<bool, bool, bool>((arg("update_means") = true, arg("update_variances") = false, arg("update_weights") = false)))
  ;

  class_<SVDPCATrainer, bases<Trainer<EigenMachine, FrameSample>, FrameSample>, boost::noncopyable >("SVDPCATrainer", init<>())
  .def("train", &SVDPCATrainer::train, args("machine", "sampler"))
  ;
  
  bind_trainer_exception();
}
