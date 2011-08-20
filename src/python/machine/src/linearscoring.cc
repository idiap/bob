#include <boost/python.hpp>
#include <machine/LinearScoring.h>
#include <vector>

using namespace boost::python;

static boost::shared_ptr<blitz::Array<double, 2> > linearScoring(list models,
                                                          Torch::machine::GMMMachine& ubm,
                                                          list test_stats,
                                                          blitz::Array<double, 2>* test_channelOffset = NULL,
                                                          bool frame_length_normalisation = false) {
  int size_models = len(models);
  std::vector<Torch::machine::GMMMachine*> models_c;

  for(int i = 0; i < size_models; i++) {
    models_c.push_back(extract<Torch::machine::GMMMachine*>(models[i]));
  }

  int size_test_stats = len(test_stats);
  std::vector<Torch::machine::GMMStats*> test_stats_c;

  for(int i = 0; i < size_test_stats; i++) {
    test_stats_c.push_back(extract<Torch::machine::GMMStats*>(test_stats[i]));
  }

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);
  
  Torch::machine::linearScoring(models_c, ubm, test_stats_c, test_channelOffset, frame_length_normalisation, *ret.get());
  
  return ret;
}


static boost::shared_ptr<blitz::Array<double, 2> > linearScoring1(list models,
                                                          const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                                                          list test_stats,
                                                          list test_channelOffset,
                                                          bool frame_length_normalisation = false) 
{
  int size_models = len(models);
  std::vector<blitz::Array<double,1> > models_c;

  for(int i = 0; i < size_models; i++) {
    models_c.push_back(extract<blitz::Array<double,1> >(models[i]));
  }

  int size_test_stats = len(test_stats);
  std::vector<const Torch::machine::GMMStats*> test_stats_c;

  for(int i = 0; i < size_test_stats; i++) {
    test_stats_c.push_back(extract<const Torch::machine::GMMStats*>(test_stats[i]));
  }

  int size_test_channelOffset = len(test_channelOffset);
  std::vector<blitz::Array<double,1> > test_channelOffset_c;

  for(int i = 0; i < size_test_channelOffset; i++) {
    test_channelOffset_c.push_back(extract<blitz::Array<double,1> >(test_channelOffset[i]));
  }

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);
  
  Torch::machine::linearScoring(models_c, ubm_mean, ubm_variance, test_stats_c, test_channelOffset_c, frame_length_normalisation, *ret.get());
  
  return ret;
}



static boost::shared_ptr<blitz::Array<double, 2> > linearScoring2(list models,
                                                          const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                                                          list test_stats,
                                                          bool frame_length_normalisation = false) 
{
  int size_models = len(models);
  std::vector<blitz::Array<double,1> > models_c;

  for(int i = 0; i < size_models; i++) {
    models_c.push_back(extract<blitz::Array<double,1> >(models[i]));
  }

  int size_test_stats = len(test_stats);
  std::vector<const Torch::machine::GMMStats*> test_stats_c;

  for(int i = 0; i < size_test_stats; i++) {
    test_stats_c.push_back(extract<const Torch::machine::GMMStats*>(test_stats[i]));
  }

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);
  
  Torch::machine::linearScoring(models_c, ubm_mean, ubm_variance, test_stats_c, frame_length_normalisation, *ret.get());
  
  return ret;
}


static boost::shared_ptr<blitz::Array<double, 2> > linearScoring3(list models,
                                                          Torch::machine::GMMMachine& ubm,
                                                          list test_stats,
                                                          bool frame_length_normalisation = false) 
{
  int size_models = len(models);
  std::vector<const Torch::machine::GMMMachine*> models_c;

  for(int i = 0; i < size_models; i++) {
    models_c.push_back(extract<const Torch::machine::GMMMachine*>(models[i]));
  }

  int size_test_stats = len(test_stats);
  std::vector<const Torch::machine::GMMStats*> test_stats_c;

  for(int i = 0; i < size_test_stats; i++) {
    test_stats_c.push_back(extract<const Torch::machine::GMMStats*>(test_stats[i]));
  }

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);
  
  Torch::machine::linearScoring(models_c, ubm, test_stats_c, frame_length_normalisation, *ret.get());
  
  return ret;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring_overloads, linearScoring, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring1_overloads, linearScoring1, 5, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring2_overloads, linearScoring2, 4, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring3_overloads, linearScoring3, 3, 4)

void bind_machine_linear_scoring() {
  def("linearScoring",
      linearScoring,
      linearScoring_overloads(args("models", "ubm", "test_stats", "test_channelOffset", "frame_length_normalisation"),
                              "Compute a matrix of scores using linear scoring.\n"
                              "Return a 2D matrix of scores, scores[m, s] is the score for model m against statistics s\n"
                              "\n"
                              "Warning Each GMM must have the same size.\n"
                              "\n"
                              "models      -- list of client models\n"
                              "ubm         -- world model\n"
                              "test_stats  -- list of accumulate statistics for each test trial\n"
                              "test_channelOffset -- \n"
                              "frame_length_normlisation -- perform a normalization by the number of feature vectors\n"));
  def("linearScoring1", linearScoring1, linearScoring1_overloads(args("models", "ubm_mean", "ubm_variance", "test_stats", "test_channelOffset", "frame_length_normalisation"),""));
  def("linearScoring2", linearScoring2, linearScoring2_overloads(args("models", "ubm_mean", "ubm_variance", "test_stats", "frame_length_normalisation"),""));
  def("linearScoring3", linearScoring3, linearScoring3_overloads(args("models", "ubm", "test_stats", "frame_length_normalisation"),""));
}
