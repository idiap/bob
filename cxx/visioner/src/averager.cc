#include "visioner/model/trainers/averager.h"
#include "visioner/util/timer.h"

namespace bob { namespace visioner {

  // Constructor
  Averager::Averager(const param_t& param)
    :	Trainer(param)

  {		
  }

  // Train a model using the given training and validation samples
  bool Averager::train(	
      const Sampler& t_sampler, const Sampler&,
      Model& model)
  {
    Timer timer;

    // Sample uniformly the training data
    indices_t t_samples;                
    DataSet t_data;

    t_sampler.sample(m_param.m_train_samples, t_samples);
    std::sort(t_samples.begin(), t_samples.end());

    t_sampler.map(t_samples, model, t_data);

    log_info("Averager", "train") << "timing: sampling ~ " << timer.elapsed() << ".\n";

    // Check parameters
    if (t_data.empty())
    {
      log_error("Averager", "train") << "Invalid training samples!\n";
      return false;
    }

    log_info("Averager", "train")
      << "using "
      << t_data.n_samples() << " training samples with "
      << t_data.n_features() << " features.\n";

    // Train the model ...		
    scalars_t avg_outputs(t_data.n_outputs(), 0.0);
    for (index_t s = 0; s < t_data.n_samples(); s ++)
    {
      for (index_t o = 0; o < t_data.n_outputs(); o ++)
      {
        avg_outputs[o] += t_data.targets()(s, o);
      }
    }

    for (index_t o = 0; o < t_data.n_outputs(); o ++)
    {
      avg_outputs[o] *= inverse(t_data.n_samples());

      log_info("Averager", "train")
        << "output [" << (o + 1) << "/" << t_data.n_outputs() << "] = "
        << avg_outputs[o] << ".\n";
    }

    MultiLUTs mluts(t_data.n_outputs(), LUTs(1, LUT(0, t_data.n_fvalues())));		
    for (index_t o = 0; o < t_data.n_outputs(); o ++)
    {
      LUT& lut = mluts[o][0];
      std::fill(lut.begin(), lut.end(), avg_outputs[o]);
    }

    // Finished
    return model.set(mluts);
  }	

}}
