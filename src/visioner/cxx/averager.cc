/**
 * @file visioner/cxx/averager.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bob/core/logging.h"

#include "bob/visioner/model/trainers/averager.h"
#include "bob/visioner/util/timer.h"

namespace bob { namespace visioner {

  // Constructor
  Averager::Averager(const param_t& param)
    :	Trainer(param)

  {		
  }

  // Train a model using the given training and validation samples
  bool Averager::train(	
      const Sampler& t_sampler, const Sampler&,
      Model& model, size_t threads)
  {
    Timer timer;

    // Sample uniformly the training data
    std::vector<uint64_t> t_samples;                
    DataSet t_data;

    t_sampler.sample(m_param.m_train_samples, t_samples, threads);
    std::sort(t_samples.begin(), t_samples.end());

    t_sampler.map(t_samples, model, t_data, threads);

    bob::core::info << "timing: sampling ~ " << timer.elapsed() << "." << std::endl;

    // Check parameters
    if (t_data.empty())
    {
      bob::core::error << "Invalid training samples!" << std::endl;
      return false;
    }

    bob::core::info << "using "
      << t_data.n_samples() << " training samples with "
      << t_data.n_features() << " features." << std::endl;

    // Train the model ...		
    std::vector<double> avg_outputs(t_data.n_outputs(), 0.0);
    for (uint64_t s = 0; s < t_data.n_samples(); s ++)
    {
      for (uint64_t o = 0; o < t_data.n_outputs(); o ++)
      {
        avg_outputs[o] += t_data.targets()(s, o);
      }
    }

    for (uint64_t o = 0; o < t_data.n_outputs(); o ++)
    {
      avg_outputs[o] *= inverse(t_data.n_samples());

      bob::core::info
        << "output [" << (o + 1) << "/" << t_data.n_outputs() << "] = "
        << avg_outputs[o] << "." << std::endl;
    }

    std::vector<std::vector<LUT> > mluts(t_data.n_outputs(), std::vector<LUT>(1, LUT(0, t_data.n_fvalues())));		
    for (uint64_t o = 0; o < t_data.n_outputs(); o ++)
    {
      LUT& lut = mluts[o][0];
      std::fill(lut.begin(), lut.end(), avg_outputs[o]);
    }

    // Finished
    return model.set(mluts);
  }	

}}
