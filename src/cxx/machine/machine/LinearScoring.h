/**
 * @file cxx/machine/machine/LinearScoring.h
 * @date Wed Jul 13 16:00:04 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#ifndef TORCH5SPRO_MACHINE_LINEARSCORING_H
#define TORCH5SPRO_MACHINE_LINEARSCORING_H

#include <blitz/array.h>
#include <vector>
#include "machine/GMMMachine.h"

namespace Torch { namespace machine {

  void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                     const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                     const std::vector<const Torch::machine::GMMStats*>& test_stats,
                     const std::vector<blitz::Array<double, 1> >& test_channelOffset,
                     const bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores);
  void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                     const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                     const std::vector<const Torch::machine::GMMStats*>& test_stats,
                     const bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores);
  void linearScoring(const std::vector<const Torch::machine::GMMMachine*>& models,
                     const Torch::machine::GMMMachine& ubm,
                     const std::vector<const Torch::machine::GMMStats*>& test_stats,
                     const bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores);

  /**
   * Compute a matrix of scores using linear scoring.
   *
   * @warning Each GMM must have the same size.
   * 
   * @param models      list of client models
   * @param ubm         world model
   * @param test_stats  list of accumulate statistics for each test trial
   * @param test_channelOffset 
   * @param frame_length_normlisation perform a normalization by the number of feature vectors
   * @param[out] scores 2D matrix of scores, <tt>scores[m, s]</tt> is the score for model @c m against statistics @c s
   */
  void linearScoring(std::vector<Torch::machine::GMMMachine*>& models,
                     Torch::machine::GMMMachine& ubm,
                     std::vector<Torch::machine::GMMStats*>& test_stats,
                     blitz::Array<double, 2>* test_channelOffset,
                     bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores);
}
}

#endif // TORCH5SPRO_MACHINE_LINEARSCORING_H
