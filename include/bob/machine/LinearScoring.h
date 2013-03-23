/**
 * @file bob/machine/LinearScoring.h
 * @date Wed Jul 13 16:00:04 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#ifndef BOB_MACHINE_LINEARSCORING_H
#define BOB_MACHINE_LINEARSCORING_H

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <bob/machine/GMMMachine.h>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * Compute a matrix of scores using linear scoring.
 *
 * @warning Each GMM must have the same size.
 * 
 * @param models        list of mean supervector for the client models
 * @param ubm_mean      mean supervector of the world model
 * @param ubm_variance  variance supervector of the world model
 * @param test_stats    list of accumulate statistics for each test trial
 * @param test_channelOffset  list of channel offset if any (for JFA/ISA for instance)
 * @param frame_length_normalisation   perform a normalisation by the number of feature vectors
 * @param[out] scores 2D matrix of scores, <tt>scores[m, s]</tt> is the score for model @c m against statistics @c s
 * @warning the output scores matrix should have the correct size (number of models x number of test_stats)
 */
void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                   const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const std::vector<blitz::Array<double, 1> >& test_channelOffset,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores);
void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                   const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores);

/**
 * Compute a matrix of scores using linear scoring.
 *
 * @warning Each GMM must have the same size.
 * 
 * @param models      list of client models as GMMMachines
 * @param ubm         world model as a GMMMachine
 * @param test_stats  list of accumulate statistics for each test trial
 * @param frame_length_normalisation   perform a normalisation by the number of feature vectors
 * @param[out] scores 2D matrix of scores, <tt>scores[m, s]</tt> is the score for model @c m against statistics @c s
 * @warning the output scores matrix should have the correct size (number of models x number of test_stats)
 */
void linearScoring(const std::vector<boost::shared_ptr<const bob::machine::GMMMachine> >& models,
                   const bob::machine::GMMMachine& ubm,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores);
/**
 * Compute a matrix of scores using linear scoring.
 *
 * @warning Each GMM must have the same size.
 * 
 * @param models      list of client models as GMMMachines
 * @param ubm         world model as a GMMMachine
 * @param test_stats  list of accumulate statistics for each test trial
 * @param test_channelOffset  list of channel offset if any (for JFA/ISA for instance)
 * @param frame_length_normalisation   perform a normalisation by the number of feature vectors
 * @param[out] scores 2D matrix of scores, <tt>scores[m, s]</tt> is the score for model @c m against statistics @c s
 * @warning the output scores matrix should have the correct size (number of models x number of test_stats)
 */
void linearScoring(const std::vector<boost::shared_ptr<const bob::machine::GMMMachine> >& models,
                   const bob::machine::GMMMachine& ubm,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const std::vector<blitz::Array<double, 1> >& test_channelOffset,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores);

/**
 * @}
 */
}}

#endif // BOB_MACHINE_LINEARSCORING_H
