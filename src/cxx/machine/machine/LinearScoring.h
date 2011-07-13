#ifndef _LINEARSCORING_H
#define _LINEARSCORING_H

#include <blitz/array.h>
#include <machine/GMMMachine.h>
#include <vector>

namespace Torch { namespace machine {
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

#endif // _LINEARSCORING_H