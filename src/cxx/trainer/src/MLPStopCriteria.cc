/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 16:08:06 CEST
 *
 * @brief Implementation of several stopping criterias for MLP training.
 */

#include "trainer/MLPStopCriteria.h"

namespace train = Torch::trainer;

train::NumberOfIterationsCriteria::NumberOfIterationsCriteria(size_t n):
  m_n(n)
{
}

train::NumberOfIterationsCriteria::~NumberOfIterationsCriteria() { }

bool train::NumberOfIterationsCriteria::operator()
  (const Torch::machine::MLP&, size_t iteration) const {
    return (iteration > m_n);
  }
