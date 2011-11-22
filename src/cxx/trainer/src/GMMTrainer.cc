/**
 * @file cxx/trainer/src/GMMTrainer.cc
 * @date Tue May 10 11:35:58 2011 +0200
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
#include "trainer/GMMTrainer.h"

using namespace Torch::machine;

Torch::trainer::GMMTrainer::GMMTrainer(bool update_means, bool update_variances, bool update_weights, 
    double mean_var_update_responsibilities_threshold):
  EMTrainer<GMMMachine, Torch::io::Arrayset>(), update_means(update_means), update_variances(update_variances), 
  update_weights(update_weights), m_mean_var_update_responsibilities_threshold(mean_var_update_responsibilities_threshold) {

}

Torch::trainer::GMMTrainer::~GMMTrainer() {
  
}

void Torch::trainer::GMMTrainer::initialization(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data) {
  // Allocate memory for the sufficient statistics and initialise
  m_ss.resize(gmm.getNGaussians(),gmm.getNInputs());
}

double Torch::trainer::GMMTrainer::eStep(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data) {
  m_ss.init();
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, m_ss);
  return m_ss.log_likelihood / m_ss.T;
}
