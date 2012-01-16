/**
 * @file src/cxx/trainer/src/GMMTrainer.cc
 * @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
 * @author <a href="mailto:Francois.Moulin@idiap.ch">Francois Moulin</a>
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
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

using namespace bob::machine;

bob::trainer::GMMTrainer::GMMTrainer(bool update_means, bool update_variances, bool update_weights, 
    double mean_var_update_responsibilities_threshold):
  EMTrainerNew<GMMMachine, bob::io::Arrayset>(), update_means(update_means), update_variances(update_variances), 
  update_weights(update_weights), m_mean_var_update_responsibilities_threshold(mean_var_update_responsibilities_threshold) {

}

bob::trainer::GMMTrainer::~GMMTrainer() {
  
}

void bob::trainer::GMMTrainer::initialization(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data) {
  // Allocate memory for the sufficient statistics and initialise
  m_ss.resize(gmm.getNGaussians(),gmm.getNInputs());
}

void bob::trainer::GMMTrainer::eStep(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data) {
  m_ss.init();
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, m_ss);
}

double bob::trainer::GMMTrainer::computeLikelihood(bob::machine::GMMMachine& gmm) {
  return m_ss.log_likelihood / m_ss.T;
}

void bob::trainer::GMMTrainer::finalization(bob::machine::GMMMachine& gmm, const bob::io::Arrayset& data) {
}
