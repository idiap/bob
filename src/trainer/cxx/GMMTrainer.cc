/**
 * @file trainer/cxx/GMMTrainer.cc
 * @date Tue May 10 11:35:58 2011 +0200
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

#include <bob/trainer/GMMTrainer.h>
#include <bob/core/assert.h>

bob::trainer::GMMTrainer::GMMTrainer(bool update_means, bool update_variances, bool update_weights, 
    double mean_var_update_responsibilities_threshold):
  EMTrainer<bob::machine::GMMMachine, blitz::Array<double,2> >(), update_means(update_means), update_variances(update_variances), 
  update_weights(update_weights), m_mean_var_update_responsibilities_threshold(mean_var_update_responsibilities_threshold) {

}

bob::trainer::GMMTrainer::~GMMTrainer() {
  
}

void bob::trainer::GMMTrainer::initialization(bob::machine::GMMMachine& gmm, const blitz::Array<double,2>& data) {
  // Allocate memory for the sufficient statistics and initialise
  m_ss.resize(gmm.getNGaussians(),gmm.getNInputs());
}

void bob::trainer::GMMTrainer::eStep(bob::machine::GMMMachine& gmm, const blitz::Array<double,2>& data) {
  m_ss.init();
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, m_ss);
}

double bob::trainer::GMMTrainer::computeLikelihood(bob::machine::GMMMachine& gmm) {
  return m_ss.log_likelihood / m_ss.T;
}

void bob::trainer::GMMTrainer::finalization(bob::machine::GMMMachine& gmm, const blitz::Array<double,2>& data) {
}

void bob::trainer::GMMTrainer::setGMMStats(const bob::machine::GMMStats& stats)
{
  bob::core::array::assertSameShape(m_ss.sumPx, stats.sumPx);
  m_ss = stats;
}
