/**
 * @file machine/cxx/GMMLLRMachine.cc
 * @date Fri Jul 8 13:01:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include "bob/machine/GMMLLRMachine.h"
#include "bob/machine/Exception.h"

bob::machine::GMMLLRMachine::GMMLLRMachine(bob::io::HDF5File& config) {
  load(config);
}

bob::machine::GMMLLRMachine::GMMLLRMachine(bob::io::HDF5File& client, bob::io::HDF5File& ubm) {
  m_gmm_client = new GMMMachine();
  m_gmm_client->load(client);
  m_gmm_ubm = new GMMMachine();
  m_gmm_ubm->load(ubm);

  // check and assign n_inputs
  if(m_gmm_client->getNInputs() != m_gmm_ubm->getNInputs())
    throw NInputsMismatch(m_gmm_client->getNInputs(), m_gmm_ubm->getNInputs());
  m_n_inputs = m_gmm_client->getNInputs();
}

bob::machine::GMMLLRMachine::GMMLLRMachine(const bob::machine::GMMMachine& client, const bob::machine::GMMMachine& ubm) {
  // check and assign n_inputs
  if(client.getNInputs() != ubm.getNInputs())
    throw NInputsMismatch(client.getNInputs(), ubm.getNInputs());
  m_n_inputs = client.getNInputs();

  m_gmm_client = new GMMMachine();
  *m_gmm_client = client;
  m_gmm_ubm = new GMMMachine();
  *m_gmm_ubm = ubm;
}

bob::machine::GMMLLRMachine::GMMLLRMachine(const GMMLLRMachine& other): Machine<blitz::Array<double,1>, double>(other) {
  copy(other);
}

bob::machine::GMMLLRMachine & bob::machine::GMMLLRMachine::operator= (const GMMLLRMachine &other) {
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // by convention, always return *this
  return *this;
}

bool bob::machine::GMMLLRMachine::operator==(const bob::machine::GMMLLRMachine& b) const {
  if(m_n_inputs != b.m_n_inputs) {
    return false;
  }

  if(m_gmm_client != b.m_gmm_client)
    return false;

  if(m_gmm_ubm != b.m_gmm_ubm)
    return false;

  return true;
}

void bob::machine::GMMLLRMachine::copy(const GMMLLRMachine& other) {
  m_n_inputs = other.m_n_inputs;

  // Initialize GMMMachines
  *m_gmm_client = *(other.m_gmm_client);
  *m_gmm_ubm = *(other.m_gmm_ubm);
}

bob::machine::GMMLLRMachine::~GMMLLRMachine() {
  delete m_gmm_client;
  delete m_gmm_ubm;
}

int bob::machine::GMMLLRMachine::getNInputs() const {
  return m_n_inputs;
}

void bob::machine::GMMLLRMachine::forward(const blitz::Array<double,1>& input, double& output) const {
  if (input.extent(0) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }
  forward_(input,output);
}

void bob::machine::GMMLLRMachine::forward_(const blitz::Array<double,1>& input, double& output) const {
  double s_u;
  m_gmm_client->forward(input,output);
  m_gmm_ubm->forward(input, s_u);
  output -= s_u;
}

bob::machine::GMMMachine* bob::machine::GMMLLRMachine::getGMMClient() const {
  return m_gmm_client;
}

bob::machine::GMMMachine* bob::machine::GMMLLRMachine::getGMMUBM() const {
  return m_gmm_ubm;
}

void bob::machine::GMMLLRMachine::save(bob::io::HDF5File& config) const {
  config.set("m_n_inputs", m_n_inputs);

  std::ostringstream oss_client;
  oss_client << "m_gmm_client";

  if (!config.hasGroup(oss_client.str())) config.createGroup(oss_client.str());
  config.cd(oss_client.str());
  m_gmm_client->save(config);
  config.cd("..");

  std::ostringstream oss_ubm;
  oss_ubm << "m_gmm_ubm";
  config.cd(oss_ubm.str());
  m_gmm_ubm->save(config);
  config.cd("..");
}

void bob::machine::GMMLLRMachine::load(bob::io::HDF5File& config) {
  m_n_inputs = config.read<int64_t>("m_n_inputs");

  std::ostringstream oss_client;
  oss_client << "m_gmm_client";
  config.cd(oss_client.str());
  m_gmm_client = new GMMMachine();
  m_gmm_client->load(config);
  config.cd("..");

  std::ostringstream oss_ubm;
  oss_ubm << "m_gmm_ubm";
  config.cd(oss_ubm.str());
  m_gmm_ubm = new GMMMachine();
  m_gmm_ubm->load(config);
  config.cd("..");
}

namespace bob {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const GMMLLRMachine& machine) {
      os << "n_inputs = " << machine.m_n_inputs << std::endl;
      os << "GMM Client: " << std::endl << machine.m_gmm_client;
      os << "GMM UBM: " << std::endl << machine.m_gmm_ubm;

      return os;
    }
  }
}
