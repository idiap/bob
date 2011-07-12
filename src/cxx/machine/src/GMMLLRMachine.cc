#include "machine/GMMLLRMachine.h"
#include "machine/Exception.h"

using namespace Torch::machine::Log;

Torch::machine::GMMLLRMachine::GMMLLRMachine(Torch::io::HDF5File& config) {
  load(config);
}

Torch::machine::GMMLLRMachine::GMMLLRMachine(Torch::io::HDF5File& client, Torch::io::HDF5File& ubm) {
  m_gmm_client = new GMMMachine();
  m_gmm_client->load(client);
  m_gmm_ubm = new GMMMachine();
  m_gmm_ubm->load(ubm);

  // check and assign n_inputs
  if(m_gmm_client->getNInputs() != m_gmm_ubm->getNInputs())
    throw NInputsMismatch(m_gmm_client->getNInputs(), m_gmm_ubm->getNInputs());
  m_n_inputs = m_gmm_client->getNInputs();
}

Torch::machine::GMMLLRMachine::GMMLLRMachine(const Torch::machine::GMMMachine& client, const Torch::machine::GMMMachine& ubm) {
  // check and assign n_inputs
  if(client.getNInputs() != ubm.getNInputs())
    throw NInputsMismatch(client.getNInputs(), ubm.getNInputs());
  m_n_inputs = client.getNInputs();

  m_gmm_client = new GMMMachine();
  *m_gmm_client = client;
  m_gmm_ubm = new GMMMachine();
  *m_gmm_ubm = ubm;
}

Torch::machine::GMMLLRMachine::GMMLLRMachine(const GMMLLRMachine& other): Machine<blitz::Array<double,1>, double>(other) {
  copy(other);
}

Torch::machine::GMMLLRMachine & Torch::machine::GMMLLRMachine::operator= (const GMMLLRMachine &other) {
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // by convention, always return *this
  return *this;
}

bool Torch::machine::GMMLLRMachine::operator==(const Torch::machine::GMMLLRMachine& b) const {
  if(m_n_inputs != b.m_n_inputs) {
    return false;
  }

  if(m_gmm_client != b.m_gmm_client)
    return false;

  if(m_gmm_ubm != b.m_gmm_ubm)
    return false;

  return true;
}

void Torch::machine::GMMLLRMachine::copy(const GMMLLRMachine& other) {
  m_n_inputs = other.m_n_inputs;

  // Initialize GMMMachines
  *m_gmm_client = *(other.m_gmm_client);
  *m_gmm_ubm = *(other.m_gmm_ubm);
}

Torch::machine::GMMLLRMachine::~GMMLLRMachine() {
  delete m_gmm_client;
  delete m_gmm_ubm;
}

int Torch::machine::GMMLLRMachine::getNInputs() const {
  return m_n_inputs;
}

void Torch::machine::GMMLLRMachine::forward(const blitz::Array<double,1>& input, double& output) const {
  if (input.extent(0) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }

  double s_u;
  m_gmm_client->forward(input,output);
  m_gmm_ubm->forward(input, s_u);
  output -= s_u;
}

Torch::machine::GMMMachine* Torch::machine::GMMLLRMachine::getGMMClient() const {
  return m_gmm_client;
}

Torch::machine::GMMMachine* Torch::machine::GMMLLRMachine::getGMMUBM() const {
  return m_gmm_ubm;
}

void Torch::machine::GMMLLRMachine::save(Torch::io::HDF5File& config) const {
  config.set("m_n_inputs", m_n_inputs);

  std::ostringstream oss_client;
  oss_client << "m_gmm_client";
  config.cd(oss_client.str());
  m_gmm_client->save(config);
  config.cd("..");

  std::ostringstream oss_ubm;
  oss_ubm << "m_gmm_ubm";
  config.cd(oss_ubm.str());
  m_gmm_ubm->save(config);
  config.cd("..");
}

void Torch::machine::GMMLLRMachine::load(Torch::io::HDF5File& config) {
  config.read("m_n_inputs", m_n_inputs);

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

namespace Torch {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const GMMLLRMachine& machine) {
      os << "n_inputs = " << machine.m_n_inputs << std::endl;
      os << "GMM Client: " << std::endl << machine.m_gmm_client;
      os << "GMM UBM: " << std::endl << machine.m_gmm_ubm;

      return os;
    }
  }
}
