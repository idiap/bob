/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue 11 oct 2011
 *
 * @brief Machines that implements the PLDA model
 */

#include "core/array_assert.h"
#include "machine/PLDAMachine.h"


namespace mach = Torch::machine;
namespace tca = Torch::core::array;

mach::PLDABaseMachine::PLDABaseMachine():
  m_F(0,0), m_G(0,0), m_sigma(0), m_mu(0)
{
}

mach::PLDABaseMachine::PLDABaseMachine(const size_t d, const size_t nf,
    const size_t ng):
  m_F(d,nf), m_G(d,ng), m_sigma(d), m_mu(d)
{
}


mach::PLDABaseMachine::PLDABaseMachine(const mach::PLDABaseMachine& other):
  m_F(Torch::core::array::ccopy(other.m_F)), 
  m_G(Torch::core::array::ccopy(other.m_G)), 
  m_sigma(Torch::core::array::ccopy(other.m_sigma)), 
  m_mu(Torch::core::array::ccopy(other.m_mu))
{
}

mach::PLDABaseMachine::PLDABaseMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::PLDABaseMachine::~PLDABaseMachine() {
}

mach::PLDABaseMachine& mach::PLDABaseMachine::operator=
    (const mach::PLDABaseMachine& other) 
{
  m_F.reference(Torch::core::array::ccopy(other.m_F));
  m_G.reference(Torch::core::array::ccopy(other.m_G));
  m_sigma.reference(Torch::core::array::ccopy(other.m_sigma));
  m_mu.reference(Torch::core::array::ccopy(other.m_mu));
  return *this;
}

void mach::PLDABaseMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_F.reference(config.readArray<double,2>("F"));
  m_G.reference(config.readArray<double,2>("G"));
  m_sigma.reference(config.readArray<double,1>("sigma"));
  m_mu.reference(config.readArray<double,1>("mu"));
}

void mach::PLDABaseMachine::save(Torch::io::HDF5File& config) const {
  config.setArray("F", m_F);
  config.setArray("G", m_G);
  config.setArray("sigma", m_sigma);
  config.setArray("mu", m_mu);
}

void mach::PLDABaseMachine::resize(const size_t d, const size_t nf, 
    const size_t ng) 
{
  m_F.resize(d,nf);
  m_G.resize(d,ng);
  m_sigma.resize(d);
  m_mu.resize(d);
}

void mach::PLDABaseMachine::setF(const blitz::Array<double,2>& F) {
  tca::assertSameShape(F, m_F);
  m_F.reference(Torch::core::array::ccopy(F));
}

void mach::PLDABaseMachine::setG(const blitz::Array<double,2>& G) {
  tca::assertSameShape(G, m_G);
  m_G.reference(Torch::core::array::ccopy(G));
}

void mach::PLDABaseMachine::setSigma(const blitz::Array<double,1>& sigma) {
  tca::assertSameShape(sigma, m_sigma);
  m_sigma.reference(Torch::core::array::ccopy(sigma));
}

void mach::PLDABaseMachine::setMu(const blitz::Array<double,1>& mu) {
  tca::assertSameShape(mu, m_mu);
  m_mu.reference(Torch::core::array::ccopy(mu));
}
