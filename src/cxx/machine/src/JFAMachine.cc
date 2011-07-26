/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 20 Jul 2011 19:29:20
 *
 * @brief Implements a JFABaseMachine
 */

#include <cmath>

#include "io/Arrayset.h"
#include "machine/JFAMachine.h"
#include "machine/Exception.h"
#include "math/linear.h"

namespace mach = Torch::machine;
namespace math = Torch::math;

mach::JFABaseMachine::JFABaseMachine():
  m_ubm(boost::shared_ptr<Torch::machine::GMMMachine>()), m_ru(0), m_rv(0),
  m_U(0,0),
  m_V(0,0),
  m_d(0)
{
}

mach::JFABaseMachine::JFABaseMachine(const boost::shared_ptr<Torch::machine::GMMMachine> ubm, 
    int ru, int rv):
  m_ubm(ubm), m_ru(ru), m_rv(rv),
  m_U(getDimC()*getDimD(),ru),
  m_V(getDimC()*getDimD(),rv),
  m_d(getDimC()*getDimD())
{
}


mach::JFABaseMachine::JFABaseMachine(const mach::JFABaseMachine& other):
  m_ubm(other.m_ubm), m_ru(other.m_ru), m_rv(other.m_rv),
  m_U(other.m_U.copy()),
  m_V(other.m_V.copy()),
  m_d(other.m_d.copy())
{
}

mach::JFABaseMachine::JFABaseMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::JFABaseMachine::~JFABaseMachine() {
}

mach::JFABaseMachine& mach::JFABaseMachine::operator=
(const mach::JFABaseMachine& other) {
  m_ubm = other.m_ubm;
  m_ru = other.m_ru;
  m_rv = other.m_rv;
  m_U.reference(other.m_U.copy());
  m_V.reference(other.m_V.copy());
  m_d.reference(other.m_d.copy());
  return *this;
}

void mach::JFABaseMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_U.reference(config.readArray<double,2>("U"));
  m_V.reference(config.readArray<double,2>("V"));
  m_d.reference(config.readArray<double,1>("d"));
  m_ru = m_U.extent(1);
  m_rv = m_V.extent(1);
}

void mach::JFABaseMachine::save(Torch::io::HDF5File& config) const {
  config.setArray("U", m_U);
  config.setArray("V", m_V);
  config.setArray("d", m_d);
}

/*
void mach::JFAMachine::setUbmMean(const blitz::Array<double,1>& mean) {
  if(mean.extent(0) != m_ubm_mean.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(mean.extent(0), m_ubm_mean.extent(0));
  }
  m_ubm_mean.reference(mean.copy());
}

void mach::JFAMachine::setUbmVar(const blitz::Array<double,1>& var) {
  if(var.extent(0) != m_ubm_var.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(var.extent(0), m_ubm_var.extent(0));
  }
  m_ubm_var.reference(var.copy());
}
*/
void mach::JFABaseMachine::setUbm(const boost::shared_ptr<Torch::machine::GMMMachine> ubm) {
  m_ubm = ubm;
}

void mach::JFABaseMachine::setU(const blitz::Array<double,2>& U) {
  if(U.extent(0) != m_U.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(0), m_U.extent(0));
  }
  if(U.extent(1) != m_U.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(1), m_U.extent(1));
  }
  m_U.reference(U.copy());
}

void mach::JFABaseMachine::setV(const blitz::Array<double,2>& V) {
  if(V.extent(0) != m_V.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(0), m_V.extent(0));
  }
  if(V.extent(1) != m_V.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(1), m_V.extent(1));
  }
  m_V.reference(V.copy());
}

void mach::JFABaseMachine::setD(const blitz::Array<double,1>& d) {
  if(d.extent(0) != m_d.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(d.extent(0), m_d.extent(0));
  }
  m_d.reference(d.copy());
}



mach::JFAMachine::JFAMachine():
  m_jfa_base(boost::shared_ptr<Torch::machine::JFABaseMachine>()),
  m_y(0),
  m_z(0)
{
}

mach::JFAMachine::JFAMachine(const boost::shared_ptr<Torch::machine::JFABaseMachine> jfa_base): 
  m_jfa_base(jfa_base),
  m_y(0),
  m_z(0)
{
}


mach::JFAMachine::JFAMachine(const mach::JFAMachine& other):
  m_jfa_base(other.m_jfa_base),
  m_y(other.m_y.copy()),
  m_z(other.m_z.copy())
{
}

mach::JFAMachine::JFAMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::JFAMachine::~JFAMachine() {
}

mach::JFAMachine& mach::JFAMachine::operator=
(const mach::JFAMachine& other) {
  m_jfa_base = other.m_jfa_base;
  m_y.reference(other.m_y.copy());
  m_z.reference(other.m_z.copy());
  return *this;
}

void mach::JFAMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_y.reference(config.readArray<double,1>("y"));
  m_z.reference(config.readArray<double,1>("z"));
}

void mach::JFAMachine::save(Torch::io::HDF5File& config) const {
  config.setArray("y", m_y);
  config.setArray("z", m_z);
}

void mach::JFAMachine::setJFABase(const boost::shared_ptr<Torch::machine::JFABaseMachine> jfa_base) {
  m_jfa_base = jfa_base;
}

void mach::JFAMachine::setY(const blitz::Array<double,1>& y) {
  if(y.extent(0) != m_y.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(y.extent(0), m_y.extent(0));
  }
  m_y.reference(y.copy());
}

void mach::JFAMachine::setZ(const blitz::Array<double,1>& z) {
  if(z.extent(0) != m_z.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(z.extent(0), m_z.extent(0));
  }
  m_z.reference(z.copy());
}

