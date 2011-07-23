/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 20 Jul 2011 19:29:20
 *
 * @brief Implements a JFAMachine
 */

#include <cmath>

#include "io/Arrayset.h"
#include "machine/JFAMachine.h"
#include "machine/Exception.h"
#include "math/linear.h"

namespace mach = Torch::machine;
namespace math = Torch::math;

mach::JFAMachine::JFAMachine():
  m_C(0), m_D(0), m_CD(0), m_ru(0), m_rv(0),
  m_ubm_mean(0),
  m_ubm_var(0),
  m_U(0,0),
  m_V(0,0),
  m_d(0),
  m_X(0,0),
  m_Y(0,0),
  m_Z(0,0)
{
}

mach::JFAMachine::JFAMachine(int C, int D, int ru, int rv):
  m_C(C), m_D(D), m_CD(C*D), m_ru(ru), m_rv(rv),
  m_ubm_mean(m_CD),
  m_ubm_var(m_CD),
  m_U(m_CD,ru),
  m_V(m_CD,rv),
  m_d(m_CD),
  m_X(ru,0),
  m_Y(rv,0),
  m_Z(m_CD,0)
{
}


mach::JFAMachine::JFAMachine(const mach::JFAMachine& other):
  m_C(other.m_C), m_D(other.m_D), m_CD(other.m_CD), 
  m_ru(other.m_ru), m_rv(other.m_rv),
  m_ubm_mean(other.m_ubm_mean.copy()),
  m_ubm_var(other.m_ubm_var.copy()),
  m_U(other.m_U.copy()),
  m_V(other.m_V.copy()),
  m_d(other.m_d.copy()),
  m_X(other.m_X.copy()),
  m_Y(other.m_Y.copy()),
  m_Z(other.m_Z.copy())
{
}

mach::JFAMachine::JFAMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::JFAMachine::~JFAMachine() {
}

mach::JFAMachine& mach::JFAMachine::operator=
(const mach::JFAMachine& other) {
  m_C = other.m_C;
  m_D = other.m_D;
  m_CD = other.m_CD;
  m_ru = other.m_ru;
  m_rv = other.m_rv;
  m_ubm_mean.reference(other.m_ubm_mean.copy());
  m_ubm_var.reference(other.m_ubm_var.copy());
  m_U.reference(other.m_U.copy());
  m_V.reference(other.m_V.copy());
  m_d.reference(other.m_d.copy());
  m_X.reference(other.m_X.copy());
  m_Y.reference(other.m_Y.copy());
  m_Z.reference(other.m_Z.copy());
  return *this;
}

void mach::JFAMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  uint32_t D = 0;
  config.read("D", D);
  m_D = static_cast<int>(D);
  m_ubm_mean.reference(config.readArray<double,1>("ubm_mean"));
  m_ubm_var.reference(config.readArray<double,1>("ubm_var"));
  m_U.reference(config.readArray<double,2>("U"));
  m_V.reference(config.readArray<double,2>("V"));
  m_d.reference(config.readArray<double,1>("d"));
  m_X.reference(config.readArray<double,2>("X"));
  m_Y.reference(config.readArray<double,2>("Y"));
  m_Z.reference(config.readArray<double,2>("Z"));
  m_CD = m_ubm_mean.extent(0);
  m_C = m_CD / m_D;
  m_ru = m_U.extent(1);
  m_rv = m_V.extent(1);
}

void mach::JFAMachine::save (Torch::io::HDF5File& config) const {
  config.set("D", static_cast<uint32_t>(m_D));
  config.setArray("ubm_mean", m_ubm_mean);
  config.setArray("ubm_var", m_ubm_var);
  config.setArray("U", m_U);
  config.setArray("V", m_V);
  config.setArray("d", m_d);
  config.setArray("X", m_X);
  config.setArray("Y", m_Y);
  config.setArray("Z", m_Z);
}


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

void mach::JFAMachine::setU(const blitz::Array<double,2>& U) {
  if(U.extent(0) != m_U.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(0), m_U.extent(0));
  }
  if(U.extent(1) != m_U.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(1), m_U.extent(1));
  }
  m_U.reference(U.copy());
}

void mach::JFAMachine::setV(const blitz::Array<double,2>& V) {
  if(V.extent(0) != m_V.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(0), m_V.extent(0));
  }
  if(V.extent(1) != m_V.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(1), m_V.extent(1));
  }
  m_V.reference(V.copy());
}

void mach::JFAMachine::setD(const blitz::Array<double,1>& d) {
  if(d.extent(0) != m_d.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(d.extent(0), m_d.extent(0));
  }
  m_d.reference(d.copy());
}

