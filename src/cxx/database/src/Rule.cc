/**
 * @file src/Rule.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Declares the Dataset Rules
 */

#include "Database/Rule.h"

Rule::Rule(): 
  m_arraysetrole(""), m_min(1), m_max(1) { }

Rule::~Rule() {
  TDEBUG3("Rule destructor (Arrayset-role: " << getArraysetRole() << ")");
}
