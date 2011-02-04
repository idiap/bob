/**
 * @file src/Member.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief  
 */

#include "database/Member.h"

Member::Member(): 
  m_array_id(0), m_arrayset_id(0) { }

Member::~Member() {
  TDEBUG3("Member destructor (id: " << getArrayId() << "-" << 
    getArraysetId() << ")");
}
