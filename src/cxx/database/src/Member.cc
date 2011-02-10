/**
 * @file database/src/Member.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief  
 */

#include "database/Member.h"
#include "database/dataset_common.h"

namespace db = Torch::database;

db::Member::Member(size_t arrayset_id, size_t array_id) :
  m_arrayset_id(arrayset_id),
  m_array_id(array_id)
{
  if (!m_arrayset_id) throw db::IndexError();
}

db::Member::~Member() { }
