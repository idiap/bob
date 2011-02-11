/**
 * @file src/Rule.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Declares the Dataset Rules
 */

#include "database/Rule.h"
#include "database/Relation.h"

namespace db = Torch::database;

db::Rule::Rule (const std::string& role, size_t min, size_t max) :
  m_role(role),
  m_min(min),
  m_max(max)
{
}

db::Rule::~Rule() { }
