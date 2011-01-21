/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 21 Jan 08:35:51 2011 
 *
 * @brief Bindings to Dataset::Relationset 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;

static tuple get_relation_ids(const db::Relationset& rs) {
  list l;
  for (db::Relationset::const_iterator it=rs.begin(); it!=rs.end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_relations(const db::Relationset& rs) {
  list l;
  for (db::Relationset::const_iterator it=rs.begin(); it!=rs.end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static tuple get_roles(const db::Relationset& rs) {
  list l;
  for (db::Relationset::rule_const_iterator it=rs.rule_begin(); it!=rs.rule_end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_rules(const db::Relationset& rs) {
  list l;
  for (db::Relationset::rule_const_iterator it=rs.rule_begin(); it!=rs.rule_end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static const char* get_name(const db::Relationset& rs) {
  return rs.getName().c_str();
}

static void set_name(db::Relationset& rs, const char* name) {
  std::string n(name);
  rs.setName(n);
}

void bind_database_relationset() {
  class_<db::Relationset, boost::shared_ptr<db::Relationset> >("Relationset", "A Relationset describes groupings of Array/Arraysets in a Dataset.", init<>("Builds a new Relationset."))
    .def("addRelation", &db::Relationset::addRelation)
    .def("addRule", &db::Relationset::addRule)
    .add_property("name", &get_name, &set_name)
    .def("relationIds", &get_relation_ids, "All Relation id's in this Relationset")    
    .def("relations", &get_relations, "All Relation's in this Relationset")    
    .def("roles", &get_roles, "All roles described in this Relationset")
    .def("rules", &get_rules, "All rules described in this Relationset")
    .def("__getitem__", &db::Relationset::getRelation)
    ;
}
