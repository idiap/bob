/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 21 Jan 08:35:51 2011 
 *
 * @brief Bindings to Dataset::Relationset 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "database/Relationset.h"

using namespace boost::python;
namespace db = Torch::database;

static tuple get_relation_ids(const db::Relationset& rs) {
  list l;
  for (std::map<size_t, boost::shared_ptr<db::Relation> >::const_iterator it=rs.relations().begin(); it!=rs.relations().end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_relations(const db::Relationset& rs) {
  list l;
  for (std::map<size_t, boost::shared_ptr<db::Relation> >::const_iterator it=rs.relations().begin(); it!=rs.relations().end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static tuple get_roles(const db::Relationset& rs) {
  list l;
  for (std::map<std::string, boost::shared_ptr<db::Rule> >::const_iterator it=rs.rules().begin(); it!=rs.rules().end(); ++it) {
    l.append(it->first);
  }
  return tuple(l);
}

static tuple get_rules(const db::Relationset& rs) {
  list l;
  for (std::map<std::string, boost::shared_ptr<db::Rule> >::const_iterator it=rs.rules().begin(); it!=rs.rules().end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static void pythonic_set_relation (db::Relationset& rs, size_t id, boost::shared_ptr<const db::Relation> obj) {
  if (rs.exists(id)) rs.set(id, obj);
  else rs.add(id, obj);
}

static void pythonic_set_rule (db::Relationset& rs, const std::string& role, boost::shared_ptr<const db::Rule> obj) {
  if (rs.exists(role)) rs.set(role, obj);
  else rs.add(role, obj);
}

void bind_database_relationset() {
  class_<db::Relationset, boost::shared_ptr<db::Relationset> >("Relationset", "A Relationset describes groupings of Array/Arraysets in a Dataset.", init<>("Builds a new Relationset."))
    .def("consolidateIds", &db::Relationset::consolidateIds, "Re-writes the ids of every relation so they are numbered sequentially and by the order of insertion.")
    .def("clearRelations", &db::Relationset::clearRelations, "Removes all relations from this set.")
    .def("clearRules", &db::Relationset::clearRules, "Removes all rules from this set, if there are no relations.")

    .def("ids", &get_relation_ids, "All Relation ids in this Relationset")    
    .def("relations", &get_relations, "All Relation's in this Relationset")    
    .def("roles", &get_roles, "All roles described in this Relationset")
    .def("rules", &get_rules, "All rules described in this Relationset")
    
    //some manipulations
    .def("append", (size_t (db::Relationset::*)(boost::shared_ptr<const db::Relation>))&db::Relationset::add, (arg("self"), arg("relation")))

    //some dictionary-like manipulations
    .def("__getitem__", (boost::shared_ptr<db::Relation> (db::Relationset::*)(const size_t))&db::Relationset::ptr, (arg("self"), arg("relation_id")))
    .def("__getitem__", (boost::shared_ptr<db::Rule> (db::Relationset::*)(const std::string&))&db::Relationset::ptr, (arg("self"), arg("rule_role")))
    .def("__setitem__", &pythonic_set_relation)
    .def("__setitem__", &pythonic_set_rule)
    ;
}
