/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 20 Jan 19:07:45 2011 
 *
 * @brief Python bindings to Relationset::Relation's 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;

static tuple get_members(const db::Relation& r) {
  list l;
  for (db::Relation::const_iterator it=r.begin(); it!=r.end(); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

static tuple get_members_with_role(const db::Relation& r, const char* role=0) {
  list l;
  for (db::Relation::const_iterator_b it=r.begin(role); it!=r.end(role); ++it) {
    l.append(it->second);
  }
  return tuple(l);
}

void bind_database_relation() {
  class_<db::Relation, boost::shared_ptr<db::Relation> >("Relation", "A Relation describes a single grouping of Array/Arraysets in a RelationSet.", no_init)
    .def("addMember", &db::Relation::addMember, (arg("self"), arg("member")), "Adds a new member to this relation")
    .add_property("id", &db::Relation::getId, &db::Relation::setId)
    .add_property("members", &get_members, "All members of this Relation")
    .def("getMembersWithRole", &get_members_with_role, (arg("self"), arg("role")), "Returns all members with a certain role in this relation")
    ;
}
