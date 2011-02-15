/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 20 Jan 19:07:45 2011 
 *
 * @brief Python bindings to Relationset::Relation's 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "database/Relation.h"

using namespace boost::python;
namespace db = Torch::database;

static tuple get_members(const db::Relation& r) {
  list l;
  for (size_t i=0; i<r.members().size(); ++i) {
    l.append(make_tuple(r[i].first, r[i].second));
  }
  return tuple(l);
}

//interface to add an iterable with 2 elements
template <typename T>
static void add_iterable(db::Relation& r, T t) {
  if (len(t) == 1) {
    r.add(extract<size_t>(t[0])());
    return;
  }
  else if (len(t) == 2) {
    r.add(extract<size_t>(t[0])(), extract<size_t>(t[1])());
    return;
  }

  //if you get here, just throw an error into python
  PyErr_SetString(PyExc_TypeError, "only accepts iterables (tuples or lists) with 1 or 2 elements representing arrayset-id and array-id respectively");
  throw_error_already_set();
}

//interface to add an iterable with 2 elements
template <typename T>
static void remove_iterable(db::Relation& r, T t) {
  if (len(t) == 1) {
    r.remove(extract<size_t>(t[0])());
    return;
  }
  else if (len(t) == 2) {
    r.remove(extract<size_t>(t[0])(), extract<size_t>(t[1])());
    return;
  }

  //if you get here, just throw an error into python
  PyErr_SetString(PyExc_TypeError, "only accepts iterables (tuples or lists) with 1 or 2 elements representing arrayset-id and array-id respectively");
  throw_error_already_set();
}

static tuple relation_getitem(const db::Relation& r, size_t index) {
  const std::pair<size_t,size_t>& p = r[index];
  return make_tuple(p.first, p.second);
}

template <typename T>
static void setitem(db::Relation& r, size_t index, T t) {
  if (len(t) == 1) {
    r.set(index, extract<size_t>(t[0])());
    return;
  }
  else if (len(t) == 2) {
    r.set(extract<size_t>(t[0])(), extract<size_t>(t[1])());
    return;
  }
  //if you get here, just throw an error into python
  PyErr_SetString(PyExc_TypeError, "only accepts iterables (tuples or lists) with 1 or 2 elements representing arrayset-id and array-id respectively");
  throw_error_already_set();
}

static void setitem2(db::Relation& r, size_t index, size_t asid) {
  r.set(index, asid);
}

static void setitem3(db::Relation& r, size_t index, size_t asid, size_t aid) {
  r.set(index, asid, aid);
}

static size_t length(const db::Relation& r) {
  return r.members().size();
}

void bind_database_relation() {
  class_<db::Relation, boost::shared_ptr<db::Relation> >("Relation", "A Relation describes a single grouping of Array/Arraysets in a RelationSet. The pythonic bindings for this class represent a Relation as a list of tuples (arrayset-id, array-id).", init<>("Constructs a relation that is uninitialized"))
    .add_property("id", &db::Relation::getId, &db::Relation::setId)
    .def("members", &get_members, "All members of this Relation")
    .def("add", (void (db::Relation::*)(size_t))&db::Relation::add, (arg("self"), arg("arrayset_id")), "Adds a new member to this relation that is an arrayset")
    .def("add", (void (db::Relation::*)(size_t, size_t))&db::Relation::add, (arg("self"), arg("arrayset_id"), arg("array_id")), "Adds a new member to this relation that is a specific array in an arrayset")
    .def("add", &add_iterable<tuple>, (arg("self"), arg("tuple")), "Adds a new member to this relation that is a specific array in an arrayset (tuple length == 2) or a whole arrayset (tuple length == 1)")
    .def("add", &add_iterable<list>, (arg("self"), arg("list")), "Adds a new member to this relation that is a specific array in an arrayset (list length == 2) or a whole arrayset (list length == 1)")
    .def("remove", (void (db::Relation::*)(size_t))&db::Relation::remove, (arg("self"), arg("arrayset_id")), "Removes a member from this relation that is an arrayset")
    .def("remove", (void (db::Relation::*)(size_t, size_t))&db::Relation::remove, (arg("self"), arg("arrayset_id"), arg("array_id")), "Removes a member from this relation that is a specific array in an arrayset")
    .def("remove", &remove_iterable<tuple>, (arg("self"), arg("tuple")), "Removes a member from this relation that is a specific array in an arrayset (tuple length == 2) or a whole arrayset (tuple length == 1)")
    .def("remove", &remove_iterable<list>, (arg("self"), arg("list")), "Removes a member from this relation that is a specific array in an arrayset (list length == 2) or a whole arrayset (list length == 1)")

    //mapping type operations
    .def("__getitem__", &relation_getitem, (arg("self"), arg("index")))
    .def("__setitem__", &setitem<tuple>, (arg("self"), arg("index"), arg("tuple")))
    .def("__setitem__", &setitem<list>, (arg("self"), arg("index"), arg("list")))
    .def("__setitem__", &setitem2, (arg("self"), arg("index"), arg("arrayset_id")))
    .def("__setitem__", &setitem3, (arg("self"), arg("index"), arg("arrayset_id"), arg("array_id")))
    .def("__delitem__", &db::Relation::erase, (arg("self"), arg("index")))
    .def("__len__", &length)
    ;
}
