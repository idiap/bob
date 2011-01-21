/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 20 Jan 19:07:45 2011 
 *
 * @brief Python bindings to Relationset::Rule's 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;

static const char* get_role(const db::Rule& r) {
  return r.getArraysetRole().c_str();
}

static void set_role(db::Rule& r, const char* name) {
  std::string n(name);
  r.setArraysetRole(n);
}


void bind_database_rule() {
  class_<db::Rule, boost::shared_ptr<db::Rule> >("Rule", "A Rule describes restrictions on Array/Arrayset associations in a Dataset", init<>("Initializes a new rule"))
    .add_property("role", &get_role, &set_role)
    .add_property("min", &db::Rule::getMin, &db::Rule::setMin)
    .add_property("max", &db::Rule::getMax, &db::Rule::setMax)
    ;
}
