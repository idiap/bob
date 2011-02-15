/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 20 Jan 19:07:45 2011 
 *
 * @brief Python bindings to Relationset::Rule's 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "database/Rule.h"

using namespace boost::python;
namespace db = Torch::database;

static const char* get_role(const db::Rule& r) {
  return r.getRole().c_str();
}

void bind_database_rule() {
  class_<db::Rule, boost::shared_ptr<db::Rule> >("Rule", "A Rule describes restrictions on Array/Arrayset associations in a Dataset", init<const std::string&, optional<size_t, size_t> >((arg("role"), arg("min"), arg("max")), "Initializes a new rule"))
    .add_property("role", &get_role)
    .add_property("min", &db::Rule::getMin)
    .add_property("max", &db::Rule::getMax)
    ;
}
