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

void bind_database_rule() {
  class_<db::Rule, boost::shared_ptr<db::Rule> >("Rule", "A Rule describes restrictions on Array/Arrayset associations in a Dataset", init<optional<size_t, size_t> >((arg("min"), arg("max")), "Initializes a new rule"))
    .add_property("min", &db::Rule::getMin)
    .add_property("max", &db::Rule::getMax)
    ;
}
