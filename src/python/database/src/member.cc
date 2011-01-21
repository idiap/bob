/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 20 Jan 19:07:45 2011 
 *
 * @brief Python bindings to Relationset::Member's 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/Dataset2.h"

using namespace boost::python;
namespace db = Torch::core;

void bind_database_member() {
  class_<db::Member, boost::shared_ptr<db::Member> >("Member", "A Member defines a concrete entry in a relation ship bound by rules. Members point to array or whole arraysets.", init<>("Initializes a new member"))
    .add_property("arrayId", &db::Member::getArrayId, &db::Member::setArrayId)
    .add_property("arraysetId", &db::Member::getArraysetId, &db::Member::setArraysetId)
    ;
}
