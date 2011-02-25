/**
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_database_array();
void bind_database_arrayset();
void bind_database_dataset();
void bind_database_exception();
void bind_database_rule();
void bind_database_relation();
void bind_database_relationset();
void bind_database_binfile();
void bind_database_codec();
void bind_database_transcode();
void bind_database_datetime();

BOOST_PYTHON_MODULE(libpytorch_database) {
  scope().attr("__doc__") = "Torch classes and sub-classes for database access";
  bind_database_array();
  bind_database_arrayset();
  bind_database_dataset();
  bind_database_exception();
  bind_database_rule();
  bind_database_relation();
  bind_database_relationset();
  bind_database_binfile();
  bind_database_codec();
  bind_database_transcode();
  bind_database_datetime();
}
