/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings to the database transcoding functionality 
 */

#include <boost/python.hpp>
#include "database/transcode.h"

using namespace boost::python;
namespace db = Torch::database;

void bind_database_transcode() {
  def("array_transcode", (void (*)(const std::string&, const std::string&))&db::array_transcode, "Transcodes a file containing a single array to another file, possibly in a different format. The codecs to be used are derived from the filename extensions. In the event of readout errors or unsupported data types, exceptions will be thrown.");

  def("array_transcode", (void (*)(const std::string&, const std::string&, const std::string&, const std::string&))&db::array_transcode, "Transcodes a file containing a single array to another file, possibly in a different format. The codecs are defined by the codec names given as parameter. In the event of readout errors or unsupported data types, exceptions will be thrown.");

  def("arrayset_transcode", (void (*)(const std::string&, const std::string&))&db::arrayset_transcode, "Transcodes a file containing a arrayset to another file, possibly in a different format. The codecs to be used are derived from the filename extensions. In the event of readout errors or unsupported data types, exceptions will be thrown.");

  def("arrayset_transcode", (void (*)(const std::string&, const std::string&, const std::string&, const std::string&))&db::array_transcode, "Transcodes a file containing a arrayset to another file, possibly in a different format. The codecs are defined by the codec names given as parameter. In the event of readout errors or unsupported data types, exceptions will be thrown.");
}
