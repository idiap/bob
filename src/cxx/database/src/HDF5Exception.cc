/**
 * @file database/src/HDF5Exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements exceptions that can be thrown by the HDF5 support code.
 */

#include <boost/format.hpp>
#include "database/HDF5Exception.h"
#include "database/HDF5Types.h"

namespace db = Torch::database;

db::HDF5Exception::HDF5Exception() throw()
{
}

db::HDF5Exception::~HDF5Exception() throw() { }

const char* db::HDF5Exception::what() const throw() {
  static const char* emergency = "Generic HDF5 exception. You should never see this!";
  return emergency;
}

db::HDF5UnsupportedCxxTypeError::HDF5UnsupportedCxxTypeError() throw():
  db::HDF5Exception()
{
}

db::HDF5UnsupportedCxxTypeError::~HDF5UnsupportedCxxTypeError() throw() { }

const char* db::HDF5UnsupportedCxxTypeError::what() const throw() {
  try {
    boost::format message("The C++ type given is not supported by our HDF5 interface");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5UnsupportedTypeError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5UnsupportedTypeError::HDF5UnsupportedTypeError(const boost::shared_ptr<hid_t>&) throw():
  db::HDF5Exception()
{
}

db::HDF5UnsupportedTypeError::HDF5UnsupportedTypeError() throw():
  db::HDF5Exception()
{
}

db::HDF5UnsupportedTypeError::~HDF5UnsupportedTypeError() throw() { }

const char* db::HDF5UnsupportedTypeError::what() const throw() {
  try {
    boost::format message("The C++ type given is not supported by our HDF5 interface");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5UnsupportedTypeError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5UnsupportedDimensionError::HDF5UnsupportedDimensionError(size_t n_dim) 
  throw():
  db::HDF5Exception(), 
  m_n_dim(n_dim)
{
}

db::HDF5UnsupportedDimensionError::~HDF5UnsupportedDimensionError() throw() { }

const char* db::HDF5UnsupportedDimensionError::what() const throw() {
  try {
    boost::format message("Got an array with %u dimensions, but we only support up to %d dimensions");
    message % m_n_dim % Torch::core::array::N_MAX_DIMENSIONS_ARRAY;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5UnsupportedDimensionError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5InvalidPath::HDF5InvalidPath(const std::string& filename,
    const std::string& path) throw():
  db::HDF5Exception(),
  m_filename(filename),
  m_path(path)
{
}

db::HDF5InvalidPath::~HDF5InvalidPath() throw() { }

const char* db::HDF5InvalidPath::what() const throw() {
  try {
    boost::format message("Cannot find path '%s' in the HDF5 file '%s'");
    message % m_path % m_filename;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5InvalidPath: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5InvalidFileAccessModeError::HDF5InvalidFileAccessModeError
(const unsigned int mode) throw():
  db::HDF5Exception(),
  m_mode(mode)
{
}

db::HDF5InvalidFileAccessModeError::~HDF5InvalidFileAccessModeError() throw() { }

const char* db::HDF5InvalidFileAccessModeError::what() const throw() {
  try {
    boost::format message("Trying to use an undefined access mode '%d'");
    message % m_mode;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5InvalidFileAccessModeError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5StatusError::HDF5StatusError(const std::string& call, herr_t status)
  throw():
    db::HDF5Exception(),
    m_call(call),
    m_status(status)
{
}

db::HDF5StatusError::~HDF5StatusError() throw() { }

static std::string format_hdf5_error() {
  const std::vector<std::string>& stack = db::HDF5Error::get(); 
  std::ostringstream retval;
  std::string prefix(" ");
  if (stack.size()) retval << prefix << stack[0];
  for (size_t i=1; i<stack.size(); ++i) 
    retval << std::endl << prefix << stack[i];
  db::HDF5Error::clear();
  return retval.str();
}

const char* db::HDF5StatusError::what() const throw() {
  try {
    boost::format message("Call to HDF5 C-function '%s' returned '%d'. HDF5 error statck follows:\n%s");
    message % m_call % m_status % format_hdf5_error();
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5StatusError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5IndexError::HDF5IndexError(const std::string& filename, 
    const std::string& dataset, size_t size, size_t asked_for) throw():
  db::HDF5Exception(),
  m_filename(filename),
  m_dataset(dataset),
  m_size(size),
  m_asked_for(asked_for)
{
}

db::HDF5IndexError::~HDF5IndexError() throw() { }

const char* db::HDF5IndexError::what() const throw() {
  try {
    boost::format message("Trying to access element %d in Dataset '%s' at HDF5 file '%s' that only contains %d elements");
    message % m_asked_for % m_dataset % m_filename % m_size;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5IndexError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5IncompatibleIO::HDF5IncompatibleIO(const std::string& filename, 
    const std::string& dataset, 
    const std::string& supported,
    const std::string& user_input) throw():
  db::HDF5Exception(),
  m_filename(filename),
  m_dataset(dataset),
  m_supported(supported),
  m_user_input(user_input)
{
}

db::HDF5IncompatibleIO::~HDF5IncompatibleIO() throw() { }

const char* db::HDF5IncompatibleIO::what() const throw() {
  try {
    boost::format message("Trying to read or write '%s' at dataset '%s' on HDF5 file '%s' that only accepts '%s'");
    message % m_user_input % m_dataset % m_filename % m_supported;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5IndexError: cannot format, exception raised";
    return emergency;
  }
}

db::HDF5NotExpandible::HDF5NotExpandible(const std::string& filename, 
    const std::string& dataset) throw():
  db::HDF5Exception(),
  m_filename(filename),
  m_dataset(dataset)
{
}

db::HDF5NotExpandible::~HDF5NotExpandible() throw() { }

const char* db::HDF5NotExpandible::what() const throw() {
  try {
    boost::format message("Trying to append to dataset '%s' on HDF5 file '%s' that is not expandible");
    message % m_dataset % m_filename;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::HDF5NotExpandible: cannot format, exception raised";
    return emergency;
  }
}
