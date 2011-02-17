/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 16 Feb 14:13:38 2011 
 *
 * @brief  
 */

#include <boost/format.hpp>
#include "database/Exception.h"

namespace db = Torch::database;
namespace core = Torch::core;
namespace array = Torch::core::array;

db::Exception::Exception() throw() {
}

db::Exception::~Exception() throw() {
}

const char* db::Exception::what() const throw() {
 static const char* what_string = "Generic database::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

db::NameError::NameError(const std::string& key) throw(): m_name(key) {
}

db::NameError::~NameError() throw() {
}

const char* db::NameError::what() const throw() {
  try {
    boost::format message("Cannot use key '%s'");
    message % m_name;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::NameError: cannot format, exception raised";
    return emergency;
  }
}

db::IndexError::IndexError(size_t id) throw(): m_id(id) {
}

db::IndexError::~IndexError() throw() {
}

const char* db::IndexError::what() const throw() {
  try {
    boost::format message("Cannot use index '%u'");
    message % m_id;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::IndexError: cannot format, exception raised";
    return emergency;
  }
}

db::DimensionError::DimensionError(size_t got, size_t expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

db::DimensionError::~DimensionError() throw() { }

const char* db::DimensionError::what() const throw() {
  try {
    boost::format message("Expected '%u' dimensions, but got '%u'");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::DimensionError: cannot format, exception raised";
    return emergency;
  }
}

db::TypeError::TypeError(Torch::core::array::ElementType got, 
    Torch::core::array::ElementType expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

db::TypeError::~TypeError() throw() { }

const char* db::TypeError::what() const throw() {
  try {
    boost::format message("Expected element type '%s', but got '%s'");
    message % array::stringize(m_expected) % array::stringize(m_got);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::TypeError: cannot format, exception raised";
    return emergency;
  }
}

db::NonExistingElement::NonExistingElement() { }

db::NonExistingElement::~NonExistingElement() throw() { }

const char* db::NonExistingElement::what() const throw() {
  static const char* message = "The type of element input is unknown";
  return message;
}

db::Uninitialized::Uninitialized() throw() { }
db::Uninitialized::~Uninitialized() throw() { }
const char* db::Uninitialized::what() const throw() {
  static const char* message = "The current Relationset you are trying to use is not properly initialized with rules and a dataset parent";
  return message;
}

db::AlreadyHasRelations::AlreadyHasRelations(size_t number) throw() :
  m_number(number)
{
}

db::AlreadyHasRelations::~AlreadyHasRelations() throw() { }

const char* db::AlreadyHasRelations::what() const throw() {
  try {
    boost::format message("You are trying to remove one or more rules from a Relationset that is filled with '%d' relations. First remove those!");
    message % m_number;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::AlreadyHasRelations: cannot format, exception raised";
    return emergency;
  }
}

db::InvalidRelation::InvalidRelation() throw() { }
db::InvalidRelation::~InvalidRelation() throw() { }
const char* db::InvalidRelation::what() const throw() {
  static const char* message = "database::InvalidRelation: given relation does not respect defined rules";
  return message;
}

db::FileNotReadable::FileNotReadable(const std::string& filename) throw() :
  m_name(filename)
{
}

db::FileNotReadable::~FileNotReadable() throw() { }

const char* db::FileNotReadable::what() const throw() {
  try {
    boost::format message("Cannot read file '%s'");
    message % m_name;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::FileNotReadable: cannot format, exception raised";
    return emergency;
  }
}

db::ExtensionNotRegistered::ExtensionNotRegistered(const std::string& extension) throw() :
  m_extension(extension) 
{
}

db::ExtensionNotRegistered::~ExtensionNotRegistered() throw() {
}

const char* db::ExtensionNotRegistered::what() const throw() {
  try {
    boost::format message("Cannot find codec that matches file extension '%s'");
    message % m_extension;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::ExtensionNotRegistered: cannot format, exception raised";
    return emergency;
  }
}


db::CodecNotFound::CodecNotFound(const std::string& codecname) throw()
  : m_name(codecname)
{
}

db::CodecNotFound::~CodecNotFound() throw() {
}

const char* db::CodecNotFound::what() const throw() {
  try {
    boost::format message("Cannot find codec named '%s'");
    message % m_name;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "database::CodecNotFound: cannot format, exception raised";
    return emergency;
  }
}
