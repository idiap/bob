/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 19:41:59 2011 
 *
 * @brief Some standard exceptions that are thrown from our configuration
 * system.
 */

#ifndef TORCH_CONFIG_EXCEPTION_H 
#define TORCH_CONFIG_EXCEPTION_H

#include <boost/python.hpp>
#include "core/Exception.h"

namespace Torch { namespace config {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class KeyError: public Exception {
    public:
      KeyError(const std::string& key) throw();
      virtual ~KeyError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_key;
      mutable std::string m_message;
  };

  class UnsupportedConversion: public Exception {

    public:
      UnsupportedConversion(const std::string& varname,
          const std::type_info& cxx_type,
          boost::python::object o) throw();
      virtual ~UnsupportedConversion() throw();
      virtual const char* what() const throw();

    private:
      std::string m_varname;
      boost::python::type_info m_typeinfo;
      boost::python::object m_object;
      mutable std::string m_message;

  };

  class PythonError: public Exception {
    public:
      PythonError() throw();
      virtual ~PythonError() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

}}

#endif /* TORCH_CONFIG_EXCEPTION_H */
