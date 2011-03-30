/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 30 Mar 11:34:22 2011 
 *
 * @brief Implements python bindings to the Torch configuration system 
 */

#include <boost/python.hpp>
#include "config/Configuration.h"

using namespace boost::python;
namespace conf = Torch::config;

/**
 * Provides a dict like API to the Configuration object with some bells and
 * whistles.
 */
class PythonBindingsConfig: public conf::Configuration {
  
  public:

    PythonBindingsConfig(boost::python::dict& other)
      : conf::Configuration() 
    {
      update_dict(other);
    }

    PythonBindingsConfig(const std::string& filename)
      : conf::Configuration(filename) 
    {
    }

    PythonBindingsConfig()
      : conf::Configuration()
    {
    }

    PythonBindingsConfig(const PythonBindingsConfig& other)
      : conf::Configuration(other)
    {
    }

    virtual ~PythonBindingsConfig() {}

    PythonBindingsConfig& operator=(const PythonBindingsConfig& other) {
      conf::Configuration::operator=(other);
      return *this;
    }

    inline size_t __len__() { return len(dict()); }

    inline object __getitem__(const std::string& key) { return dict()[key]; }

    inline object __setitem__(const std::string& key, object value) {
      return dict()[key] = value; 
      return dict()[key];
    }

    inline void __delitem__(const std::string& key) {
      dict()[key].del();
    }

    inline bool has_key(const std::string& key) { return dict().has_key(key); }

    inline void update_config (PythonBindingsConfig& other) {
      update(other);
    }

    inline void update_dict (boost::python::dict& other) {
      dict().update(other); 
    }

    inline object keys () { return dict().keys(); }

    inline object values () { return dict().values(); }

    inline object iteritems() { return dict().iteritems(); }

    inline object iterkeys() { return dict().iterkeys(); }

    inline object itervalues() { return dict().itervalues(); }

    inline void clear() { conf::Configuration::clear(); }

    inline void save(const std::string& filename) { 
      conf::Configuration::save(filename);
    }

    inline object __str__() {
      return str(dict());
    }

};


void bind_config_configuration() {
  class_<PythonBindingsConfig>("Configuration", "The configuration class allows easy access to values encoded in a supported configuration file format using a map/dictionary like interface.", init<const std::string&>(arg("filename"), "Initializes a new Configuration object giving it the filename that contains the configuration items you want to load."))
    .def(init<>("Initializes a new empty Configuration object"))
    .def(init<dict&>(arg("dictionary"), "Initializes a new Configuration object giving the variables already read and placed in a standard python dictionary"))
    .def("__len__", &PythonBindingsConfig::__len__)
    .def("__getitem__", &PythonBindingsConfig::__getitem__)
    .def("__setitem__", &PythonBindingsConfig::__setitem__)
    .def("__delitem__", &PythonBindingsConfig::__delitem__)
    .def("has_key", &PythonBindingsConfig::has_key)
    .def("__contains__", &PythonBindingsConfig::has_key)
    .def("update", &PythonBindingsConfig::update_config)
    .def("update", &PythonBindingsConfig::update_config)
    .def("keys", &PythonBindingsConfig::keys)
    .def("values", &PythonBindingsConfig::values)
    .def("iteritems", &PythonBindingsConfig::iteritems)
    .def("iterkeys", &PythonBindingsConfig::iterkeys)
    .def("itervalues", &PythonBindingsConfig::itervalues)
    .def("clear", &PythonBindingsConfig::clear)
    .def("save", &PythonBindingsConfig::save)
    .def("__str__", &PythonBindingsConfig::__str__)
    ;
}
