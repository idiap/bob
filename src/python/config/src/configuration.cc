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
    .def("has_key", &PythonBindingsConfig::has_key, (arg("self"), arg("key")))
    .def("__contains__", &PythonBindingsConfig::has_key, (arg("self"),arg("key")), "Tests for the existance of a certain key name in the inner dictionary")
    .def("update", &PythonBindingsConfig::update_config, (arg("self"), arg("other")), "Updates teh inner object dictionary with other objects in the given Configuration object. If there is a name clash, the values in the given object are used.")
    .def("update", &PythonBindingsConfig::update_dict, (arg("self"),arg("dict")), "Updates the inner object dictionary with other objects in the given dictionary. If there is a name clash, the values in the given dictionary are used.")
    .def("keys", &PythonBindingsConfig::keys, arg("self"), "Returns all keys in a list")
    .def("values", &PythonBindingsConfig::values, arg("self"), "Returns all values in a list")
    .def("iteritems", &PythonBindingsConfig::iteritems, arg("self"), "Emits an iterator that loops over all (key,value) tuples")
    .def("iterkeys", &PythonBindingsConfig::iterkeys, arg("self"), "Emits an iterator that loops over all keys")
    .def("itervalues", &PythonBindingsConfig::itervalues, arg("self"), "Emits an iterator that loops over all values")
    .def("clear", &PythonBindingsConfig::clear, (arg("self")), "Removes all items from the inner dictionary")
    .def("save", &PythonBindingsConfig::save, (arg("self"), arg("filename")), "Saves the current configuration in the file specified as argument. The encoding to be used will depend on the filename extension and the currently supported configuration codecs.")
    .def("__str__", &PythonBindingsConfig::__str__)
    ;
}
