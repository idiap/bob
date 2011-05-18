/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Defines the API for the Configuration functionality
 */

#ifndef TORCH_CONFIG_CONFIGURATION_H
#define TORCH_CONFIG_CONFIGURATION_H

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include "config/Exception.h"
#include <boost/filesystem/path.hpp>

namespace Torch { namespace config {

  /**
   * The Configuration class defines the API that is required for Torch
   * building blocks configuration. 
   */
  class Configuration {

    public:

      /**
       * Builds a new Configuration object starting from an existing file.
       */
      Configuration (const std::string& path);

      /**
       * Copy construct an existing configuration
       */
      Configuration (const Configuration& other);

      /**
       * Starts a new Configuration object with an empty slate
       */
      Configuration ();

      /**
       * Destructor virtualization
       */
      virtual ~Configuration();

      /**
       * Assignment operator: copies everything.
       */
      Configuration& operator= (const Configuration& other);

      /**
       * Saves the configuration to an external file.
       */
      void save (const std::string& path) const;

      /**
       * Merges two configurations together. Items that exist on both get the
       * value of the "other".
       */
      Configuration& update (const Configuration& other);

      /**
       * Gets an element with a certain name. If the given element does not
       * exist, an exception is raised. If the element cannot be cast to the
       * given type T, an exception is also raised.
       */
      template <typename T> inline const T get (const std::string& name) const {
        std::string full_path = getAbsolutePath(name);
        if (!m_dict.has_key(full_path)) throw KeyError(full_path);
        boost::python::extract<T> extractor(m_dict.get(full_path));
        if (!extractor.check())
          throw UnsupportedConversion(full_path, typeid(T), m_dict.get(full_path));
        return extractor();
      }

      template <typename T> inline T get (const std::string& name) {
        std::string full_path = getAbsolutePath(name);
        if (!m_dict.has_key(full_path)) throw KeyError(full_path);
        boost::python::extract<T> extractor(m_dict.get(full_path));
        if (!extractor.check())
          throw UnsupportedConversion(full_path, typeid(T), m_dict.get(full_path));
        return extractor();
      }

      /**
       * Sets the element with a given name, to an arbitrary type. This method
       * should always succeed as long as the type T is supported for
       * conversion.
       */
      template <typename T> inline void set
        (const std::string& name, const T& object) {
        std::string full_path = getAbsolutePath(name);
        m_dict[full_path] = object;
      }

      /**
       * Deletes a certain element by name. If the name does not exist,
       * an exception is raised.
       */
      inline void remove (const std::string& name) { m_dict[name].del(); }

      /**
       * Clears all registered variables
       */
      inline void clear () { m_dict.clear(); }

      /**
       * Returns the number of objects in this configuration database.
       */
      inline size_t size() const { return boost::python::len(m_dict); }

      /**
       * Fills a given stl::iterable such as a vector or a list (must have
       * push_back() implemented) with the names of my existing named objects.
       */
      template <typename T> inline void keys(T& container) const {
        boost::python::list l = m_dict.keys();
        for (Py_ssize_t i=0;i<boost::python::len(l);++i) 
          container.push_back(boost::python::extract<std::string>(l[i]));
      }

      /**
       * Tells us if this Configuration has a certain key
       */
      inline bool has_key (const std::string& name) const {
        std::string full_path = getAbsolutePath(name);
        return m_dict.has_key(full_path);
      }

      /**
       * Change current path.
       * 
       * @param path If path starts with '/', it is treated as an absolute path. '..' and
       * '.' are supported.
       */
      void cd(std::string path);

    protected:

      boost::filesystem::path m_current_path;

      /**
       * Returns an absolute path using the current path. The return string doesn't start
       * with '/' and can directly be used as key for the dictionnary 
       */
      std::string getAbsolutePath(const std::string path) const;
      
      /**
       * Returns the current implementation as a python dictionary. This method
       * is useful if you are writing python bindings to the configuration
       * class and wants to have direct access to the dictionary hold the
       * variables. DO NOT USE THIS OTHERWISE.
       */
      inline boost::python::dict& dict() { return m_dict; }

    private: //representation 
      boost::python::dict m_dict; ///< place where my elements are stored

  };

}}

#endif /* TORCH_CONFIG_CONFIGURATION_H */
