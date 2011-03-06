/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 18:42:30 2011 
 *
 * @brief Handles the static initialization of the Python interpreter.
 */

#ifndef TORCH_CONFIG_PYTHON_H 
#define TORCH_CONFIG_PYTHON_H

#include <boost/shared_ptr.hpp>

namespace Torch { namespace config {

  class Python {

    public:

      /**
       * Destroys the python interpreter
       */
      virtual ~Python();

      /**
       * Gets a shared pointer to the interpreter
       */
      static boost::shared_ptr<Python> instance();

    private: //statics and representation

      /**
       * Initializes the python interpreter
       */
      Python();

      /**
       * Cannot copy construct or assign: not implemented
       */
      Python(const Python& other);
      Python& operator= (const Python& other);

      /**
       * My own pointer
       */
      static boost::shared_ptr<Python> s_instance;

  };

}}

#endif /* TORCH_CONFIG_PYTHON_H */
