/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 19 Sep 2012 12:24:28 CEST
 *
 * @brief A scoped lock-unlock mechanism to facilite the use of Python's GIL in
 * C++-Python bindings.
 */

#ifndef BOB_CORE_PYTHON_GIL_H 
#define BOB_CORE_PYTHON_GIL_H

#include <Python.h>

namespace bob { namespace python {

  /**
   * Creates and destroyes the Python GIL. Use this to prefix code that needs
   * to execute in a Python-thread-safe environment, but is not instantiated
   * from Python it self. Examples are asynchronous calls or from threads
   * created from C++.
   */
  class gil {

    public:

      /**
       * Constructor - acquires the lock
       */
      gil ();

      /**
       * Destructor - releases the lock
       */
      ~gil ();

    private:

      PyGILState_STATE m_lock; ///< the lock

  };

  /**
   * Unlocks the Python GIL
   */
  class no_gil {

    public:

      /**
       * Releases the Python GIL lock until the end of the current scope
       */
      no_gil ();

      /**
       * Re-acquires the GIL lock
       */
      ~no_gil ();

    private:

      PyThreadState* m_state;

  };

}}

#endif /* BOB_CORE_PYTHON_GIL_H */
