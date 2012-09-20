/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 19 Sep 2012 13:38:22 CEST
 *
 * @brief Implementation of Python GIL C++ locking
 */

#include <iostream>
#include <pthread.h>
#include "bob/core/python/gil.h"

bob::python::gil::gil () 
  : m_lock(PyGILState_Ensure())
{
}

bob::python::gil::~gil () {
  PyGILState_Release(m_lock);
}

bob::python::no_gil::no_gil()
  : m_state(PyEval_SaveThread())
{
}

bob::python::no_gil::~no_gil() {
  PyEval_RestoreThread(m_state);
}
