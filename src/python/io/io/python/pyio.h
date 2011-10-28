/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 11:36:00 2011 
 *
 * @brief Constructions that are useful to this and other modules
 */

#ifndef TORCH_PYTHON_PYIO_H 
#define TORCH_PYTHON_PYIO_H

#include "core/python/pycore.h"
#include "io/python/npyarray.h"

namespace Torch { namespace python {
  
  /**
   * Converts a shared io::buffer to a numpy array in the most clever way
   * possible.
   *
   * The conversion will happen in a "clever" way. Firstly we try a dynamic
   * cast from io::buffer into io::npyarray. If that succeeds, we will directly
   * return the underlying numpy array attached to the buffer given buffer
   * constraints of the underlying memory ownership.
   *
   * If the dynamic cast does not succeed it means the data was generated
   * initially on the C++ side of the running code. In this case we create a
   * new numpy array based on the data pointed by the buffer and tie the
   * lifetime of the buffer to the returned array using trick described here:
   * http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
   *
   * In this case the buffer will be marked read-only to indicate you cannot
   * change its underlying representation - you would need to copy it to do so.
   */
  boost::python::object buffer_object (boost::shared_ptr<Torch::io::buffer> b);

  /**
   * Wraps the buffer with a read-only shallow numpy array layer.
   */
  boost::python::object buffer_object (const Torch::io::buffer& b);

  /**
   * Wraps the buffer with a read-write shallow numpy array layer.
   */
  boost::python::object npyarray_object (npyarray& b);


}}

#endif /* TORCH_PYTHON_PYIO_H */
