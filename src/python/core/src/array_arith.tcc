/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat 12 Mar 21:17:53 2011 
 *
 * @brief All sorts of array arithmetic operations 
 */

#include "core/python/array_arithmetics.h"

namespace tp = Torch::python;

void bind_array_arith_%type%_%dim% () {
  tp::bind_%mathfun%_arith(tp::%type%_%dim%);
}
