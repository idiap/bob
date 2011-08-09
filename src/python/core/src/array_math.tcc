/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  9 Aug 14:18:00 2011 CEST
 *
 * @brief Mathematics operator on arrays 
 */

#include "core/python/array_math.h"

namespace tp = Torch::python;

void bind_array_math_%type%_%dim% () {
  tp::bind_%mathfun%_math(tp::%type%_%dim%);
}
