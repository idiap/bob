/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 12:02:35 2011 
 *
 * @brief Shared tools for array indexing
 */

#include "core/python/array_indexing.h"

namespace tp = Torch::python;
namespace bp = boost::python;

#ifdef HAVE_BLITZ_SIZETYPE
int tp::range::fromStart = blitz::fromStart;
int tp::range::toEnd = blitz::toEnd;
#else
int tp::range::fromStart = blitz::Range::fromStart;
int tp::range::toEnd = blitz::Range::toEnd;
#endif

bool tp::check_index(int index, int base, int extent) {
  const int limit = base + extent;
  index = (index<0)? index + limit : index;
  if (index < base) return false;
  if (index >= limit) return false;
  return true;
}

void tp::check_range(int dimension, const blitz::Range& r, int base, int extent) {
  int first = r.first(tp::range::fromStart);
  if ((first != tp::range::fromStart) && !tp::check_index(first, base, extent)) {
    boost::format msg("This blitz array (base: %d) has %d elements on dimension %d, but I got %d for the slice start, that is not invalid");
    msg % base % extent % dimension;
    msg % first;
    PyErr_SetString(PyExc_IndexError, msg.str().c_str());
    bp::throw_error_already_set();
  }
  int last = r.last(tp::range::toEnd);
  if ((last != tp::range::toEnd) && !tp::check_index(last, base, extent)) {
    boost::format msg("This blitz array (base: %d) has %d elements on dimension %d, but I got %d for the slice end, that is not invalid");
    msg % base % extent % dimension;
    msg % last;
    PyErr_SetString(PyExc_IndexError, msg.str().c_str());
    bp::throw_error_already_set();
  }
}

int tp::check_range(int dimension, int index, int base, int extent) {
  if (!tp::check_index(index, base, extent)) {
    boost::format msg("This blitz array (base: %d) has %d elements on dimension %d, but I got %d while you try to index it, that is not invalid");
    msg % base % extent % dimension;
    msg % index;
    PyErr_SetString(PyExc_IndexError, msg.str().c_str());
    bp::throw_error_already_set();
  }
  return (index<0)? index + base + extent : index;
}
