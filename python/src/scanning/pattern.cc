/**
 * @file src/scanning/pattern.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Pattern objects to python
 */

#include <boost/python.hpp>

#include "scanning/Pattern.h"

using namespace boost::python;

void bind_scanning_pattern()
{
  class_<Torch::Pattern>("Pattern", 
      init<optional<short, short, short, short, double, short> >())
    .def("copy", &Torch::Pattern::copy)
    .def("isEqual", &Torch::Pattern::isEqual)
    .def("getOverlap", (int (Torch::Pattern::*)(const Torch::Pattern&, bool) const)&Torch::Pattern::getOverlap)
    //.def("getOverlap_static", (int (Torch::Pattern::*)(int, int, int, int, int, int, int, int, bool))&Torch::Pattern::getOverlap)
    //.staticmethod("getOverlap_static")
    .def("getCenterX", &Torch::Pattern::getCenterX)
    .def("getCenterY", &Torch::Pattern::getCenterY)
    .def_readwrite("x", &Torch::Pattern::m_x)
    .def_readwrite("y", &Torch::Pattern::m_y)
    .def_readwrite("width", &Torch::Pattern::m_w)
    .def_readwrite("height", &Torch::Pattern::m_h)
    .def_readwrite("confidence", &Torch::Pattern::m_confidence)
    .def_readwrite("activation", &Torch::Pattern::m_activation)
    ;

  class_<Torch::PatternList>("PatternList", init<>())
    .def("add", (Torch::Pattern& (Torch::PatternList::*)(const Torch::Pattern&, bool))&Torch::PatternList::add, with_custodian_and_ward_postcall<0, 1, return_internal_reference<> >())
    .def("add", (void (Torch::PatternList::*)(const Torch::PatternList&, bool))&Torch::PatternList::add)
    .def("copy", &Torch::PatternList::copy)
    .def("isEmpty", &Torch::PatternList::isEmpty)
    .def("size", &Torch::PatternList::size)
    .def("capacity", &Torch::PatternList::capacity)
    .def("get", &Torch::PatternList::get, return_internal_reference<>())
    ;
}
