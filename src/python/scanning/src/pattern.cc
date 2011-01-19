/**
 * @file src/scanning/python/src/pattern.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Pattern objects to python
 */

#include <boost/python.hpp>

#include "scanning/Pattern.h"

using namespace boost::python;

void bind_scanning_pattern()
{
  class_<Torch::Pattern>("Pattern", "Patterns represent windows in an image", 
      init<optional<short, short, short, short, double, short> >((arg("x")=0, arg("")=0, arg("width")=0, arg("height")=0, arg("confidence")=0.0, arg("activation")=1), "Constructor"))
    .def("copy", &Torch::Pattern::copy, (arg("self"), arg("other")), "Copies another pattern")
    .def("isEqual", &Torch::Pattern::isEqual, (arg("self"), arg("other"), arg("radius")), "Checks if it has the same subwindow as another pattern")
    .def("getOverlap", (int (Torch::Pattern::*)(const Torch::Pattern&, bool) const)&Torch::Pattern::getOverlap, (arg("self"), arg("other"), arg("ignore_inclusion")), "Returns the percentage of the overlapping area of intersection with another pattern")
    //.def("getOverlap_static", (int (Torch::Pattern::*)(int, int, int, int, int, int, int, int, bool))&Torch::Pattern::getOverlap)
    //.staticmethod("getOverlap_static")
    .def("getCenterX", &Torch::Pattern::getCenterX, arg("self"), "Computes the windows' center 'x' value")
    .def("getCenterY", &Torch::Pattern::getCenterY, arg("self"), "Computes the window's center 'y' value")
    .def_readwrite("x", &Torch::Pattern::m_x)
    .def_readwrite("y", &Torch::Pattern::m_y)
    .def_readwrite("width", &Torch::Pattern::m_w)
    .def_readwrite("height", &Torch::Pattern::m_h)
    .def_readwrite("confidence", &Torch::Pattern::m_confidence)
    .def_readwrite("activation", &Torch::Pattern::m_activation)
    ;

  class_<Torch::PatternList>("PatternList", "Holds many patterns at once", init<>("We always start empty"))
    .def("add", (Torch::Pattern& (Torch::PatternList::*)(const Torch::Pattern&, bool))&Torch::PatternList::add, with_custodian_and_ward_postcall<0, 1, return_internal_reference<> >(), (arg("self"), arg("pattern"), arg("check_for_duplicates")), "Adds another pattern to the end of the list")
    .def("add", (void (Torch::PatternList::*)(const Torch::PatternList&, bool))&Torch::PatternList::add, (arg("self"), arg("list"), arg("check_for_duplicates")), "Adds all the patterns in a given list to the end of the current list.")
    .def("copy", &Torch::PatternList::copy, (arg("self"), arg("list"), arg("check_for_duplicates")), "Copies all the patterns in a given list to the end of the current list.")
    .def("isEmpty", &Torch::PatternList::isEmpty, arg("self"), "Says if we are empty")
    .def("size", &Torch::PatternList::size, arg("self"), "Returns the number of elements in the list")
    .def("capacity", &Torch::PatternList::capacity, arg("self"), "Returns the number of list slots already allocated")
    .def("get", &Torch::PatternList::get, return_internal_reference<>(), (arg("self"), arg("index")), "Access to a specific pattern in the list")
    ;
}
