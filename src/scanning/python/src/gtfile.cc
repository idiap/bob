/**
 * @file src/scanning/gtfile.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "core/File.h"
#include "ip/vision.h"
#include "scanning/GTFile.h"
#include "scanning/bbx2eye19x19deye10GTFile.h"
#include "scanning/bancaGTFile.h"
#include "scanning/cootesGTFile.h"
#include "scanning/eyecenterGTFile.h"
#include "scanning/eyecornerGTFile.h"
#include "scanning/frontalEyeNoseChinGTFile.h"
#include "scanning/halfprofileEyeNoseChinGTFile.h"
#include "scanning/profileEyeNoseChinGTFile.h"

using namespace boost::python;

static list gtfile_getpoints(const Torch::GTFile& gtf) {
  list retval;
  for (unsigned int i=0; i<gtf.getNPoints(); ++i) {
    retval.append(gtf.getPoint(i));
  }
  return retval;
}

void bind_scanning_gtfile()
{
  class_<Torch::GTFile, bases<Torch::File>, boost::noncopyable>("GTFile", "Base Ground Truth file type. Files of this type are used by Normalizers to crop faces from images and videos", no_init)
    .def("load", &Torch::GTFile::load, (arg("self"), arg("file")), "Loads positions for a single face from the input file\n\nReturns \"True\" if we could load the next set of values without problems")
    .def("getName", &Torch::GTFile::getName, (arg("self")), "Returns the name of this type of ground truth file")
    .def("hasLabel", &Torch::GTFile::hasLabel, (arg("self"), arg("label")), "Checks if this ground truth file type has provision for a certain type of Feature")
    .def("getIndex", &Torch::GTFile::getIndex, (arg("self"), arg("label")), "Gets the index for some label (returns -1, if the label does not exist)")
    .def("getNPoints", &Torch::GTFile::getNPoints, (arg("self")), "Returns the total number of ground-truth points available")
    .def("getPoints", &gtfile_getpoints, (arg("self")), "Returns all ground-truth points available")
    .def("getPoint", (const Torch::sPoint2D* (Torch::GTFile::*)(int) const)&Torch::GTFile::getPoint, return_internal_reference<>(), (arg("self"), arg("index")), "Returns the point associated to a particular index if it exists, otherwise None is returned")
    .def("getPoint", (const Torch::sPoint2D* (Torch::GTFile::*)(const char*) const)&Torch::GTFile::getPoint, return_internal_reference<>(), (arg("self"), arg("label")), "Returns the point associated to a particular label if it exists, other None is returned")
    .def("getLabel", &Torch::GTFile::getLabel, (arg("self"), arg("index")), "Returns the label for the point at at certain index")
    ;

  class_<Torch::bbx2eye19x19deye10_GTFile, bases<Torch::GTFile> >("BoundingBoxGTFile", "Ground Truth filetype that calculates eye coordinates based on bounding box coordinates and sizes", init<>("Constructor"))
    .def("load", (bool (Torch::bbx2eye19x19deye10_GTFile::*)(const Torch::Pattern&))&Torch::bbx2eye19x19deye10_GTFile::load, (arg("self"), arg("pattern")), "Loads from a bounding-box (Pattern) without actually reading a file")
    ;

  class_<Torch::bancaGTFile, bases<Torch::GTFile> >("bancaGTFile", "Ground Truth used in Banca files (left eye center and right eye center)", init<>());
  class_<Torch::cootesGTFile, bases<Torch::GTFile> >("cootesGTFile", "Cootes Ground Truth (left eye center and right eye center)", init<>());
  class_<Torch::eyecenterGTFile, bases<Torch::GTFile> >("eyecenterGTFile", "Eye center Ground Truth (left eye center and right eye center)", init<>());
  class_<Torch::eyecornerGTFile, bases<Torch::GTFile> >("eyecornerGTFile", "Eye corners Ground Truth (left and right, outside and inside corners and centers)", init<>());
  class_<Torch::frontalEyeNoseChinGTFile, bases<Torch::GTFile> >("frontalEyeNoseChinGTFile", "Frontal Eye, Nose and Chin Ground Truth (left and right eye corners and centers, nose tip and chin)", init<>());
  class_<Torch::halfprofileEyeNoseChinGTFile, bases<Torch::GTFile> >("halfprofileEyeNoseChinGTFile", "Half Profile Eye, Nose and Chin Ground Truth (left eye corners and centers, right eye center, nose tip and chin)", init<>());
  class_<Torch::profileEyeNoseChinGTFile, bases<Torch::GTFile> >("profileEyeNoseChinGTFile", "Profile Ground Truth (left eye center, nose tip and chin)", init<>());
}
