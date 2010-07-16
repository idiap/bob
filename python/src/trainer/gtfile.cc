/**
 * @file src/trainer/gtfile.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Scanner to python
 */

#include <boost/python.hpp>

#include "core/Object.h"
#include "core/File.h"
#include "ip/vision.h"
#include "trainer/GTFile.h"
#include "trainer/gtfiles/bbx2eye19x19deye10GTFile.h"
#include "trainer/gtfiles/bancaGTFile.h"
#include "trainer/gtfiles/cootesGTFile.h"
#include "trainer/gtfiles/eyecenterGTFile.h"
#include "trainer/gtfiles/eyecornerGTFile.h"
#include "trainer/gtfiles/frontalEyeNoseChinGTFile.h"
#include "trainer/gtfiles/halfprofileEyeNoseChinGTFile.h"
#include "trainer/gtfiles/profileEyeNoseChinGTFile.h"

using namespace boost::python;

static list gtfile_getpoints(const Torch::GTFile& gtf) {
  list retval;
  for (unsigned int i=0; i<gtf.getNPoints(); ++i) {
    retval.append(gtf.getPoint(i));
  }
  return retval;
}

// some very long constant strings
static const char GTFile_docstring[] = "\
Base Ground Truth file type. Files of this type are used by Normalizers to\n\
crop faces from images and videos.";

void bind_trainer_gtfile()
{
  class_<Torch::GTFile, bases<Torch::File>, boost::noncopyable>("GTFile", GTFile_docstring, no_init)
    .def("load", &Torch::GTFile::load)
    .def("getName", &Torch::GTFile::getName)
    .def("hasLabel", &Torch::GTFile::hasLabel)
    .def("getIndex", &Torch::GTFile::getIndex)
    .def("getNPoints", &Torch::GTFile::getNPoints)
    .def("getPoints", &gtfile_getpoints)
    .def("getPoint", (const Torch::sPoint2D* (Torch::GTFile::*)(int) const)&Torch::GTFile::getPoint, return_internal_reference<>())
    .def("getPoint", (const Torch::sPoint2D* (Torch::GTFile::*)(const char*) const)&Torch::GTFile::getPoint, return_internal_reference<>())
    .def("getLabel", &Torch::GTFile::getLabel)
    ;

  class_<Torch::bbx2eye19x19deye10_GTFile, bases<Torch::GTFile> >("BoundingBoxGTFile", init<>())
    .def("load", (bool (Torch::bbx2eye19x19deye10_GTFile::*)(const Torch::Pattern&))&Torch::bbx2eye19x19deye10_GTFile::load)
    ;

  class_<Torch::bancaGTFile, bases<Torch::GTFile> >("bancaGTFile", init<>());
  class_<Torch::cootesGTFile, bases<Torch::GTFile> >("cootesGTFile", init<>());
  class_<Torch::eyecenterGTFile, bases<Torch::GTFile> >("eyecenterGTFile", init<>());
  class_<Torch::eyecornerGTFile, bases<Torch::GTFile> >("eyecornerGTFile", init<>());
  class_<Torch::frontalEyeNoseChinGTFile, bases<Torch::GTFile> >("frontalEyeNoseChinGTFile", init<>());
  class_<Torch::halfprofileEyeNoseChinGTFile, bases<Torch::GTFile> >("halfprofileEyeNoseChinGTFile", init<>());
  class_<Torch::profileEyeNoseChinGTFile, bases<Torch::GTFile> >("profileEyeNoseChinGTFile", init<>());
}
