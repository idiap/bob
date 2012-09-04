/**
 * @file visioner/programs/face2bbx.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>

#include "core/logging.h"

#include "visioner/model/ipyramid.h"
#include "visioner/util/timer.h"

// Compute the face bbx from the eye coordinates
QRectF eyes2bbx(const QPointF& leye, const QPointF& reye)
{
  static const double D_EYES = 10.0;
  static const double Y_UPPER = 7.0;
  static const double MODEL_WIDTH = 20.0;
  static const double MODEL_HEIGHT = 24.0;

  const double EEx = reye.x() - leye.x();
  const double EEy = reye.y() - leye.y();

  const double c0x = std::min(leye.x(), reye.x()) + 0.5 * EEx;
  const double c0y = std::min(leye.y(), reye.y()) + 0.5 * EEy;

  const double ratio = std::sqrt(EEx * EEx + EEy * EEy) / D_EYES;

  return QRectF(c0x - ratio * 0.5 * MODEL_WIDTH, 
      c0y - ratio * Y_UPPER,
      ratio * MODEL_WIDTH, ratio * MODEL_HEIGHT);
}

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input face image lists");	

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("input"))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();

  // Load the face datasets
  std::vector<std::string> ifiles, gfiles;
  if (	bob::visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    bob::core::error << "Failed to load the face datasets <" << (cmd_input) << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Process each face ...
  std::vector<bob::visioner::Object> objects;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& gfile = gfiles[i];

    // Load the image and the round truth		
    if (bob::visioner::Object::load(gfile, objects) == false)
    {
      bob::core::warn << "Cannot load the ground truth <" << gfile << ">!" << std::endl;
      std::vector<bob::visioner::Object> objects;
      bob::visioner::Object::save(gfile, objects);
      continue;
    }
    if (objects.empty() == true)
    {
      continue;
    }

    // Compute the bounding box
    for (std::vector<bob::visioner::Object>::iterator it = objects.begin(); it != objects.end(); ++ it) {
      bob::visioner::Keypoint leye, reye, eye, ntip;

      if (    it->find("leye", leye) == true &&
          it->find("reye", reye) == true)
      {
        it->move(eyes2bbx(leye.m_point, reye.m_point));
      }                       
      else if (       it->find("eye", eye) == true &&
          it->find("ntip", ntip) == true)
      {                                
        const QPointF& eye_pt = eye.m_point;
        const QPointF& ntip_pt = ntip.m_point;

        const double dx = std::abs(eye_pt.x() - ntip_pt.x());
        const double width = dx * 2.5;                                

        it->move(eyes2bbx(QPointF(eye_pt.x() - 0.25 * width, eye_pt.y()),
              QPointF(eye_pt.x() + 0.25 * width, eye_pt.y())));
      }
      else
      {
        it->move(QRectF(-1, -1, -1, -1));
      }
    }

    // Save the new bounding box
    bob::visioner::Object::save(gfile, objects);

    bob::core::info 
      << "Image [" << (i + 1) << "/" << ifiles.size() << "]: "
      << objects.size() << " GTs." << std::endl;
  }

  // OK
  bob::core::info << "Program finished successfully" << std::endl;
  return EXIT_SUCCESS;

}
