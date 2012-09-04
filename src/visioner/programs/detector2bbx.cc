/**
 * @file visioner/programs/detector2bbx.cc
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

#include <fstream>

#include "bob/core/logging.h"

#include "bob/visioner/cv/cv_detector.h"
#include "bob/visioner/cv/cv_draw.h"
#include "bob/visioner/util/timer.h"

int main(int argc, char *argv[]) {	

  bob::visioner::CVDetector detector;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("results", boost::program_options::value<std::string>()->default_value("./"),
     "directory to save bounding boxes to");
  detector.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !detector.decode(po_desc, po_vm))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();
  const std::string cmd_results = po_vm["results"].as<std::string>();

  // Load the test datasets
  std::vector<std::string> ifiles, gfiles;
  if (bob::visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    bob::core::error << "Failed to load the test datasets <" << cmd_data << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  bob::visioner::Timer timer;

  // Process each image ...
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& ifile = ifiles[i];
    const std::string& gfile = gfiles[i];

    // Load the image and the ground truth
    if (detector.load(ifile, gfile) == false)
    {
      bob::core::error
        << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!" << std::endl;
      exit(EXIT_FAILURE);
    }

    timer.restart();

    // Detect objects
    std::vector<bob::visioner::detection_t> detections;                
    std::vector<int> labels;

    detector.scan(detections);
    detector.label(detections, labels);

    // Save the bounding boxes of the correct detections
    std::ofstream out((cmd_results + "/" + bob::visioner::basename(ifiles[i]) + ".det.bbx").c_str());
    if (out.is_open() == false)
    {
      continue;
    }

    for (std::size_t d = 0; d < detections.size(); d ++)
    {
      if (labels[d] == true)
      {
        const bob::visioner::detection_t& det = detections[d];
        const QRectF& bbx = det.second.first;
        out << bbx.left() << " " << bbx.top() 
          << " " << bbx.width() << " " << bbx.height() << std::endl;
      }
    }

    bob::core::info 
      << "Image [" << (i + 1) << "/" << ifiles.size() << "]: scanned " 
      << detections.size() << "/" << detector.stats().m_sws << " SWs & "
      << detector.n_objects() << "/" << detector.stats().m_gts << " GTs in " 
      << timer.elapsed() << "s." << std::endl;
  }

  // Display statistics
  detector.stats().show();

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  return EXIT_SUCCESS;

}
