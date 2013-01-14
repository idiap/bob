/**
 * @file visioner/programs/classifier.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <QApplication>

#include "bob/core/logging.h"

#include "bob/visioner/cv/cv_classifier.h"
#include "bob/visioner/cv/cv_draw.h"
#include "bob/visioner/util/timer.h"

int main(int argc, char *argv[]) {	

  QApplication app(argc, argv);
  Q_UNUSED(app);

  bob::visioner::CVDetector detector;
  bob::visioner::CVClassifier classifier;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()	
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("results", boost::program_options::value<std::string>()->default_value("./"),
     "directory to save images to");	
  detector.add_options(po_desc);
  classifier.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !detector.decode(po_desc, po_vm) ||
      !classifier.decode(po_desc, po_vm))
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
      bob::core::warn
        << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!" << std::endl;
      continue;
    }

    timer.restart();

    // Detect objects
    std::vector<bob::visioner::detection_t> detections;
    std::vector<int> labels;

    detector.scan(detections);              
    detector.label(detections, labels);

    QImage qimage = bob::visioner::draw_gt(detector.ipscale());
    bob::visioner::draw_detections(qimage, detections, detector.param(), labels);

    // Classify objects
    bob::visioner::Object object;
    for (std::vector<bob::visioner::detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++ it)
      if (detector.match(*it, object) == true)
      {
        uint64_t gt_label = 0, dt_label = 0;
        if (    classifier.classify(object, gt_label) == true && 
            classifier.classify(detector, it->second.first, dt_label) == true)
        {
          bob::visioner::draw_label(qimage, *it, classifier.param(), gt_label, dt_label);
        }          
      }

    qimage.save((cmd_results + "/" + bob::visioner::basename(ifiles[i]) + ".class.png").c_str());                

    bob::core::info 
      << "Image [" << (i + 1) << "/" << ifiles.size() << "]: classified "
      << detector.n_objects() << "/" << detector.stats().m_gts << " GTs in " 
      << timer.elapsed() << "s." << std::endl;
  }

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  return EXIT_SUCCESS;
}
