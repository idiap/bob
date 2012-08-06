/**
 * @file visioner/programs/detector_eval.cc
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

#include "core/logging.h"

#include "visioner/cv/cv_detector.h"

int main(int argc, char *argv[]) {	

  bob::visioner::CVDetector detector;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("roc", boost::program_options::value<std::string>(),
     "file to save the ROC points");
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
      !po_vm.count("roc") ||
      !detector.decode(po_desc, po_vm))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();
  const std::string cmd_roc = po_vm.count("roc") ? po_vm["roc"].as<std::string>() : "";

  // Load the test datasets
  std::vector<std::string> ifiles, gfiles;
  if (bob::visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    bob::core::error << "Failed to load the test datasets <" << cmd_data << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Build the ROC curve		
  std::vector<double> fas, tars;
  detector.evaluate(ifiles, gfiles, fas, tars);

  // ... and save it to file: TAR + FA
  if (bob::visioner::save_roc(fas, tars, cmd_roc) == false)
  {
    bob::core::error << "Failed to save the ROC points!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  return EXIT_SUCCESS;

}
