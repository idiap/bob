/**
 * @file visioner/programs/gt2pts.cc
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

#include "visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input face image lists")
    ("points", boost::program_options::value<std::string>(), 
     "facial features points")
    ("output", boost::program_options::value<std::string>(), 
     "output directory to store .pts files");	

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("input") ||
      !po_vm.count("points") ||
      !po_vm.count("output"))
  {
    bob::visioner::log_error("gt2pts") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_points = po_vm["points"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();

  // Load the face datasets
  bob::visioner::strings_t ifiles, gfiles;
  if (	bob::visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    bob::visioner::log_error("gt2pts") << "Failed to load the face datasets <" << (cmd_input) << ">!\n";
    exit(EXIT_FAILURE);
  }

  const bob::visioner::strings_t tokens = bob::visioner::split(cmd_points, ":");

  // Process each face ...
  bob::visioner::objects_t objects;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& gfile = gfiles[i];

    // Load the ground truth		
    if (bob::visioner::Object::load(gfile, objects) == false)
    {
      bob::visioner::log_warning("gt2pts") 
        << "Cannot load the ground truth <" << gfile << ">!\n";
      continue;
    }

    if (objects.size() != 1)
    {
      continue;
    }

    // Save the .pts annotations
    std::ofstream out((cmd_output + "/" + bob::visioner::basename(gfile) + ".pts").c_str());
    if (out.is_open() == false)
    {
      bob::visioner::log_warning("gt2pts") 
        << "Cannot save the .pts annotations for <" << gfile << ">!\n";
    }

    const bob::visioner::Object& object = objects[0];

    out << "version: 1\n";
    out << "n_points: " << tokens.size() << "\n";
    out << "{\n";

    for (bob::visioner::strings_t::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
    {
      bob::visioner::Keypoint keypoint;                        
      object.find(*it, keypoint);

      out << keypoint.m_point.x() << " " << keypoint.m_point.y() << "\n";
    }

    out << "}\n";

    bob::visioner::log_info("gt2pts") 
      << "Ground truth [" << (i + 1) << "/" << ifiles.size() << "]: processed.\n";
  }

  // OK
  bob::visioner::log_finished();
  return EXIT_SUCCESS;

}
