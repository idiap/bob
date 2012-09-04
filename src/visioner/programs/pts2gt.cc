/**
 * @file visioner/programs/pts2gt.cc
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
#include <boost/algorithm/string/split.hpp>

#include "core/logging.h"

#include "visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input .pts file")
    ("points", boost::program_options::value<std::string>(), 
     "facial features points")
    ("output", boost::program_options::value<std::string>(), 
     "output .gt file");	

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
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_points = po_vm["points"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();

  std::vector<std::string> tokens;
  boost::split(tokens, cmd_points, boost::is_any_of(":"));

  // Save the .gt annotations ...
  std::ifstream in(cmd_input.c_str());
  if (in.is_open() == false)
  {
    bob::core::error 
      << "Cannot load the .pts annotations <" << cmd_input << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  static char line[1024];
  for (std::size_t i = 0; i < 3; i ++)
  {
    in.getline(line, 1024);
  }

  bob::visioner::Object object("unknown", "unknown", "unknown");
  for (std::vector<std::string>::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
  {
    double x, y;
    in >> x >> y;

    bob::visioner::Keypoint keypoint(*it, x, y);
    object.add(keypoint);
  }              

  if (object.save(cmd_output) == false)
  {
    bob::core::error 
      << "Cannot save the .gt annotations <" << cmd_output << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  return EXIT_SUCCESS;

}
