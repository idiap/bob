/**
 * @file visioner/programs/downscaler.cc
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
#include <boost/filesystem.hpp>

#include "bob/core/logging.h"

#include "bob/visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input image lists")
    ("downscale", boost::program_options::value<double>()->default_value(0.5), 
     "downscale factor [0.10, 1.00)")
    ("output", boost::program_options::value<std::string>(), 
     "output image list")        
    ("output_dir", boost::program_options::value<std::string>(), 
     "output image and ground truth directory");	

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("input") ||
      !po_vm.count("output") ||
      !po_vm.count("output_dir"))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();
  const std::string cmd_output_dir = po_vm["output_dir"].as<std::string>();
  const double cmd_downscale = bob::visioner::range(po_vm["downscale"].as<double>(), 0.10, 1.00);

  // Load the image and ground truth files	
  std::vector<std::string> ifiles, gfiles;
  if (	bob::visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    bob::core::error << "Failed to load the input image lists!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Downscale each image ...
  std::vector<std::string> ifiles_proc, gfiles_proc;
  bob::visioner::ipscale_t ipscale, ipscale_proc;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string ifile_proc = bob::visioner::basename(ifiles[i]) + ".png";
    const std::string gfile_proc = boost::filesystem::path(gfiles[i]).filename().c_str();

    bob::core::info << "Downscaling [" << (i + 1) << "/" << ifiles.size() << "] ...\r";

    // Load the image and the ground truth
    if (	bob::visioner::load(ifiles[i], ipscale.m_image) == false ||
        bob::visioner::Object::load(gfiles[i], ipscale.m_objects) == false)
    {
      bob::core::error << "Failed to load the image <" << ifiles[i] << ">!" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Downscale and save them
    ipscale.scale(cmd_downscale, ipscale_proc);

    const QImage qimage = bob::visioner::convert(ipscale_proc.m_image);
    if (	qimage.save((cmd_output_dir + "/" + ifile_proc).c_str()) == false ||
        bob::visioner::Object::save(cmd_output_dir + "/" + gfile_proc, ipscale_proc.m_objects) == false)
    {
      bob::core::error 
        << "Failed to save the processed image <" 
        << (cmd_output_dir + "/" + ifile_proc) << ">!" << std::endl;
      exit(EXIT_FAILURE);
    }

    // OK                
    ifiles_proc.push_back(ifile_proc);
    gfiles_proc.push_back(gfile_proc);
  }

  // Save the file list with the downscaled images and ground truth
  std::ofstream os(cmd_output.c_str());
  if (os.is_open() == false)
  {
    bob::core::error << "Cannot open the output file list <" << cmd_output << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  os << (cmd_output_dir + "/") << std::endl;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    os << ifiles_proc[i] << " # " << gfiles_proc[i] << std::endl;
  }
  os.close();

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  exit(EXIT_SUCCESS);

}
