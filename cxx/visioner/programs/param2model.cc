/**
 * @file visioner/programs/param2model.cc
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

#include "core/logging.h"

#include "visioner/model/mdecoder.h"

int main(int argc, char *argv[]) {

  bob::visioner::param_t param;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  param.add_options(po_desc);
  po_desc.add_options()
    ("model", boost::program_options::value<std::string>(),
     "model");

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !param.decode(po_desc, po_vm) ||
      !po_vm.count("model"))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const bob::visioner::string_t cmd_model = po_vm["model"].as<std::string>();

  const bob::visioner::rmodel_t model = bob::visioner::make_model(param);
  if (model->save(cmd_model) == false)
  {
    bob::core::error
      << "Failed to save the model <" << cmd_model << ">!" << std::endl;
  }

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  exit(EXIT_SUCCESS);

}
