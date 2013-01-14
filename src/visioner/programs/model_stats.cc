/**
 * @file visioner/programs/model_stats.cc
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

#include "bob/core/logging.h"

#include "bob/visioner/model/mdecoder.h"

int main(int argc, char *argv[]) {

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input model file");	

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

  // Load the model
  boost::shared_ptr<bob::visioner::Model> model;
  if (bob::visioner::Model::load(cmd_input, model) == false)
  {
    bob::core::error 
      << "Failed to load the model <" << cmd_input << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Display statistics
  bob::core::info
    << "#outputs = " << model->n_outputs() << ", #n_features = " << model->n_features()
    << ", #n_fvalues = " << model->n_fvalues() << "." << std::endl;
  for (uint64_t o = 0; o < model->n_outputs(); o ++)
  {
    bob::core::info << "\toutput <" << (o + 1) << "/" << model->n_outputs() << ">: #luts = " << model->n_luts(o) << std::endl;
  }

  std::vector<uint64_t> features;
  for (uint64_t o = 0; o < model->n_outputs(); o ++)
  {
    for (uint64_t r = 0; r < model->n_luts(o); r ++)
    {
      const bob::visioner::LUT& lut = model->luts()[o][r];
      features.push_back(lut.feature());
    }
  }

  bob::visioner::unique(features);        
  bob::core::info
    << "#selected features = " << features.size() << "." << std::endl;

  for (uint64_t i = 0; i < features.size(); i ++)
  {
    bob::core::info << "feature <" << (i + 1) << "/" << features.size() << ">: " << model->describe(features[i]) << "." << std::endl;
  }

  // OK
  bob::core::info << "Program finished successfully" << std::endl;
  exit(EXIT_SUCCESS);

}
