/**
 * @file visioner/programs/trainer.cc
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

#include "visioner/util/timer.h"
#include "visioner/model/mdecoder.h"
#include "visioner/model/sampler.h"

// Train the <model>
static bool train(bob::visioner::Model& model) {
  bob::visioner::Timer timer;

  const bob::visioner::param_t param = model.param();

  // Load the data files        
  timer.restart();
  const bob::visioner::Sampler t_sampler(param, bob::visioner::Sampler::TrainSampler);
  const bob::visioner::Sampler v_sampler(param, bob::visioner::Sampler::ValidSampler); 
  bob::visioner::log_info("trainer") << "timing: loading ~ " << timer.elapsed() << ".\n";

  // Train the model using coarse-to-fine feature projection
  for (bob::visioner::index_t p = 0; p <= param.m_projections; p ++, model.project())
  {
    timer.restart();
    if (bob::visioner::make_trainer(param)->train(t_sampler, v_sampler, model) == false)
    {
      bob::visioner::log_error("trainer") << "Failed to train the model!\n";
      return false;
    }
    bob::visioner::log_info("trainer") << "timing: training ~ " << timer.elapsed() << ".\n";
  }

  // OK
  return true;
}

int main(int argc, char *argv[]) {

  bob::visioner::param_t param;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("model", boost::program_options::value<std::string>(),
     "model");
  param.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") ||
      !po_vm.count("model") ||
      !param.decode(po_desc, po_vm))
  {
    bob::visioner::log_error("trainer") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_model = po_vm["model"].as<std::string>();

  // Train the model	
  bob::visioner::Timer timer;
  bob::visioner::rmodel_t model = make_model(param);
  if (train(*model) == false)
  {
    bob::visioner::log_error("trainer") << "Failed to train the model!\n";
    exit(EXIT_FAILURE);
  }	
  bob::visioner::log_info("trainer") << ">>> Training done in " << timer.elapsed() << "s.\n";	

  // Save the model
  if (model->save(cmd_model) == false)
  {
    bob::visioner::log_error("trainer") << "Failed to save the model!\n";
    exit(EXIT_FAILURE);
  }

  // OK
  bob::visioner::log_finished();
  return EXIT_SUCCESS;

}
