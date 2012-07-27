#include "visioner/model/mdecoder.h"

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
    bob::visioner::log_error("model_stats") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();

  // Load the model
  bob::visioner::rmodel_t model;
  if (bob::visioner::Model::load(cmd_input, model) == false)
  {
    bob::visioner::log_error("model_stats") 
      << "Failed to load the model <" << cmd_input << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Display statistics
  bob::visioner::log_info("model_stats")
    << "#outputs = " << model->n_outputs() << ", #n_features = " << model->n_features()
    << ", #n_fvalues = " << model->n_fvalues() << ".\n";
  for (bob::visioner::index_t o = 0; o < model->n_outputs(); o ++)
  {
    bob::visioner::log_info("model_stats")
      << "\toutput <" << (o + 1) << "/" 
      << model->n_outputs() << ">: #luts = " << model->n_luts(o) << "\n";
  }

  bob::visioner::indices_t features;
  for (bob::visioner::index_t o = 0; o < model->n_outputs(); o ++)
  {
    for (bob::visioner::index_t r = 0; r < model->n_luts(o); r ++)
    {
      const bob::visioner::LUT& lut = model->luts()[o][r];
      features.push_back(lut.feature());
    }
  }

  bob::visioner::unique(features);        
  bob::visioner::log_info("model_stats")
    << "#selected features = " << features.size() << ".\n";

  for (bob::visioner::index_t i = 0; i < features.size(); i ++)
  {
    bob::visioner::log_info("model_stats")
      << "feature <" << (i + 1) << "/" 
      << features.size() << ">: " << model->describe(features[i]) << ".\n";
  }

  // OK
  bob::visioner::log_finished();
  exit(EXIT_SUCCESS);

}
