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
    visioner::log_error("model_stats") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();

  // Load the model
  visioner::rmodel_t model;
  if (visioner::Model::load(cmd_input, model) == false)
  {
    visioner::log_error("model_stats") 
      << "Failed to load the model <" << cmd_input << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Display statistics
  visioner::log_info("model_stats")
    << "#outputs = " << model->n_outputs() << ", #n_features = " << model->n_features()
    << ", #n_fvalues = " << model->n_fvalues() << ".\n";
  for (visioner::index_t o = 0; o < model->n_outputs(); o ++)
  {
    visioner::log_info("model_stats")
      << "\toutput <" << (o + 1) << "/" 
      << model->n_outputs() << ">: #luts = " << model->n_luts(o) << "\n";
  }

  visioner::indices_t features;
  for (visioner::index_t o = 0; o < model->n_outputs(); o ++)
  {
    for (visioner::index_t r = 0; r < model->n_luts(o); r ++)
    {
      const visioner::LUT& lut = model->luts()[o][r];
      features.push_back(lut.feature());
    }
  }

  visioner::unique(features);        
  visioner::log_info("model_stats")
    << "#selected features = " << features.size() << ".\n";

  for (visioner::index_t i = 0; i < features.size(); i ++)
  {
    visioner::log_info("model_stats")
      << "feature <" << (i + 1) << "/" 
      << features.size() << ">: " << model->describe(features[i]) << ".\n";
  }

  // OK
  visioner::log_finished();
  exit(EXIT_SUCCESS);

}
