#include <fstream>
#include "visioner/model/mdecoder.h"

int main(int argc, char *argv[]) {

  visioner::param_t param;

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
    visioner::log_error("param2model") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const visioner::string_t cmd_model = po_vm["model"].as<std::string>();

  const visioner::rmodel_t model = visioner::make_model(param);
  if (model->save(cmd_model) == false)
  {
    visioner::log_error("param2model")
      << "Failed to save the model <" << cmd_model << ">!\n";
  }

  // OK
  visioner::log_finished();
  exit(EXIT_SUCCESS);

}
