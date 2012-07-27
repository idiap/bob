#include "visioner/model/mdecoder.h"

int main(int argc, char *argv[]) {

  bob::visioner::param_t param;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  param.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !param.decode(po_desc, po_vm))
  {
    bob::visioner::log_error("feature_stats") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const bob::visioner::rmodel_t model = bob::visioner::make_model(param);

  bob::visioner::log_info("feature_stats")
    << "The model <" << param.m_feature << "> has " << model->n_features()
    << " features in [0, " << model->n_fvalues() << ").\n";

  // OK
  bob::visioner::log_finished();
  exit(EXIT_SUCCESS);

}
