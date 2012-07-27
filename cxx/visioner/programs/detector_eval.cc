#include "visioner/cv/cv_detector.h"

int main(int argc, char *argv[]) {	

  visioner::CVDetector detector;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("roc", boost::program_options::value<std::string>(),
     "file to save the ROC points");
  detector.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !po_vm.count("roc") ||
      !detector.decode(po_desc, po_vm))
  {
    visioner::log_error("detector_eval") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();
  const std::string cmd_roc = po_vm.count("roc") ? po_vm["roc"].as<std::string>() : "";

  // Load the test datasets
  visioner::strings_t ifiles, gfiles;
  if (visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    visioner::log_error("detector_eval") << "Failed to load the test datasets <" << cmd_data << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Build the ROC curve		
  visioner::scalars_t fas, tars;
  detector.evaluate(ifiles, gfiles, fas, tars);

  // ... and save it to file: TAR + FA
  if (visioner::save_roc(fas, tars, cmd_roc) == false)
  {
    visioner::log_error("detector_eval") << "Failed to save the ROC points!\n";
    exit(EXIT_FAILURE);
  }

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;

}
