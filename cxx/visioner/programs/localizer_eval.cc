#include "visioner/cv/cv_localizer.h"
#include "visioner/util/timer.h"

int main(int argc, char *argv[]) {	

  bob::visioner::CVDetector detector;
  bob::visioner::CVLocalizer localizer;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("loc", boost::program_options::value<std::string>(), 
     "base filename to save the localization results");	
  detector.add_options(po_desc);
  localizer.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !po_vm.count("loc") ||
      !detector.decode(po_desc, po_vm) ||
      !localizer.decode(po_desc, po_vm))
  {
    bob::visioner::log_error("localizer_eval") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();
  const std::string cmd_loc = po_vm["loc"].as<std::string>();

  // Load the test datasets
  bob::visioner::strings_t ifiles, gfiles;
  if (bob::visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    bob::visioner::log_error("localizer_eval") << "Failed to load the test datasets <" << cmd_data << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Build the normalized localization error histograms	
  bob::visioner::Histogram histo;	
  std::vector<bob::visioner::Histogram> histos;

  localizer.evaluate(ifiles, gfiles, detector, histos, histo);

  // Save the histograms and the cumulated histograms
  histo.norm();
  if (histo.save(cmd_loc + ".histo") == false)
  {
    bob::visioner::log_error("localizer_eval") << "Failed to save the localization histogram!\n";
    exit(EXIT_FAILURE);
  }
  histo.cumulate();
  if (histo.save(cmd_loc + ".cum.histo") == false)
  {
    bob::visioner::log_error("localizer_eval") << "Failed to save the localization histogram!\n";
    exit(EXIT_FAILURE);
  }

  for (std::size_t iid = 0; iid < histos.size(); iid ++)
  {
    histos[iid].norm();
    if (histos[iid].save(cmd_loc + ".histo." + 
          boost::lexical_cast<std::string>(localizer.param().m_labels[iid])) == false)
    {
      bob::visioner::log_error("localizer_eval") << "Failed to save the localization histogram!\n";
      exit(EXIT_FAILURE);
    }
    histos[iid].cumulate();
    if (histos[iid].save(cmd_loc + ".cum.histo." + 
          boost::lexical_cast<std::string>(localizer.param().m_labels[iid])) == false)
    {
      bob::visioner::log_error("localizer_eval") << "Failed to save the localization histogram!\n";
      exit(EXIT_FAILURE);
    }
  }

  // OK
  bob::visioner::log_finished();
  return EXIT_SUCCESS;

}
