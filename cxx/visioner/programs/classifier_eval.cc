#include "visioner/cv/cv_classifier.h"
#include "visioner/util/timer.h"

int main(int argc, char *argv[]) {	

  visioner::CVDetector detector;
  visioner::CVClassifier classifier;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets");
  detector.add_options(po_desc);
  classifier.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !detector.decode(po_desc, po_vm) ||
      !classifier.decode(po_desc, po_vm))
  {
    visioner::log_error("classifier_eval") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();

  // Load the test datasets
  visioner::strings_t ifiles, gfiles;
  if (visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    visioner::log_error("classifier_eval") << "Failed to load the test datasets <" << cmd_data << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Build the confusion matrix
  visioner::index_mat_t hits_mat;
  visioner::indices_t hits_cnt;

  classifier.evaluate(ifiles, gfiles, detector, hits_mat, hits_cnt);

  // Display the confusion matrix
  const visioner::index_t n_classes = classifier.n_classes();
  const visioner::strings_t& labels = classifier.param().m_labels;

  const visioner::index_t str_size = 12;
  const visioner::string_t empty_str = visioner::resize(visioner::string_t(), str_size);

  const visioner::index_t sum_hits = std::accumulate(hits_cnt.begin(), hits_cnt.end(), 0);

  // --- header
  visioner::log_info() << empty_str;
  for (visioner::index_t c1 = 0; c1 < n_classes; c1 ++)
  {
    visioner::log_info() << visioner::resize(labels[c1], str_size);
  }
  visioner::log_info() << visioner::resize("[ERR]", str_size) << "\n";

  // --- content
  visioner::index_t sum_hits_c11 = 0;
  for (visioner::index_t c1 = 0; c1 < n_classes; c1 ++)
  {
    visioner::log_info() << visioner::resize(labels[c1], str_size);

    for (visioner::index_t c2 = 0; c2 < n_classes; c2 ++)
    {
      const visioner::scalar_t c12_dr = 
        100.0 * visioner::inverse(hits_cnt[c1]) * hits_mat(c1, c2);   

      visioner::log_info()
        << visioner::resize(visioner::round(c12_dr, 2), str_size);
    }

    const visioner::scalar_t c1_err = 
      100.0 - 100.0 * visioner::inverse(hits_cnt[c1]) * hits_mat(c1, c1);  
    visioner::log_info()
      << visioner::resize(visioner::round(c1_err, 2), str_size) << "\n";

    sum_hits_c11 += hits_mat(c1, c1);
  }

  // --- end
  const visioner::scalar_t err = 
    100.0 - 100.0 * visioner::inverse(sum_hits) * sum_hits_c11;  
  visioner::log_info()
    << ">>> Average error: " << visioner::round(err, 2) << "%.\n";

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;

}
