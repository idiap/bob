#include <QApplication>

#include "visioner/cv/cv_classifier.h"
#include "visioner/cv/cv_draw.h"
#include "visioner/util/timer.h"

namespace visioner = bob::visioner;

int main(int argc, char *argv[]) {	

  QApplication app(argc, argv);
  Q_UNUSED(app);

  visioner::CVDetector detector;
  visioner::CVClassifier classifier;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()	
    ("data", boost::program_options::value<std::string>(), 
     "test datasets")
    ("results", boost::program_options::value<std::string>()->default_value("./"),
     "directory to save images to");	
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
    visioner::log_error("classifier") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();
  const std::string cmd_results = po_vm["results"].as<std::string>();

  // Load the test datasets
  visioner::strings_t ifiles, gfiles;
  if (visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    visioner::log_error("classifier") << "Failed to load the test datasets <" << cmd_data << ">!\n";
    exit(EXIT_FAILURE);
  }

  visioner::Timer timer;

  // Process each image ...
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& ifile = ifiles[i];
    const std::string& gfile = gfiles[i];

    // Load the image and the ground truth
    if (detector.load(ifile, gfile) == false)
    {
      visioner::log_warning("classifier") 
        << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!\n";
      continue;
    }

    timer.restart();

    // Detect objects
    visioner::detections_t detections;
    visioner::bools_t labels;

    detector.scan(detections);              
    detector.label(detections, labels);

    QImage qimage = visioner::draw_gt(detector.ipscale());
    visioner::draw_detections(qimage, detections, detector.param(), labels);

    // Classify objects
    visioner::Object object;
    for (visioner::detections_const_it it = detections.begin(); it != detections.end(); ++ it)
      if (detector.match(*it, object) == true)
      {
        visioner::index_t gt_label = 0, dt_label = 0;
        if (    classifier.classify(object, gt_label) == true && 
            classifier.classify(detector, it->second.first, dt_label) == true)
        {
          visioner::draw_label(qimage, *it, classifier.param(), gt_label, dt_label);
        }          
      }

    qimage.save((cmd_results + "/" + visioner::basename(ifiles[i]) + ".class.png").c_str());                

    visioner::log_info("classifier") 
      << "Image [" << (i + 1) << "/" << ifiles.size() << "]: classified "
      << detector.n_objects() << "/" << detector.stats().m_gts << " GTs in " 
      << timer.elapsed() << "s.\n";
  }

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;
}
