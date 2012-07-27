#include "visioner/model/ipyramid.h"
#include "visioner/util/timer.h"

// Compute the face bbx from the eye coordinates
bob::visioner::rect_t eyes2bbx(const bob::visioner::point_t& leye, const bob::visioner::point_t& reye)
{
  static const double D_EYES = 10.0;
  static const double Y_UPPER = 7.0;
  static const double MODEL_WIDTH = 20.0;
  static const double MODEL_HEIGHT = 24.0;

  const double EEx = reye.x() - leye.x();
  const double EEy = reye.y() - leye.y();

  const double c0x = std::min(leye.x(), reye.x()) + 0.5 * EEx;
  const double c0y = std::min(leye.y(), reye.y()) + 0.5 * EEy;

  const double ratio = bob::visioner::my_sqrt(EEx * EEx + EEy * EEy) / D_EYES;

  return bob::visioner::rect_t(c0x - ratio * 0.5 * MODEL_WIDTH, 
      c0y - ratio * Y_UPPER,
      ratio * MODEL_WIDTH, ratio * MODEL_HEIGHT);
}

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input face image lists");	

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
    bob::visioner::log_error("face2bbx") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();

  // Load the face datasets
  bob::visioner::strings_t ifiles, gfiles;
  if (	bob::visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    bob::visioner::log_error("face2bbx") << "Failed to load the face datasets <" << (cmd_input) << ">!\n";
    exit(EXIT_FAILURE);
  }

  // Process each face ...
  bob::visioner::objects_t objects;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& gfile = gfiles[i];

    // Load the image and the round truth		
    if (bob::visioner::Object::load(gfile, objects) == false)
    {
      bob::visioner::log_warning("face2bbx") 
        << "Cannot load the ground truth <" << gfile << ">!\n";
      bob::visioner::objects_t objects;
      bob::visioner::Object::save(gfile, objects);
      continue;
    }
    if (objects.empty() == true)
    {
      continue;
    }

    // Compute the bounding box
    for (bob::visioner::objects_t::iterator it = objects.begin(); it != objects.end(); ++ it)
    {
      bob::visioner::Keypoint leye, reye, eye, ntip;

      if (    it->find("leye", leye) == true &&
          it->find("reye", reye) == true)
      {
        it->move(eyes2bbx(leye.m_point, reye.m_point));
      }                       
      else if (       it->find("eye", eye) == true &&
          it->find("ntip", ntip) == true)
      {                                
        const bob::visioner::point_t& eye_pt = eye.m_point;
        const bob::visioner::point_t& ntip_pt = ntip.m_point;

        const bob::visioner::scalar_t dx = bob::visioner::my_abs(eye_pt.x() - ntip_pt.x());
        const bob::visioner::scalar_t width = dx * 2.5;                                

        it->move(eyes2bbx(
              bob::visioner::point_t(eye_pt.x() - 0.25 * width, eye_pt.y()),
              bob::visioner::point_t(eye_pt.x() + 0.25 * width, eye_pt.y())));
      }
      else
      {
        it->move(bob::visioner::rect_t(-1, -1, -1, -1));
      }
    }

    // Save the new bounding box
    bob::visioner::Object::save(gfile, objects);

    bob::visioner::log_info("face2bbx") 
      << "Image [" << (i + 1) << "/" << ifiles.size() << "]: "
      << objects.size() << " GTs.\n";
  }

  // OK
  bob::visioner::log_finished();
  return EXIT_SUCCESS;

}
