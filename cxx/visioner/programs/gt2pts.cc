#include <fstream>

#include "visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input face image lists")
    ("points", boost::program_options::value<std::string>(), 
     "facial features points")
    ("output", boost::program_options::value<std::string>(), 
     "output directory to store .pts files");	

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("input") ||
      !po_vm.count("points") ||
      !po_vm.count("output"))
  {
    visioner::log_error("gt2pts") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_points = po_vm["points"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();

  // Load the face datasets
  visioner::strings_t ifiles, gfiles;
  if (	visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    visioner::log_error("gt2pts") << "Failed to load the face datasets <" << (cmd_input) << ">!\n";
    exit(EXIT_FAILURE);
  }

  const visioner::strings_t tokens = visioner::split(cmd_points, ":");

  // Process each face ...
  visioner::objects_t objects;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string& gfile = gfiles[i];

    // Load the ground truth		
    if (visioner::Object::load(gfile, objects) == false)
    {
      visioner::log_warning("gt2pts") 
        << "Cannot load the ground truth <" << gfile << ">!\n";
      continue;
    }

    if (objects.size() != 1)
    {
      continue;
    }

    // Save the .pts annotations
    std::ofstream out((cmd_output + "/" + visioner::basename(gfile) + ".pts").c_str());
    if (out.is_open() == false)
    {
      visioner::log_warning("gt2pts") 
        << "Cannot save the .pts annotations for <" << gfile << ">!\n";
    }

    const visioner::Object& object = objects[0];

    out << "version: 1\n";
    out << "n_points: " << tokens.size() << "\n";
    out << "{\n";

    for (visioner::strings_t::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
    {
      visioner::Keypoint keypoint;                        
      object.find(*it, keypoint);

      out << keypoint.m_point.x() << " " << keypoint.m_point.y() << "\n";
    }

    out << "}\n";

    visioner::log_info("gt2pts") 
      << "Ground truth [" << (i + 1) << "/" << ifiles.size() << "]: processed.\n";
  }

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;

}
