#include <fstream>

#include "visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {	

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input .pts file")
    ("points", boost::program_options::value<std::string>(), 
     "facial features points")
    ("output", boost::program_options::value<std::string>(), 
     "output .gt file");	

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
    visioner::log_error("pts2gt") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_points = po_vm["points"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();

  const visioner::strings_t tokens = visioner::split(cmd_points, ":");

  // Save the .gt annotations ...
  std::ifstream in(cmd_input.c_str());
  if (in.is_open() == false)
  {
    visioner::log_error("pts2gt") 
      << "Cannot load the .pts annotations <" << cmd_input << ">!\n";
    exit(EXIT_FAILURE);
  }

  static char line[1024];
  for (std::size_t i = 0; i < 3; i ++)
  {
    in.getline(line, 1024);
  }

  visioner::Object object("unknown", "unknown", "unknown");
  for (visioner::strings_t::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
  {
    visioner::scalar_t x, y;
    in >> x >> y;

    visioner::Keypoint keypoint(*it, x, y);
    object.add(keypoint);
  }              

  if (object.save(cmd_output) == false)
  {
    visioner::log_error("pts2gt") 
      << "Cannot save the .gt annotations <" << cmd_output << ">!\n";
    exit(EXIT_FAILURE);
  }

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;

}
