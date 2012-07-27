#include <fstream>

#include "visioner/model/ipyramid.h"

int main(int argc, char *argv[]) {

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("input", boost::program_options::value<std::string>(), 
     "input image lists")
    ("downscale", boost::program_options::value<double>()->default_value(0.5), 
     "downscale factor [0.10, 1.00)")
    ("output", boost::program_options::value<std::string>(), 
     "output image list")        
    ("output_dir", boost::program_options::value<std::string>(), 
     "output image and ground truth directory");	

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("input") ||
      !po_vm.count("output") ||
      !po_vm.count("output_dir"))
  {
    visioner::log_error("downscaler") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const std::string cmd_input = po_vm["input"].as<std::string>();
  const std::string cmd_output = po_vm["output"].as<std::string>();
  const std::string cmd_output_dir = po_vm["output_dir"].as<std::string>();
  const double cmd_downscale = visioner::range(po_vm["downscale"].as<double>(), 0.10, 1.00);

  // Load the image and ground truth files	
  visioner::strings_t ifiles, gfiles;
  if (	visioner::load_listfiles(cmd_input, ifiles, gfiles) == false ||
      ifiles.empty() || ifiles.size() != gfiles.size())
  {
    visioner::log_error("downscaler") << "Failed to load the input image lists!\n";
    exit(EXIT_FAILURE);
  }

  // Downscale each image ...
  visioner::strings_t ifiles_proc, gfiles_proc;
  visioner::ipscale_t ipscale, ipscale_proc;
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    const std::string ifile_proc = visioner::basename(ifiles[i]) + ".png";
    const std::string gfile_proc = visioner::filename(gfiles[i]);

    visioner::log_info("downscaler") << "Downscaling [" << (i + 1) << "/" << ifiles.size() << "] ...\r";

    // Load the image and the ground truth
    if (	visioner::load(ifiles[i], ipscale.m_image) == false ||
        visioner::Object::load(gfiles[i], ipscale.m_objects) == false)
    {
      visioner::log_error("downscaler") << "Failed to load the image <" << ifiles[i] << ">!\n";
      exit(EXIT_FAILURE);
    }

    // Downscale and save them
    ipscale.scale(cmd_downscale, ipscale_proc);

    const QImage qimage = visioner::convert(ipscale_proc.m_image);
    if (	qimage.save((cmd_output_dir + "/" + ifile_proc).c_str()) == false ||
        visioner::Object::save(cmd_output_dir + "/" + gfile_proc, ipscale_proc.m_objects) == false)
    {
      visioner::log_error("downscaler") 
        << "Failed to save the processed image <" 
        << (cmd_output_dir + "/" + ifile_proc) << ">!\n";
      exit(EXIT_FAILURE);
    }

    // OK                
    ifiles_proc.push_back(ifile_proc);
    gfiles_proc.push_back(gfile_proc);
  }

  // Save the file list with the downscaled images and ground truth
  std::ofstream os(cmd_output.c_str());
  if (os.is_open() == false)
  {
    visioner::log_error("downscaler") << "Cannot open the output file list <" << cmd_output << ">!\n";
    exit(EXIT_FAILURE);
  }

  os << (cmd_output_dir + "/") << "\n";
  for (std::size_t i = 0; i < ifiles.size(); i ++)
  {
    os << ifiles_proc[i] << " # " << gfiles_proc[i] << "\n";
  }
  os.close();

  // OK
  visioner::log_finished();
  exit(EXIT_SUCCESS);

}
