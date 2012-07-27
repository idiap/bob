#include "visioner/vision/object.h"

// Parse a .dat CMUProfile ground truth fileb
bool parse(const visioner::string_t& file) {
  // Load file content
  visioner::string_t text;
  if (visioner::load_file(file, text) == false)
  {
    visioner::log_error("readcmuprofile") << "Failed to load <" << file << ">!\n";
    return false;
  }

  const visioner::strings_t lines = visioner::split(text, "\n");
  for (visioner::index_t i = 0; i < lines.size(); i ++)
  {
    const visioner::strings_t tokens = visioner::split(lines[i], "\t {}");
    if (tokens.empty() == true)
    {
      continue;
    }

    const visioner::string_t ifile = tokens[0];
    const visioner::string_t gfile = visioner::basename(ifile) + ".gt";

    visioner::Object object("face", "unknown", "unknown");                
    for (visioner::index_t j = 0; 3 * j < tokens.size() - 3; j ++)
    {
      const visioner::string_t k = tokens[3 * j + 1];
      const visioner::string_t x = tokens[3 * j + 2];
      const visioner::string_t y = tokens[3 * j + 3];

      visioner::string_t new_k;
      if (k == "leye")
      {
        new_k = k;
      }
      else if (k == "reye")
      {
        new_k = k;
      }
      else if (k == "lmouth")
      {
        new_k = "lmc";
      }
      else if (k == "rmouth")
      {
        new_k = "rmc";
      }
      else if (k == "nose")
      {
        new_k = "nose";
      }
      else if (k == "chin")
      {
        new_k = "chin";
      }

      if (new_k.empty() == false)
      {                        
        object.add(visioner::Keypoint(
              new_k, 
              boost::lexical_cast<float>(x),
              boost::lexical_cast<float>(y)));
      }
    }

    visioner::objects_t objects;
    visioner::Object::load(gfile, objects);

    objects.push_back(object);
    visioner::Object::save(gfile, objects); 
  }

  // OK
  return true;
}

int main(int argc, char *argv[]) {	

  const visioner::string_t profile_dat = "testing_profile_ground_truth.dat";
  const visioner::string_t frontal_dat = "testing_frontal_ground_truth.dat";

  parse(profile_dat);
  parse(frontal_dat);

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;
}
