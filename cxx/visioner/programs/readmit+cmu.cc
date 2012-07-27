#include "visioner/vision/object.h"

// Parse a MIT+CMU ground truth file
bool parse(const visioner::string_t& file)
{
  // Load file content
  visioner::string_t text;
  if (visioner::load_file(file, text) == false)
  {
    visioner::log_error("readcmuprofile") << "Failed to load <" << file << ">!\n";
    return false;
  }

  static const visioner::index_t n_points = 6;
  static const visioner::string_t points[n_points] = 
  {
    "leye", "reye", "nose", "lmc", "mc", "rmc"      
  };

  const visioner::strings_t lines = visioner::split(text, "\n");
  for (visioner::index_t i = 0; i < lines.size(); i ++)
  {
    const visioner::strings_t tokens = visioner::split(lines[i], "\t {}");
    if (tokens.size() != 2 * n_points + 1)
    {
      continue;
    }

    const visioner::string_t ifile = tokens[0];
    const visioner::string_t gfile = visioner::basename(ifile) + ".gt";

    visioner::Object object("face", "unknown", "unknown");    

    for (visioner::index_t j = 0; j < n_points; j ++)
    {
      const visioner::string_t x = tokens[2 * j + 1];
      const visioner::string_t y = tokens[2 * j + 2];

      object.add(visioner::Keypoint(
            points[j], 
            boost::lexical_cast<float>(x),
            boost::lexical_cast<float>(y)));
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

  const visioner::string_t input = "annotations";

  parse(input);

  // OK
  visioner::log_finished();
  return EXIT_SUCCESS;
}
