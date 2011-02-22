/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements a Portable Pixelmap (PPM) P6 image format reader/writer
 *
 * This codec will only be able to work with three-dimension input.
 */

#include "database/PPMImageArrayCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <unistd.h>

namespace db = Torch::database;

namespace Torch { namespace database {
  class PPMImageException: public Torch::core::Exception { };
}}

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::PPMImageArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::PPMImageArrayCodec::PPMImageArrayCodec()
  : m_name("torch.image.ppm_p6"),
    m_extensions()
{ 
  m_extensions.push_back(".ppm");
}

db::PPMImageArrayCodec::~PPMImageArrayCodec() { }

void db::PPMImageArrayCodec::parseHeader( std::ifstream& ifile, 
  size_t& height, size_t& width) const
{
  const size_t buffer_size = 200;
  char buffer[buffer_size];

  size_t dim[2];
  // Read magic number P6
  ifile.getline(buffer, buffer_size);
  std::string str(buffer);
  if( str.compare("P6") ) throw db::PPMImageException();

  // Read comments
  ifile.getline(buffer, buffer_size);
  if(ifile.gcount() == 0) throw db::PPMImageException();
  while(buffer[0] == '#' || buffer[0] == '\n') {  
    ifile.getline(buffer, buffer_size);
    if(ifile.gcount() == 0) throw db::PPMImageException();
  }

  // Read width and height
  std::string str_tok(buffer);
  boost::char_separator<char> Separator(" \n");
  boost::tokenizer< boost::char_separator<char> > 
    tok( str_tok, Separator );
  size_t i=0;
  // Convert string to int
  for(boost::tokenizer< boost::char_separator<char> >::const_iterator
      it=tok.begin(); it!=tok.end(); ++it, ++i)
    dim[i] = atoi((*it).c_str());
  // Read height on the next line if required
  if(i==1) {
    ifile.getline(buffer, buffer_size);
    if(ifile.gcount() == 0) throw db::PPMImageException();
    dim[1] = atoi(buffer);
  } else if(i==2) {
    width = dim[0]; 
    height = dim[1];
  } else
    throw db::PPMImageException();

  // Read the "maxval" number 
  ifile.getline(buffer, buffer_size);
  if(ifile.gcount() == 0) throw db::PPMImageException();
  if( atoi(buffer)!=255) throw db::PPMImageException();
}

void db::PPMImageArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const 
{
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);

  // Get the with and the height
  db::PPMImageArrayCodec::parseHeader( ifile, shape[1], shape[2]);
  
  // Set other attributes
  eltype = Torch::core::array::t_uint8;
  shape[0] = 3;
  ndim = 3;
     
  // Close the file 
  ifile.close();
}

db::detail::InlinedArrayImpl 
db::PPMImageArrayCodec::load(const std::string& filename) const {
  std::ifstream ifile(filename.c_str(), std::ios::binary|std::ios::in);
  if (!ifile) throw db::FileNotReadable(filename);

  // Get the with and the height
  size_t shape[2];
  db::PPMImageArrayCodec::parseHeader( ifile, shape[0], shape[1]);
  // Get the with and the height
  int height = shape[0];
  int width = shape[1];

  blitz::Array<uint8_t,3> data(3, height, width);
  for (int h=0; h<height; ++h)
    for (int w=0; w<width; ++w)
      for (int c=0; c<3; ++c)
        ifile >> data(c,h,w);

  // Close the file 
  ifile.close();

  return db::detail::InlinedArrayImpl(data);
}

void db::PPMImageArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  //can only save two-dimensional data, so throw if that is not the case
  if (data.getNDim() != 3) throw db::DimensionError(data.getNDim(), 3);

  const size_t *shape = data.getShape();
  if (shape[0] != 3) throw db::PPMImageException();

  std::ofstream ofile(filename.c_str(), std::ios::binary|std::ios::out);

  // Write header
  const std::string s_p6("P6\n");
  ofile.write( s_p6.c_str(), s_p6.size());

  std::string s_wh( boost::lexical_cast<std::string>(shape[2]) );
  s_wh.append(" ");
  s_wh.append( boost::lexical_cast<std::string>(shape[1]) );
  s_wh.append("\n");
  ofile.write( s_wh.c_str(), s_wh.size());

  const std::string s_255("255\n");
  ofile.write( s_255.c_str(), s_255.size());

  int height = shape[1];
  int width = shape[2];
  // Write data
  blitz::Array<uint8_t,3> pix = data.get<uint8_t,3>();
  uint8_t v;
  for (int h=0; h<height; ++h)
    for (int w=0; w<width; ++w)
      for (int c=0; c<3; ++c) {
        v = pix(c,h,w);
        ofile.write( reinterpret_cast<char*>(&v), sizeof(uint8_t) );
      }

  // Close the file
  ofile.close();
}
