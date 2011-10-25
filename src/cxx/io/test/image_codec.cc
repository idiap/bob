/**
 * @file src/cxx/io/test/image_codec.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief ImageArrayCodec tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ImageArrayCodec Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <blitz/array.h>
#include "core/logging.h"
#include "io/Array.h"
#include "io/ImageArrayCodec.h"

struct T {
  blitz::Array<uint8_t,2> a;
  blitz::Array<uint8_t,3> b;
  blitz::Array<uint16_t,3> c;

  T() {
    a.resize(6,4);
    a = 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24;
    b.resize(3,2,4);
    b = 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24;
    c.resize(3,6,4);
    c = 2;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file(const std::string& ext) {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  std::string filename("torchtest_core_binformatXXXXXX");
  filename.append(ext);
  tpl /= filename;
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(), ext.size());
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), Torch::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,3>& a, const blitz::Array<U,3>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        BOOST_CHECK_EQUAL(a(i,j,k), Torch::core::cast<T>(b(i,j,k)));
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( image_gif ) 
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to gif image
  std::string filename = temp_file(".gif");
  db_b.save( filename);

  // Load from gif image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}

BOOST_AUTO_TEST_CASE( image_bmp )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to bmp image
  std::string filename = temp_file(".bmp");
  db_b.save( filename);

  // Load from bmp image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}

/*
BOOST_AUTO_TEST_CASE( image_jpg )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to jpg image
  std::string filename = temp_file(".jpg");
  db_b.save( filename);

  // Load from jpg image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}
*/

BOOST_AUTO_TEST_CASE( image_pbm )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  check_equal( db_a.get<uint8_t,2>(), a );

  // Save to pbm image
  std::string filename = temp_file(".pbm");
  db_a.save( filename);

  // Load from pbm image
  Torch::io::Array db_a_read( filename);
//  check_equal( db_a_read.get<uint8_t,2>(), a );
}

BOOST_AUTO_TEST_CASE( image_pgm )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  check_equal( db_a.get<uint8_t,2>(), a );

  // Save to pgm image
  std::string filename = temp_file(".pgm");
  db_a.save( filename);

  // Load from pgm image
  Torch::io::Array db_a_read( filename);
  check_equal( db_a_read.get<uint8_t,2>(), a );
}

BOOST_AUTO_TEST_CASE( image_png )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to png image
  std::string filename = temp_file(".png");
  db_b.save( filename);

  // Load from png image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}

BOOST_AUTO_TEST_CASE( image_ppm )
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to ppm image
  std::string filename = temp_file(".ppm");
  db_b.save( filename);

  // Load from ppm image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}

BOOST_AUTO_TEST_CASE( image_tiff ) 
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to tiff image
  std::string filename = temp_file(".tiff");
  db_b.save( filename);

  // Load from tiff image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}

/*
BOOST_AUTO_TEST_CASE( image_xcf ) 
{
  // Prepare io Array from blitz array
  Torch::io::Array db_b(b);
  BOOST_CHECK_EQUAL(db_b.getNDim(), b.dimensions());
  BOOST_CHECK_EQUAL(db_b.getElementType(), Torch::core::array::t_uint8);
  BOOST_CHECK_EQUAL(db_b.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_b.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_b.getCodec().use_count(), 0);
  for(size_t i=0; i<db_b.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_b.getShape()[i], b.extent(i));
  check_equal( db_b.get<uint8_t,3>(), b );

  // Save to xcf image
  std::string filename = temp_file(".xcf");
  db_b.save( filename);

  // Load from xcf image
  Torch::io::Array db_b_read( filename);
  db_b_read.get<uint8_t,3>();
  check_equal( db_b_read.get<uint8_t,3>(), b );
}
*/

BOOST_AUTO_TEST_SUITE_END()
