/**
 * @file src/cxx/database/test/binary_file.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief BinOutputFile tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BinaryFile Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <blitz/array.h>
#include "core/cast.h"
#include "database/BinFile.h"
#include "database/Array.h"

#include<iostream>

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<uint32_t,1> c;

  blitz::Array<float,2> d;
  blitz::Array<float,2> e;
  blitz::Array<float,2> f;

  blitz::Array<double,4> g;
  blitz::Array<double,4> h;

  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;

    d.resize(2,2);
    d = 1, 2, 3, 4;
    e.resize(2,2);
    e = 5, 6, 7, 8;

    g.resize(2,3,4,5);
    g = 37.;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_core_binformatXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
//  return char_tpl.get();
}

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
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
void check_equal_4d(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  BOOST_REQUIRE_EQUAL(a.extent(3), b.extent(3));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          BOOST_CHECK_EQUAL(a(i,j,k,l), Torch::core::cast<T>(b(i,j,k,l)));
        }
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( blitz1d )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( a);
  out.close();

  Torch::database::BinFile in(tmp_file, Torch::database::BinFile::in);
  blitz::Array<double,1> a_read = in.read<double,1>();

  check_equal_1d( a, a_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz1d_withcast )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( c);
  out.close();

  Torch::database::BinFile in(tmp_file, Torch::database::BinFile::in);
  
  blitz::Array<double,1> c_read = in.read<double,1>();
  check_equal_1d( c, c_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz2d )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( d);
  out.close();

  Torch::database::BinFile in(tmp_file, Torch::database::BinFile::in);
  blitz::Array<float,2> d_read = in.read<float,2>();

  check_equal_2d( d, d_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz1d_inout )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( a);
  out.close();

  Torch::database::BinFile inoutap(tmp_file, Torch::database::BinFile::in | 
    Torch::database::BinFile::out | Torch::database::BinFile::append);
  
  inoutap.write( a);
  inoutap.write( a);
  inoutap.write( a);

  blitz::Array<double,1> a_read = inoutap.read<double,1>(0);
  check_equal_1d( a, a_read);
  inoutap.close();
}

BOOST_AUTO_TEST_CASE( blitz2d_withcast )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( d);
  out.close();

  Torch::database::BinFile in(tmp_file, Torch::database::BinFile::in);
  blitz::Array<uint32_t,2> d_read = in.read<uint32_t,2>();

  check_equal_2d( d, d_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz2d_directaccess )
{
  std::string tmp_file = temp_file();
  Torch::database::BinFile out(tmp_file, Torch::database::BinFile::out);

  out.write( d);
  out.write( e);
  out.write( d);
  out.close();

  Torch::database::BinFile in(tmp_file, Torch::database::BinFile::in);
  blitz::Array<float,2> e_read = in.read<float,2>(1);
  
  check_equal_2d( e, e_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz4d_slice )
{
  std::string tmp_file1 = temp_file();
  Torch::database::BinFile out1(tmp_file1, Torch::database::BinFile::out);
  std::string tmp_file2 = temp_file();
  Torch::database::BinFile out2(tmp_file2, Torch::database::BinFile::out);

  for(int i=0; i<2;++i)
    for(int j=0; j<3;++j)
      for(int k=0; k<4;++k)
        for(int l=0; l<5;++l)
          g(i,j,k,l) = i*3*4*5+j*4*5+k*5+l;

  blitz::Array<double,4> g_sliced1 = g(blitz::Range::all(), blitz::Range(0,0),
    blitz::Range::all(), blitz::Range::all());

  out1.write( g_sliced1);
  out1.close();

  Torch::database::BinFile in1(tmp_file1, Torch::database::BinFile::in);
  
  blitz::Array<double,4> g_sliced1_read = in1.read<double,4>();
  check_equal_4d( g_sliced1, g_sliced1_read);
  in1.close();

  blitz::Array<double,4> g_sliced2 = g(blitz::Range(0,0), blitz::Range::all(),
    blitz::Range::all(), blitz::Range::all());

  out2.write( g_sliced2);
  out2.close();

  Torch::database::BinFile in2(tmp_file2, Torch::database::BinFile::in);
  
  blitz::Array<double,4> g_sliced2_read = in2.read<double,4>();
  check_equal_4d( g_sliced2, g_sliced2_read);
  in1.close();
}

BOOST_AUTO_TEST_CASE( dbArray_creation_blitz )
{
  // Create database Arrays from blitz::arrays and check properties

  // double,1
  Torch::database::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  check_equal_1d( db_a.get<double,1>(), a );

  // float,2
  Torch::database::Array db_d(d);
  BOOST_CHECK_EQUAL(db_d.getNDim(), d.dimensions());
  BOOST_CHECK_EQUAL(db_d.getElementType(), Torch::core::array::t_float32);
  BOOST_CHECK_EQUAL(db_d.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_d.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_d.getCodec().use_count(), 0);
  for(size_t i=0; i<db_d.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_d.getShape()[i], d.extent(i));
  check_equal_2d( db_d.get<float,2>(), d );

  // double,4
  Torch::database::Array db_g(g);
  BOOST_CHECK_EQUAL(db_g.getNDim(), g.dimensions());
  BOOST_CHECK_EQUAL(db_g.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_g.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_g.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_g.getCodec().use_count(), 0);
  for(size_t i=0; i<db_g.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_g.getShape()[i], g.extent(i));
  check_equal_4d( db_g.get<double,4>(), g );
}

BOOST_AUTO_TEST_CASE( dbArray_creation_binaryfile )
{
  // Create a database Array from a blitz::array and save it to a binary file
  Torch::database::Array db_a(a);
  std::string tmp_file = temp_file();
  db_a.save( tmp_file);

  // Create a database Array from a binary file and check its properties
  Torch::database::Array db_a_read(tmp_file);
  BOOST_CHECK_EQUAL(db_a_read.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a_read.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a_read.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a_read.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(
    db_a_read.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a_read.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a_read.getShape()[i], a.extent(i));

  // Load a blitz array from the database Array and check that the values 
  // remain unchanged
  blitz::Array<double,1> bl_read = db_a_read.load<double,1>();
  BOOST_CHECK_EQUAL(db_a_read.isLoaded(), true);
  check_equal_1d( a, bl_read);
}

BOOST_AUTO_TEST_CASE( dbArray_transform_getload )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  
  // Save it to a binary file
  std::string tmp_file = temp_file();
  db_a.save( tmp_file);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));

  // Call the get function and check that properties remain unchanged
  blitz::Array<double,1> a_get = db_a.get<double,1>();
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  // Check that the 'get' array is unchanged
  check_equal_1d( a, a_get);

  // Call the load function and check that properties are updated
  blitz::Array<double,1> a_load = db_a.load<double,1>();
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  // Check that the 'get' array is unchanged
  check_equal_1d( a, a_load);
}

BOOST_AUTO_TEST_CASE( dbArray_transform_move )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  
  // Save it to a binary file
  std::string tmp_file = temp_file();
  db_a.save( tmp_file);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  // Check that the 'get' array is unchanged
  check_equal_1d( a, db_a.get<double,1>());

  // Move it to another binary file
  std::string tmp_file2 = temp_file();
  db_a.save( tmp_file2);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(tmp_file2), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));
  check_equal_1d( a, db_a.get<double,1>());
}

BOOST_AUTO_TEST_CASE( dbArray_cast_inline )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);

  // Call the get function and check that properties remain unchanged
  blitz::Array<uint8_t,1> a_get_uint8 = db_a.get<uint8_t,1>();
  blitz::Array<float,1> a_get_float = db_a.get<float,1>();
  check_equal_1d( a_get_uint8, a_get_float);

  // Create a database Array from a blitz::array
  Torch::database::Array db_g(g);

  // Call the get function and check that properties remain unchanged
  blitz::Array<uint8_t,4> g_get_uint8 = db_g.get<uint8_t,4>();
  blitz::Array<float,4> g_get_float = db_g.get<float,4>();
  check_equal_4d( g_get_uint8, g_get_float);
}

BOOST_AUTO_TEST_CASE( dbArray_cast_external )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);
  // Save it to a binary file
  std::string tmp_file_a = temp_file();
  db_a.save( tmp_file_a);

  // Call the get function and check that properties remain unchanged
  blitz::Array<uint8_t,1> a_get_uint8 = db_a.get<uint8_t,1>();
  blitz::Array<float,1> a_get_float = db_a.get<float,1>();
  check_equal_1d( a_get_uint8, a_get_float);

  // Create a database Array from a blitz::array
  Torch::database::Array db_g(g);
  // Save it to a binary file
  std::string tmp_file_g = temp_file();
  db_a.save( tmp_file_g);

  // Call the get function and check that properties remain unchanged
  blitz::Array<uint8_t,4> g_get_uint8 = db_g.get<uint8_t,4>();
  blitz::Array<float,4> g_get_float = db_g.get<float,4>();
  check_equal_4d( g_get_uint8, g_get_float);
}

BOOST_AUTO_TEST_CASE( dbArray_copy_constructor_inline )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_a.getFilename().size(), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));

  // Test copy constructor
  Torch::database::Array db_a_copy1(db_a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), db_a_copy1.getNDim());
  BOOST_CHECK_EQUAL(db_a.getElementType(), db_a_copy1.getElementType());
  BOOST_CHECK_EQUAL(db_a.isLoaded(), db_a_copy1.isLoaded());
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(
    db_a_copy1.getFilename()), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 
    db_a_copy1.getCodec().use_count());
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], db_a_copy1.getShape()[i]);
  check_equal_1d( db_a.get<double,1>(), db_a_copy1.get<double,1>() );

  // Test copy constructor (assignment)
  Torch::database::Array db_a_copy2 = db_a;
  BOOST_CHECK_EQUAL(db_a.getNDim(), db_a_copy2.getNDim());
  BOOST_CHECK_EQUAL(db_a.getElementType(), db_a_copy2.getElementType());
  BOOST_CHECK_EQUAL(db_a.isLoaded(), db_a_copy2.isLoaded());
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(
    db_a_copy2.getFilename()), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec().use_count(), 
    db_a_copy2.getCodec().use_count());
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], db_a_copy2.getShape()[i]);
  check_equal_1d( db_a.get<double,1>(), db_a_copy2.get<double,1>() );
}

BOOST_AUTO_TEST_CASE( dbArray_copy_constructor_external )
{
  // Create a database Array from a blitz::array
  Torch::database::Array db_a(a);
  std::string tmp_file = temp_file();
  db_a.save( tmp_file);
  BOOST_CHECK_EQUAL(db_a.getNDim(), a.dimensions());
  BOOST_CHECK_EQUAL(db_a.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_a.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_a.getCodec()->name().compare("torch.array.binary"), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], a.extent(i));

  // Test copy constructor
  Torch::database::Array db_a_copy1(db_a);
  BOOST_CHECK_EQUAL(db_a.getNDim(), db_a_copy1.getNDim());
  BOOST_CHECK_EQUAL(db_a.getElementType(), db_a_copy1.getElementType());
  BOOST_CHECK_EQUAL(db_a.isLoaded(), db_a_copy1.isLoaded());
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(
    db_a_copy1.getFilename()), 0);
  BOOST_CHECK_EQUAL(
    db_a.getCodec()->name().compare(db_a_copy1.getCodec()->name()), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], db_a_copy1.getShape()[i]);
  check_equal_1d( db_a.get<double,1>(), db_a_copy1.get<double,1>() );

  // Test copy constructor (assignment)
  Torch::database::Array db_a_copy2 = db_a;
  BOOST_CHECK_EQUAL(db_a.getNDim(), db_a_copy2.getNDim());
  BOOST_CHECK_EQUAL(db_a.getElementType(), db_a_copy2.getElementType());
  BOOST_CHECK_EQUAL(db_a.isLoaded(), db_a_copy2.isLoaded());
  BOOST_CHECK_EQUAL(db_a.getFilename().compare(
    db_a_copy2.getFilename()), 0);
  BOOST_CHECK_EQUAL(
    db_a.getCodec()->name().compare(db_a_copy2.getCodec()->name()), 0);
  for(size_t i=0; i<db_a.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_a.getShape()[i], db_a_copy2.getShape()[i]);
  check_equal_1d( db_a.get<double,1>(), db_a_copy2.get<double,1>() );
}

BOOST_AUTO_TEST_SUITE_END()

