/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief IO Arrayset tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DbArrayset Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <vector>

#include <blitz/array.h>
#include <string>
#include "core/logging.h" // for Torch::core::tmpdir()
#include "core/cast.h"
#include "io/BinFile.h"
#include "io/Arrayset.h"

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;

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
  tpl /= "torchtest_core_binformatXXXXXX.hdf5";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),5);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
//  return char_tpl.get();
}

/**
 * Use this static variable to count the check_equal() calls so you can know
 * which of the tests failed. Uncomment the first line in each check_equal()
 * implementation to printout the compared arrays and an instance number.
 */
//static size_t counter = 0;

template<typename T, typename U> 
void check_equal(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), Torch::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
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

BOOST_AUTO_TEST_CASE( dbArrayset_construction_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  boost::shared_ptr<Torch::io::Array> db_a(new Torch::io::Array(a));
  boost::shared_ptr<Torch::io::Array> db_b(new Torch::io::Array(b));
  boost::shared_ptr<Torch::io::Array> db_c(new Torch::io::Array(c));

  // Put these io Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::io::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::io::Arrayset db_Ar(vec);

  // Set and get attributes
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true );
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64 );
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1 );
  const size_t* shape = db_Ar.getShape();
  BOOST_CHECK_EQUAL(shape[0], 4 );
  BOOST_CHECK_EQUAL(db_Ar.size(), 3 );
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0 );

  // Get the content of the io Arrays in blitz Arrays
  blitz::Array<double,1> a_get = db_Ar[0].get<double,1>();
  blitz::Array<double,1> b_get = db_Ar[1].get<double,1>();
  blitz::Array<double,1> c_get = db_Ar[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);

  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), Torch::io::Exception );


  // Copy constructor
  Torch::io::Arrayset db_Ar2(db_Ar);
  
  // Check attributes 
  BOOST_CHECK_EQUAL( db_Ar2.isLoaded(), true );
  BOOST_CHECK_EQUAL( db_Ar2.getElementType(), Torch::core::array::t_float64 );
  BOOST_CHECK_EQUAL( db_Ar2.getNDim(), 1 );
  shape = db_Ar2.getShape();
  BOOST_CHECK_EQUAL( shape[0], 4 );
  BOOST_CHECK_EQUAL( db_Ar2.size(), 3 );
  BOOST_CHECK_EQUAL( db_Ar2.getFilename().compare(""), 0 );

  // Get the content of the io Arrays in blitz Arrays
  a_get = db_Ar2[0].get<double,1>();
  b_get = db_Ar2[1].get<double,1>();
  c_get = db_Ar2[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);
 
  // Assignment
  db_Ar2 = db_Ar;
  
  // Check attributes 
  BOOST_CHECK_EQUAL( db_Ar2.isLoaded(), true );
  BOOST_CHECK_EQUAL( db_Ar2.getElementType(), Torch::core::array::t_float64 );
  BOOST_CHECK_EQUAL( db_Ar2.getNDim(), 1 );
  shape = db_Ar2.getShape();
  BOOST_CHECK_EQUAL( shape[0], 4 );
  BOOST_CHECK_EQUAL( db_Ar2.size(), 3 );
  BOOST_CHECK_EQUAL( db_Ar2.getFilename().compare(""), 0 );
  BOOST_CHECK_EQUAL( db_Ar2.getCodec(), db_Ar.getCodec());

  // Get the content of the io Arrays in blitz Arrays
  a_get = db_Ar2[0].get<double,1>();
  b_get = db_Ar2[1].get<double,1>();
  c_get = db_Ar2[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);
}

BOOST_AUTO_TEST_CASE( dbArrayset_loadsave_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  boost::shared_ptr<Torch::io::Array> db_a(new Torch::io::Array(a));
  boost::shared_ptr<Torch::io::Array> db_b(new Torch::io::Array(b));
  boost::shared_ptr<Torch::io::Array> db_c(new Torch::io::Array(c));

  // Put these io Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::io::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  BOOST_REQUIRE_NO_THROW(Torch::io::Arrayset db_Ar(vec));
  Torch::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);

  // Save the Arrayset to a file
  std::string tmp_file = temp_file();
  BOOST_REQUIRE_NO_THROW(db_Ar.save( tmp_file ));
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare("hdf5.arrayset.binary"), 0);
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), Torch::io::Exception );
  
  // Create an Arrayset from a file and check its properties
  Torch::io::Arrayset db_Ar_read(tmp_file);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), db_Ar_read.isLoaded());
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), db_Ar_read.getElementType());
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), db_Ar_read.getNDim());
  BOOST_CHECK_EQUAL(db_Ar.size(), db_Ar_read.size());
  for( size_t i=0; i<db_Ar.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_Ar.getShape()[i], db_Ar_read.getShape()[i]);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(db_Ar_read.getFilename()), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare(db_Ar_read.getCodec()->name()), 0); 
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar_read.add(g), Torch::io::Exception );

  db_Ar_read.load();
  BOOST_CHECK_EQUAL(db_Ar_read.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar_read.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar_read.getCodec().use_count(), 0);  
}

BOOST_AUTO_TEST_CASE( dbArrayset_cast_remove_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  boost::shared_ptr<Torch::io::Array> db_a(new Torch::io::Array(a));
  boost::shared_ptr<Torch::io::Array> db_b(new Torch::io::Array(b));
  boost::shared_ptr<Torch::io::Array> db_c(new Torch::io::Array(c));

  // Put these io Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::io::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);

  // Check the content
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(b, db_Ar[1].get<double,1>());
  check_equal(c, db_Ar[2].get<double,1>());

  // Check that an exception is thrown when accessing a non-existent array
  BOOST_CHECK_THROW(db_Ar[137], Torch::io::IndexError);

  // Check the content when using the cast function
  check_equal(a, db_Ar[0].cast<uint32_t,1>());

  // Remove the second array and check
  db_Ar.remove(1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 2); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());

  // Add a blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());

  // Add a io array and check
  boost::shared_ptr<Torch::io::Array> db_b2(new Torch::io::Array(b));
  db_Ar.add(db_b2);
  BOOST_CHECK_EQUAL(db_Ar.size(), 4); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());

  // Add a blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 5); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());
  check_equal(b, db_Ar[4].get<double,1>());

  // Add a io array and check
  boost::shared_ptr<Torch::io::Array> db_b3(new Torch::io::Array(b));
  db_Ar.add(db_b2);
  BOOST_CHECK_EQUAL(db_Ar.size(), 6); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());
  check_equal(b, db_Ar[4].get<double,1>());
  check_equal(b, db_Ar[5].get<double,1>());
}

BOOST_AUTO_TEST_CASE( dbArrayset_remove_external )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  boost::shared_ptr<Torch::io::Array> db_a(new Torch::io::Array(a));
  boost::shared_ptr<Torch::io::Array> db_b(new Torch::io::Array(b));
  boost::shared_ptr<Torch::io::Array> db_c(new Torch::io::Array(c));

  // Put these io Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::io::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  BOOST_REQUIRE_NO_THROW(Torch::io::Arrayset db_Ar(vec));
  Torch::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);
  
  // Check the content
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( b, db_Ar[1].get<double,1>() );
  check_equal( c, db_Ar[2].get<double,1>() );

  // Save the Arrayset to a file
  std::string tmp_file = temp_file();
  BOOST_REQUIRE_NO_THROW(db_Ar.save( tmp_file ));
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare("hdf5.arrayset.binary"), 0);
  // Check data
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( b, db_Ar[1].get<double,1>() );
  check_equal( c, db_Ar[2].get<double,1>() );

  // Check the content when using the cast function
  check_equal( a, db_Ar[0].cast<uint32_t,1>() );

  // Remove the second array and check
  db_Ar.remove(1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 2); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( c, db_Ar[1].get<double,1>() );

  // Add blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
}

BOOST_AUTO_TEST_SUITE_END()
