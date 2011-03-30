/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 17:17:52 2011 
 *
 * @brief Tests the basic retrieval functionality for the configuration
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Configuration Tests
#define BOOST_TEST_MAIN

#include <string>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <blitz/array.h>

#include "core/logging.h"
#include "core/Exception.h"
#include "config/Configuration.h"
#include "database/Array.h"
#include "database/Arrayset.h"

namespace conf = Torch::config;
namespace fs = boost::filesystem;

static fs::path datapath () {
  const char *cpath = getenv("TORCH_TESTDATA_DIR");
  if( !cpath || !strcmp(cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_DIR " 
      << "is not set. " << "Have you setup your working environment " 
      << "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  return fs::path(cpath);
}

BOOST_AUTO_TEST_CASE ( transparent_retrieval ) {
  fs::path file = datapath() / "example1.py";

  /**
   * Constructing a new Configuration object is as easy as passing the name of
   * the file that contains the objects you want to load.
   */
  BOOST_REQUIRE_NO_THROW(conf::Configuration c_try(file.string()));
  conf::Configuration c(file.string());

  BOOST_CHECK_EQUAL(c.size(), 9);

  //the actual configuration parameters have to be there
  BOOST_CHECK_EQUAL(c.has_key("param1"), true);
  BOOST_CHECK_EQUAL(c.has_key("param2"), true);
  BOOST_CHECK_EQUAL(c.has_key("param3"), true);
  BOOST_CHECK_EQUAL(c.has_key("param4"), true);
  BOOST_CHECK_EQUAL(c.has_key("param5"), true);

  /**
   * The other 4 components are the modules "os" and "torch", the temporary
   * variable "curdir" and the module documentation "__doc__". You can retrieve
   * them w/o problems, but for this exercise we are only going to make sure
   * that our parameters are ok and that we can deal with them.
   */

  //Parameter 1 is a string.
  BOOST_CHECK_EQUAL(c.get<std::string>("param1"), "my test string");

  //Parameter 2 is a floating number (close to pi)
  BOOST_CHECK_CLOSE(c.get<float>("param2"), 3.1416, 1e-5);

  //The whole C++ conversion mechanism works out of the box, as long as the
  //C++ object is registered in python. Every time there is a reflection, you
  //can get<>() the C++ object. So, for example, param3 is a
  //blitz::array<int16_t, 3> instantiated from python.
  typedef blitz::Array<int16_t,3> good_type;
  BOOST_REQUIRE_NO_THROW(c.get<good_type&>("param3"));
  good_type& par3 = c.get<good_type&>("param3");
  BOOST_CHECK_EQUAL(par3.extent(0), 2);
  BOOST_CHECK_EQUAL(par3.extent(1), 3);
  BOOST_CHECK_EQUAL(par3.extent(2), 4);
  int16_t v=0;

  //You can proceed the way you want with the array from this point onwards.
  //The following iteration just checks that the values I set in the
  //configuration file are the same.
  for (int i=0; i<par3.extent(0); ++i)
    for (int j=0; j<par3.extent(1); ++j)
      for (int k=0; k<par3.extent(2); ++k)
        BOOST_CHECK_EQUAL(par3(i,j,k), v++);

  //The correctness of the input for the other parameters will not be checked
  //in this test as this is already covered by other tests, like the ones in
  //the database or core packages.
}

BOOST_AUTO_TEST_CASE ( extraction_error ) {
  fs::path file = datapath() / "example1.py";

  /**
   * Constructing a new Configuration object is as easy as passing the name of
   * the file that contains the objects you want to load.
   */
  BOOST_REQUIRE_NO_THROW(conf::Configuration c_try(file.string()));
  conf::Configuration c(file.string());

  //If you try to extract the wrong type of parameter, you will get an
  //exception.
  typedef blitz::Array<float,3> bad_type;
  BOOST_CHECK_THROW(c.get<bad_type&>("param3"), conf::UnsupportedConversion);
}

BOOST_AUTO_TEST_CASE ( syntax_error ) {
  fs::path file = datapath() / "example2.py";
  BOOST_CHECK_THROW(conf::Configuration c(file.string()), conf::PythonError);
}
